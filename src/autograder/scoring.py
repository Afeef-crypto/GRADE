from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

from autograder.embeddings import cosine_similarity, embed_text_local
from autograder.schemas import EvaluationResult, RubricScores

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent / "prompts" / "grading_v2.txt"


def prompt_hash() -> str:
    payload = PROMPT_PATH.read_text(encoding="utf-8") if PROMPT_PATH.exists() else ""
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _legacy_fallback_score(student_answer: str, model_answer: str, max_marks: float) -> EvaluationResult:
    s_vec = embed_text_local(student_answer)
    m_vec = embed_text_local(model_answer)
    sim = max(0.0, cosine_similarity(s_vec, m_vec))
    awarded = round(sim * max_marks, 1)
    # map similarity into 0..4 buckets for rubric compatibility
    rub = int(round(min(4.0, sim * 4.0)))
    return EvaluationResult(
        awarded_marks=awarded,
        max_marks=max_marks,
        rubric_scores=RubricScores(
            factual_accuracy=rub,
            conceptual_completeness=rub,
            reasoning=rub,
            domain_terminology=rub,
        ),
        feedback="Fallback scoring used due to LLM unavailability.",
        grading_confidence="low",
        flags=["llm_unavailable", "review_required"],
    )


def _from_json_payload(payload: Dict[str, Any], max_marks: float) -> EvaluationResult:
    # Clamp in case upstream returns slight overflow.
    payload = dict(payload)
    payload["max_marks"] = max_marks
    payload["awarded_marks"] = min(max(0.0, float(payload.get("awarded_marks", 0.0))), max_marks)
    rs = payload.get("rubric_scores")
    if isinstance(rs, dict):
        fixed: Dict[str, int] = {}
        for k in ("factual_accuracy", "conceptual_completeness", "reasoning", "domain_terminology"):
            v = rs.get(k, 0)
            try:
                iv = int(round(float(v)))
            except (TypeError, ValueError):
                iv = 0
            fixed[k] = max(0, min(4, iv))
        payload["rubric_scores"] = fixed
    return EvaluationResult.model_validate(payload)


def _gemini_api_key() -> Optional[str]:
    return os.environ.get("GRADE_GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")


def _parse_llm_json_text(text: str) -> Dict[str, Any]:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()
    return json.loads(text)


def _score_with_gemini(
    *,
    student_answer: str,
    model_answer: str,
    max_marks: float,
    question_text: str,
    subject_domain: str,
    ocr_confidence: float,
    rubric_override: Optional[dict],
    model_id: str,
    api_key: str,
) -> EvaluationResult:
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    role = PROMPT_PATH.read_text(encoding="utf-8") if PROMPT_PATH.exists() else ""
    payload_hint = (
        "Return ONLY JSON with keys: "
        "awarded_marks (number), rubric_scores (object with factual_accuracy, "
        "conceptual_completeness, reasoning, domain_terminology — each integer 0–4), "
        "feedback (string), grading_confidence (exactly one of: high, medium, low), "
        "flags (array of strings). "
        "Do not include max_marks."
    )
    user = f"""{payload_hint}

Question ID: {question_text}
Subject domain: {subject_domain}
Max marks for this question: {max_marks}

Reference / model answer:
{model_answer}

Student answer (from OCR; reported OCR confidence {ocr_confidence:.2f}):
{student_answer}

Rubric override (JSON, or null): {json.dumps(rubric_override) if rubric_override else "null"}
"""
    full_prompt = f"{role}\n\n{user}"
    gm = genai.GenerativeModel(model_id)
    resp = gm.generate_content(
        full_prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.15,
            response_mime_type="application/json",
        ),
    )
    raw = (resp.text or "").strip()
    if not raw:
        raise RuntimeError("Gemini returned empty response")
    data = _parse_llm_json_text(raw)
    return _from_json_payload(data, max_marks)


def score_answer_llm(
    *,
    student_answer: str,
    model_answer: str,
    max_marks: float,
    question_text: str,
    subject_domain: str,
    ocr_confidence: float,
    rubric_override: Optional[dict] = None,
) -> tuple[EvaluationResult, str]:
    """
    Return (EvaluationResult, llm_model).

    If no LLM runtime is configured, fallback scorer is used.
    LLM integration point is environment-driven to keep repo runnable offline.
    """
    llm_on = os.environ.get("GRADE_ENABLE_LLM", "").lower() in {"1", "true", "yes"}
    if not llm_on:
        label = os.environ.get("GRADE_LLM_MODEL", "fallback-legacy")
        return _legacy_fallback_score(student_answer, model_answer, max_marks), label

    model_name = os.environ.get("GRADE_LLM_MODEL", "gemini-2.0-flash")

    # Optional runtime hook: external script writes JSON to env var for deterministic testing.
    raw_json = os.environ.get("GRADE_LLM_MOCK_RESPONSE")
    if raw_json:
        try:
            parsed = json.loads(raw_json)
            return _from_json_payload(parsed, max_marks), model_name
        except Exception:
            return _legacy_fallback_score(student_answer, model_answer, max_marks), model_name

    gkey = _gemini_api_key()
    if gkey:
        try:
            res = _score_with_gemini(
                student_answer=student_answer,
                model_answer=model_answer,
                max_marks=max_marks,
                question_text=question_text,
                subject_domain=subject_domain,
                ocr_confidence=ocr_confidence,
                rubric_override=rubric_override,
                model_id=model_name,
                api_key=gkey,
            )
            flags = list(res.flags)
            if "gemini_ok" not in flags:
                flags.append("gemini_ok")
            res = res.model_copy(update={"flags": flags})
            return res, model_name
        except Exception as exc:
            logger.warning("Gemini grading failed, using embedding fallback: %s", exc)
            fb = _legacy_fallback_score(student_answer, model_answer, max_marks)
            merged_flags = list(dict.fromkeys(list(fb.flags) + ["gemini_error", "review_required"]))
            fb = fb.model_copy(
                update={
                    "feedback": f"{fb.feedback} (Gemini error: {exc})",
                    "flags": merged_flags,
                }
            )
            return fb, model_name

    # LLM on but no Gemini key in process env (often .env not loaded — fix: restart API from repo or set GRADE_GEMINI_API_KEY).
    fb = _legacy_fallback_score(student_answer, model_answer, max_marks)
    fb = fb.model_copy(
        update={
            "feedback": (
                "LLM is enabled but no Gemini API key is set in the API process. "
                "Set GRADE_GEMINI_API_KEY in the repo-root .env and restart uvicorn, or export the variable in the shell."
            ),
            "flags": list(
                dict.fromkeys(
                    [f for f in fb.flags if f != "llm_unavailable"]
                    + ["gemini_key_missing", "review_required"]
                )
            ),
        }
    )
    return fb, model_name
