"""
FastAPI backend for GRADE — Phases 3–5.

Endpoints: health, integration status, upload key (JSON body or file), upload sheet,
evaluate, result, rubric, PDF report, sheet file download.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from autograder.db import (
    get_result,
    get_sheet,
    init_db,
    insert_answer_key,
    insert_evaluation_result,
    insert_sheet,
    list_answer_keys,
    next_answer_key_sort_order_start,
)
from autograder.embeddings import embed_text_local, retrieve_top_k
from autograder.key_pdf import pdf_bytes_to_upload_request
from autograder.ocr import (
    google_vision_configured,
    google_vision_credentials_file_ok,
    ocr_patch,
    ocr_patch_consensus,
)
from autograder.preprocessing import PATCH_SIZE, preprocess_pipeline
from autograder.report_pdf import build_evaluation_pdf
from autograder.scoring import prompt_hash, score_answer_llm
from autograder.schemas import (
    EvaluateRequest,
    EvaluateResponse,
    QuestionResult,
    RubricBreakdownResponse,
    UploadKeyRequest,
    UploadKeyResponse,
    UploadSheetResponse,
)

logger = logging.getLogger(__name__)


def _preprocess_options_from_env() -> dict:
    """Patch size, PDF zoom, segmentation padding, optional full-page OCR."""
    patch_size = PATCH_SIZE
    raw_ps = os.environ.get("GRADE_OCR_PATCH_SIZE", "").strip()
    if raw_ps:
        try:
            patch_size = max(128, min(2048, int(raw_ps)))
        except ValueError:
            pass
    pdf_render_scale = 3.0
    raw_z = os.environ.get("GRADE_PDF_RENDER_SCALE", "").strip()
    if raw_z:
        try:
            pdf_render_scale = max(1.0, min(6.0, float(raw_z)))
        except ValueError:
            pass
    full_page = os.environ.get("GRADE_OCR_FULL_PAGE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    bbox_padding_frac = 0.04
    raw_pf = os.environ.get("GRADE_OCR_BBOX_PADDING_FRAC", "").strip()
    if raw_pf:
        try:
            bbox_padding_frac = max(0.0, min(0.2, float(raw_pf)))
        except ValueError:
            pass
    bbox_padding_px_min = 16
    raw_pm = os.environ.get("GRADE_OCR_BBOX_PADDING_PX_MIN", "").strip()
    if raw_pm:
        try:
            bbox_padding_px_min = max(0, min(120, int(raw_pm)))
        except ValueError:
            pass
    return {
        "patch_size": patch_size,
        "pdf_render_scale": pdf_render_scale,
        "full_page": full_page,
        "bbox_padding_frac": bbox_padding_frac,
        "bbox_padding_px_min": bbox_padding_px_min,
    }


def _load_dotenv() -> None:
    """Load `.env`: repo root (always), then current working directory (optional overrides)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    # src/autograder/api.py -> parents[2] is repository root (where pyproject.toml lives).
    repo_root = Path(__file__).resolve().parents[2]
    try:
        load_dotenv(repo_root / ".env", interpolate=False, override=True)
        load_dotenv(interpolate=False)
    except TypeError:
        load_dotenv(repo_root / ".env", override=True)
        load_dotenv()


_load_dotenv()


def _aggregate_grading_confidence(conf_levels: List[str]) -> str:
    if any(c == "low" for c in conf_levels):
        return "low"
    if any(c == "medium" for c in conf_levels):
        return "medium"
    return "high"


def _cors_origins() -> List[str]:
    raw = os.environ.get(
        "GRADE_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000",
    )
    return [o.strip() for o in raw.split(",") if o.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    Path("data/uploads").mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    title="GRADE",
    description="Automatic Handwritten Answer Sheet Evaluator",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    use_pg = bool(
        (os.environ.get("GRADE_DATABASE_URL") or os.environ.get("DATABASE_URL") or "").strip()
    )
    return {
        "status": "ok",
        "phase": "5",
        "version": "v2-ui",
        "integrations": {
            "preprocessing": True,
            "ocr": True,
            "storage_backend": "postgres" if use_pg else "unconfigured",
            "scoring_llm_fallback": True,
            "pdf_report": True,
        },
    }


@app.get("/api/integrations")
def integrations_status():
    """Explicit checklist for Phase 1–6 wiring (useful for QA)."""
    pdf_ok = True
    try:
        import reportlab  # noqa: F401
    except ImportError:
        pdf_ok = False
    pypdf_ok = True
    try:
        import pypdf  # noqa: F401
    except ImportError:
        pypdf_ok = False
    pymupdf_ok = True
    try:
        import fitz  # noqa: F401  # PyMuPDF
    except ImportError:
        pymupdf_ok = False
    llm_flag = os.environ.get("GRADE_ENABLE_LLM", "").lower() in {"1", "true", "yes"}
    gem_key = bool(
        os.environ.get("GRADE_GEMINI_API_KEY", "").strip()
        or os.environ.get("GEMINI_API_KEY", "").strip()
    )
    db_url_set = bool(
        (os.environ.get("GRADE_DATABASE_URL") or os.environ.get("DATABASE_URL") or "").strip()
    )
    pg_deps_ok = True
    if db_url_set:
        try:
            import psycopg  # noqa: F401
            import pgvector  # noqa: F401
        except ImportError:
            pg_deps_ok = False

    gcv_installed = True
    try:
        import google.cloud.vision  # noqa: F401
    except ImportError:
        gcv_installed = False
    g_vision_cfg = google_vision_configured()
    g_vision_file = google_vision_credentials_file_ok()
    ocr_cloud_only = os.environ.get("GRADE_OCR_CLOUD_ONLY", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    ocr_google_only = os.environ.get("GRADE_OCR_GOOGLE_ONLY", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    return {
        "phase_1_preprocess": {"module": "autograder.preprocessing", "ok": True},
        "phase_1_sheet_pdf": {"module": "pymupdf", "installed": pymupdf_ok},
        "phase_2_ocr": {
            "module": "autograder.ocr",
            "ok": True,
            "google_cloud_vision_sdk_installed": gcv_installed,
            "google_vision_configured": g_vision_cfg,
            "google_application_credentials_file_ok": g_vision_file,
            "document_text_detection": True,
            "GRADE_OCR_CLOUD_ONLY": ocr_cloud_only,
            "GRADE_OCR_GOOGLE_ONLY": ocr_google_only,
            "ready_google_vision": gcv_installed and g_vision_cfg,
        },
        "llm_gemini": {
            "GRADE_ENABLE_LLM": llm_flag,
            "api_key_present": gem_key,
            "ready": llm_flag and gem_key,
        },
        "phase_3_db": {
            "module": "autograder.db",
            "backend": "postgres",
            "database_url_set": db_url_set,
            "postgres_deps_installed": pg_deps_ok,
            "ok": bool(db_url_set and pg_deps_ok),
        },
        "phase_3_embeddings": {"module": "autograder.embeddings", "ok": True},
        "phase_3_scoring": {"module": "autograder.scoring", "ok": True},
        "phase_3_key_pdf": {"module": "pypdf", "installed": pypdf_ok},
        "phase_4_api": {"ok": True},
        "phase_5_pdf": {"module": "reportlab", "installed": pdf_ok},
        "phase_5_ui": {"note": "Serve frontend via npm run dev", "ok": True},
        "phase_6": {"script": "scripts/verify_phases.py", "ok": True},
    }


@app.post("/upload/key", response_model=UploadKeyResponse)
def upload_key(payload: UploadKeyRequest):
    if not payload.questions:
        raise HTTPException(status_code=400, detail="questions cannot be empty")

    sort_base = next_answer_key_sort_order_start(payload.exam_id)
    key_ids: List[str] = []
    for i, q in enumerate(payload.questions):
        embedding = embed_text_local(q.expected_answer)
        key_id = insert_answer_key(
            exam_id=payload.exam_id,
            question_id=q.question_id,
            expected_answer=q.expected_answer,
            embedding=embedding,
            max_marks=q.max_marks,
            domain=q.domain,
            rubric_override=q.rubric_override,
            sort_order=sort_base + i,
        )
        key_ids.append(key_id)
    return UploadKeyResponse(key_ids=key_ids)


@app.post("/upload/key/file", response_model=UploadKeyResponse)
async def upload_key_file(
    file: UploadFile = File(...),
    exam_id: str | None = Form(default=None),
    default_max_marks: float = Form(default=4.0, ge=0.1, le=1000.0),
):
    """
    Answer key file: `.json` (UploadKeyRequest) or text-based `.pdf`.

    For PDF, include form field `exam_id` (and optionally `default_max_marks` per question).
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    name = file.filename.lower()
    raw = await file.read()

    if name.endswith(".json"):
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e
        if exam_id and isinstance(data, dict) and not data.get("exam_id"):
            data["exam_id"] = exam_id.strip()
        try:
            payload = UploadKeyRequest.model_validate(data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Schema error: {e}") from e
        return upload_key(payload)

    if name.endswith(".pdf"):
        if not exam_id or not str(exam_id).strip():
            raise HTTPException(
                status_code=400,
                detail="PDF answer keys require form field exam_id (e.g. same exam as your sheet evaluation).",
            )
        try:
            payload = pdf_bytes_to_upload_request(raw, exam_id.strip(), float(default_max_marks))
        except RuntimeError as e:
            raise HTTPException(status_code=501, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return upload_key(payload)

    raise HTTPException(status_code=400, detail="Upload a .json or text-based .pdf answer key file")


@app.post("/upload/sheet", response_model=UploadSheetResponse)
def upload_sheet(file: UploadFile = File(...)):
    """
    Separate upload path for scanned answer sheets (image).
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    suffix = Path(file.filename).suffix or ".png"
    Path("data/uploads").mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="sheet_", suffix=suffix, dir="data/uploads")
    try:
        data = file.file.read()
        Path(tmp_path).write_bytes(data)
    finally:
        os.close(fd)

    sheet_id = insert_sheet(file.filename, tmp_path)
    return UploadSheetResponse(sheet_id=sheet_id, filename=file.filename)


@app.get("/sheet/{sheet_id}/file")
def download_sheet_file(sheet_id: str):
    row = get_sheet(sheet_id)
    if not row:
        raise HTTPException(status_code=404, detail="sheet_id not found")
    path = Path(row["path"])
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Stored file missing on disk")
    return FileResponse(
        path,
        filename=row.get("filename") or path.name,
        media_type="application/octet-stream",
    )


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest):
    sheet = get_sheet(req.sheet_id)
    if not sheet:
        raise HTTPException(status_code=404, detail="sheet_id not found")

    keys = list_answer_keys(req.exam_id)
    if not keys:
        raise HTTPException(status_code=404, detail="No answer keys found for exam_id")

    try:
        pp = preprocess_pipeline(
            sheet["path"],
            expected_num_regions=req.expected_num_regions,
            **_preprocess_options_from_env(),
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=501, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if not pp.patches:
        raise HTTPException(status_code=422, detail="No answer regions detected")

    q_results: List[QuestionResult] = []
    all_flags: List[str] = []
    ocr_engines: List[str] = []
    llm_model = "fallback-legacy"

    try:
        for idx, patch in enumerate(pp.patches):
            ocr_res = (
                ocr_patch_consensus(patch)
                if req.use_consensus_ocr
                else ocr_patch(patch)
            )
            ocr_engines.append(ocr_res.engine)

            if idx < len(keys):
                chosen = keys[idx]
            else:
                top = retrieve_top_k(ocr_res.text, keys, top_k=req.top_k)
                chosen = top[0][0] if top else keys[-1]

            eval_res, llm_model = score_answer_llm(
                student_answer=ocr_res.text,
                model_answer=chosen["expected_answer"],
                max_marks=float(chosen["max_marks"]),
                question_text=chosen["question_id"],
                subject_domain=chosen.get("domain") or "general",
                ocr_confidence=float(ocr_res.confidence),
                rubric_override=chosen.get("rubric_override"),
            )

            flags = list(set((ocr_res.flags or []) + eval_res.flags))
            all_flags.extend(flags)

            q_results.append(
                QuestionResult(
                    question_id=chosen["question_id"],
                    student_answer=ocr_res.text,
                    awarded_marks=eval_res.awarded_marks,
                    max_marks=eval_res.max_marks,
                    rubric_scores=eval_res.rubric_scores,
                    feedback=eval_res.feedback,
                    grading_confidence=eval_res.grading_confidence,
                    ocr_confidence=ocr_res.confidence,
                    flags=flags,
                )
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("evaluate failed sheet=%s exam=%s", req.sheet_id, req.exam_id)
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {type(e).__name__}: {e}",
        ) from e

    total_marks = round(sum(q.awarded_marks for q in q_results), 1)
    max_total = round(sum(q.max_marks for q in q_results), 1)
    grading_conf = _aggregate_grading_confidence([q.grading_confidence for q in q_results])
    confidence_flag = any(q.ocr_confidence < 0.60 for q in q_results) or grading_conf == "low"

    payload = {
        "sheet_id": req.sheet_id,
        "exam_id": req.exam_id,
        "questions": [q.model_dump() for q in q_results],
        "total_marks": total_marks,
        "max_total": max_total,
        "confidence_flag": confidence_flag,
        "grading_confidence": grading_conf,
        "ocr_engine_used": ",".join(sorted(set(ocr_engines))),
        "prompt_hash": prompt_hash(),
        "llm_model": llm_model,
        "flags": sorted(set(all_flags)),
    }
    result_id = insert_evaluation_result(payload)

    return EvaluateResponse(
        result_id=result_id,
        sheet_id=req.sheet_id,
        exam_id=req.exam_id,
        total_marks=total_marks,
        max_total=max_total,
        confidence_flag=confidence_flag,
        grading_confidence=grading_conf,
        prompt_hash=payload["prompt_hash"],
        llm_model=llm_model,
        flags=payload["flags"],
        questions=q_results,
    )


@app.get("/result/{result_id}", response_model=EvaluateResponse)
def get_result_endpoint(result_id: str):
    row = get_result(result_id)
    if not row:
        raise HTTPException(status_code=404, detail="result_id not found")

    questions = [QuestionResult.model_validate(q) for q in row["questions"]]
    return EvaluateResponse(
        result_id=result_id,
        sheet_id=row.get("sheet_id") or "",
        exam_id=row.get("exam_id") or "",
        total_marks=row["total_marks"],
        max_total=row["max_total"],
        confidence_flag=row["confidence_flag"],
        grading_confidence=row["grading_confidence"],
        prompt_hash=row["prompt_hash"],
        llm_model=row["llm_model"],
        flags=row.get("flags") or [],
        questions=questions,
    )


@app.get("/result/{result_id}/rubric", response_model=RubricBreakdownResponse)
def get_rubric_breakdown(result_id: str):
    row = get_result(result_id)
    if not row:
        raise HTTPException(status_code=404, detail="result_id not found")

    questions = [QuestionResult.model_validate(q) for q in row["questions"]]
    totals: Dict[str, int] = {
        "factual_accuracy": 0,
        "conceptual_completeness": 0,
        "reasoning": 0,
        "domain_terminology": 0,
    }
    for q in questions:
        totals["factual_accuracy"] += q.rubric_scores.factual_accuracy
        totals["conceptual_completeness"] += q.rubric_scores.conceptual_completeness
        totals["reasoning"] += q.rubric_scores.reasoning
        totals["domain_terminology"] += q.rubric_scores.domain_terminology

    return RubricBreakdownResponse(
        result_id=result_id,
        dimension_totals=totals,
        questions=questions,
    )


@app.get("/report/{result_id}/pdf")
def report_pdf(result_id: str):
    row = get_result(result_id)
    if not row:
        raise HTTPException(status_code=404, detail="result_id not found")
    row_full = dict(row)
    row_full["id"] = result_id
    try:
        pdf_bytes = build_evaluation_pdf(row_full)
    except RuntimeError as e:
        raise HTTPException(status_code=501, detail=str(e)) from e
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="grade-report-{result_id}.pdf"'},
    )
