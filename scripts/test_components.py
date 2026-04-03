#!/usr/bin/env python3
"""
Exercise GRADE components individually (local checks + optional live calls).

Usage (from repo root):
  PYTHONPATH=src python scripts/test_components.py
  PYTHONPATH=src python scripts/test_components.py --live-ocr path/to/patch.png
  PYTHONPATH=src python scripts/test_components.py --live-gemini

Live flags may incur API usage (Vision / Gemini). Omit them for offline-only checks.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}", flush=True)


def test_embeddings() -> bool:
    from autograder.embeddings import cosine_similarity, embed_text_local, retrieve_top_k

    a = embed_text_local("hello world")
    b = embed_text_local("hello world")
    assert len(a) == 128 and a == b
    assert cosine_similarity(a, b) > 0.99
    z = embed_text_local("")
    assert all(v == 0.0 for v in z)
    cands = [
        {"question_id": "q1", "embedding": embed_text_local("foo"), "expected_answer": "x"},
        {"question_id": "q2", "embedding": embed_text_local("bar baz"), "expected_answer": "y"},
    ]
    top = retrieve_top_k("bar terminology", cands, top_k=1)
    assert top[0][0]["question_id"] == "q2"
    print("[ok] embed_text_local, cosine_similarity, retrieve_top_k", flush=True)
    return True


def test_preprocessing_synthetic() -> bool:
    import numpy as np

    from autograder.preprocessing import preprocess_patch

    patch = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    out = preprocess_patch(patch, size=384)
    assert out.shape == (384, 384)
    print("[ok] preprocess_patch on synthetic grayscale", flush=True)
    return True


def test_scoring_paths() -> bool:
    from autograder.scoring import _legacy_fallback_score, score_answer_llm

    r = _legacy_fallback_score("identical", "identical", 10.0)
    assert r.max_marks == 10.0 and r.awarded_marks >= 9.0
    print("[ok] _legacy_fallback_score (identical texts)", flush=True)

    env_backup = {
        k: os.environ.get(k)
        for k in (
            "GRADE_ENABLE_LLM",
            "GRADE_GEMINI_API_KEY",
            "GEMINI_API_KEY",
            "GRADE_LLM_MOCK_RESPONSE",
        )
    }
    try:
        os.environ.pop("GRADE_ENABLE_LLM", None)
        os.environ.pop("GRADE_GEMINI_API_KEY", None)
        os.environ.pop("GEMINI_API_KEY", None)
        os.environ.pop("GRADE_LLM_MOCK_RESPONSE", None)
        res, label = score_answer_llm(
            student_answer="a",
            model_answer="a",
            max_marks=2.0,
            question_text="Q",
            subject_domain="general",
            ocr_confidence=1.0,
        )
        assert label == "fallback-legacy"
        assert "llm_unavailable" in res.flags
        print("[ok] score_answer_llm with LLM off -> fallback", flush=True)
    finally:
        for k, v in env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    return True


def probe_ocr_env() -> None:
    gac = bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip())
    gv = bool(os.environ.get("GOOGLE_CLOUD_VISION_API_KEY", "").strip())
    az = bool(os.environ.get("AZURE_VISION_ENDPOINT", "").strip()) and bool(
        os.environ.get("AZURE_VISION_KEY", "").strip()
    )
    print(f"  Google service account JSON path set: {gac}")
    print(f"  GOOGLE_CLOUD_VISION_API_KEY set: {gv}")
    print(f"  Azure Vision configured: {az}")
    try:
        import paddleocr  # noqa: F401

        print("  paddleocr: import ok")
    except Exception as e:
        print(f"  paddleocr: not available ({type(e).__name__})")
    try:
        import transformers  # noqa: F401

        print("  transformers: import ok (TrOCR possible)")
    except Exception as e:
        print(f"  transformers: not available ({type(e).__name__})")


def test_live_ocr(image_path: Path) -> bool:
    import cv2
    import numpy as np

    from autograder.ocr import ocr_patch

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[fail] cannot read image: {image_path}")
        return False
    if img.shape[0] != 384 or img.shape[1] != 384:
        img = cv2.resize(img, (384, 384), interpolation=cv2.INTER_AREA)
    patch = np.asarray(img)
    res = ocr_patch(patch, retry_delay=0.0)
    print(f"  engine={res.engine!r} conf={res.confidence:.3f}")
    print(f"  text (first 200 chars): {res.text[:200]!r}")
    print(f"  flags: {res.flags}")
    if res.text.strip():
        print("[ok] live OCR returned non-empty text")
    else:
        print("[warn] live OCR returned empty text (tiers may all have failed)")
    return True


def test_live_gemini() -> bool:
    key = os.environ.get("GRADE_GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not (key and str(key).strip()):
        print("[skip] set GRADE_GEMINI_API_KEY or GEMINI_API_KEY")
        return True
    os.environ["GRADE_ENABLE_LLM"] = "true"
    from autograder.scoring import score_answer_llm

    res, model = score_answer_llm(
        student_answer="The capital of France is Paris.",
        model_answer="Paris is the capital of France.",
        max_marks=5.0,
        question_text="Q1",
        subject_domain="geography",
        ocr_confidence=0.95,
    )
    print(f"  llm_model={model!r} awarded={res.awarded_marks} flags={res.flags}")
    print(f"  feedback (first 120 chars): {res.feedback[:120]!r}...")
    if "gemini_ok" in res.flags or ("gemini_error" not in res.flags and "gemini_key_missing" not in res.flags):
        if "gemini_ok" in res.flags:
            print("[ok] Gemini path succeeded (gemini_ok)")
        else:
            print("[warn] no gemini_ok flag; check feedback for fallback")
    else:
        print("[warn] Gemini error or key issue — see flags/feedback")
    return True


def test_db_roundtrip() -> bool:
    from autograder.db import get_conn, init_db, insert_answer_key, list_answer_keys
    from autograder.embeddings import embed_text_local

    init_db()
    exam = "__component_test_exam__"
    conn = get_conn()
    conn.execute("DELETE FROM answer_keys WHERE exam_id = ?", (exam,))
    conn.commit()
    conn.close()
    vec = embed_text_local("model answer text")
    insert_answer_key(exam, "Q1", "model answer text", vec, 5.0, "general", None)
    keys = list_answer_keys(exam)
    assert len(keys) == 1 and keys[0]["question_id"] == "Q1"
    conn = get_conn()
    conn.execute("DELETE FROM answer_keys WHERE exam_id = ?", (exam,))
    conn.commit()
    conn.close()
    print("[ok] SQLite insert + list_answer_keys + cleanup")
    return True


def main() -> int:
    os.chdir(_repo_root())
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass
    parser = argparse.ArgumentParser(description="Component tests for GRADE")
    parser.add_argument(
        "--live-ocr",
        type=Path,
        metavar="IMAGE",
        default=None,
        help="Run ocr_patch on a grayscale/BGR image (resized to 384). May call cloud APIs.",
    )
    parser.add_argument(
        "--live-gemini",
        action="store_true",
        help="One score_answer_llm call with LLM on (uses GRADE_GEMINI_API_KEY / GEMINI_API_KEY).",
    )
    parser.add_argument("--skip-db", action="store_true", help="Skip SQLite roundtrip")
    args = parser.parse_args()

    ok = True
    section("Embeddings")
    try:
        test_embeddings()
    except Exception as e:
        ok = False
        print(f"[fail] {e}")

    section("Preprocessing")
    try:
        test_preprocessing_synthetic()
    except Exception as e:
        ok = False
        print(f"[fail] {e}")

    section("Scoring (offline paths)")
    try:
        test_scoring_paths()
    except Exception as e:
        ok = False
        print(f"[fail] {e}")

    section("OCR environment")
    probe_ocr_env()

    if args.live_ocr:
        section("Live OCR")
        try:
            test_live_ocr(args.live_ocr)
        except Exception as e:
            ok = False
            print(f"[fail] {e}")

    if args.live_gemini:
        section("Live Gemini")
        try:
            test_live_gemini()
        except Exception as e:
            ok = False
            print(f"[fail] {e}")

    if not args.skip_db:
        section("Database")
        try:
            test_db_roundtrip()
        except Exception as e:
            ok = False
            print(f"[fail] {e}")

    section("pytest (automated suites)")
    import subprocess

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_embeddings_scoring.py",
            "tests/test_preprocessing.py",
            "tests/test_ocr.py",
            "tests/test_key_pdf.py",
            "-q",
            "--tb=no",
        ],
        cwd=_repo_root(),
        env=env,
    )
    if r.returncode != 0:
        ok = False
        print("[fail] one or more pytest modules failed (run with -v for details)")
    else:
        print("[ok] pytest modules passed")

    print("\nDone." + ("" if ok else " Some checks failed."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
