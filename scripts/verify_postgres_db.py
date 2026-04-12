#!/usr/bin/env python3
"""
Smoke-test PostgreSQL / Supabase storage (pgvector) for GRADE.

Requires:
  - GRADE_DATABASE_URL or DATABASE_URL in the environment (or in repo-root ``.env``)
  - ``pip install -e ".[postgres]"``

Usage (from repo root):
  set PYTHONPATH=src
  python scripts/verify_postgres_db.py
  python scripts/verify_postgres_db.py --no-cleanup

Exits 0 on success, 1 on failure.
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_env_file_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if "=" not in line:
        return None
    key, _, val = line.partition("=")
    key = key.strip()
    val = val.strip()
    if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
        val = val[1:-1]
    return key, val


def _load_database_urls_from_repo_env_file() -> None:
    """
    Read GRADE_DATABASE_URL / DATABASE_URL from repo-root .env without ``$`` interpolation.

    python-dotenv expands ``$VAR`` inside values by default, which breaks passwords containing ``$``.
    This runs before dotenv so a URL like ``...postgres:$MyPass$@host...`` stays intact.
    """
    path = _repo_root() / ".env"
    if not path.is_file():
        return
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    for line in text.splitlines():
        parsed = _parse_env_file_line(line)
        if not parsed:
            continue
        key, val = parsed
        if key not in ("GRADE_DATABASE_URL", "DATABASE_URL"):
            continue
        if not val.strip():
            continue
        if not os.environ.get(key):
            os.environ[key] = val


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        print(
            "[warn] python-dotenv is not installed; install with: pip install python-dotenv",
            file=sys.stderr,
        )
        return
    root = _repo_root()
    try:
        load_dotenv(root / ".env", interpolate=False)
        load_dotenv(interpolate=False)
    except TypeError:
        load_dotenv(root / ".env")
        load_dotenv()


def _database_url() -> str:
    return (os.environ.get("GRADE_DATABASE_URL") or os.environ.get("DATABASE_URL") or "").strip()


def _cleanup_pg(exam_id: str, sheet_id: str, result_id: str) -> None:
    from autograder.db import get_conn

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM public.evaluation_results WHERE id = %s", (result_id,))
            cur.execute("DELETE FROM public.answer_keys WHERE exam_id = %s", (exam_id,))
            cur.execute("DELETE FROM public.sheets WHERE id = %s", (sheet_id,))


def main() -> int:
    os.chdir(_repo_root())
    _load_database_urls_from_repo_env_file()
    _load_dotenv()

    parser = argparse.ArgumentParser(description="Verify GRADE Postgres / Supabase DB + pgvector")
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Leave the test sheet, answer key, and result rows in the database",
    )
    args = parser.parse_args()

    url = _database_url()
    if not url:
        env_path = _repo_root() / ".env"
        hint = ""
        if env_path.is_file():
            hint = (
                f" Found {env_path.name} but URL is empty or was stripped. "
                "Passwords with $ confuse some loaders: quote the whole URL in .env, "
                "or run this script after the fix in verify_postgres_db.py (raw .env read). "
            )
        print(
            "[fail] Set GRADE_DATABASE_URL or DATABASE_URL (e.g. in .env). "
            + hint
            + "See .env.example and Supabase Dashboard (Connect).",
            file=sys.stderr,
        )
        return 1

    try:
        import psycopg  # noqa: F401
        import pgvector  # noqa: F401
    except ImportError:
        print("[fail] Install Postgres deps: pip install -e \".[postgres]\"", file=sys.stderr)
        return 1

    sys.path.insert(0, str(_repo_root() / "src"))

    from autograder.db import (
        get_result,
        init_db,
        insert_answer_key,
        insert_evaluation_result,
        insert_sheet,
        list_answer_keys,
        next_answer_key_sort_order_start,
    )
    from autograder.embeddings import EMBEDDING_DIMS, embed_text_local

    print("[info] Using PostgreSQL backend (database URL is set).")
    try:
        init_db()
        print("[ok] init_db()")
    except Exception as e:
        print(f"[fail] init_db(): {e}", file=sys.stderr)
        return 1

    exam_id = f"__verify_pg__{uuid.uuid4().hex[:12]}"
    vec = embed_text_local("canonical model answer for smoke test")

    try:
        sheet_id = insert_sheet("verify_postgres_db.png", "/tmp/verify_postgres_db_placeholder.png")
        print(f"[ok] insert_sheet -> {sheet_id}")

        base = next_answer_key_sort_order_start(exam_id)
        assert base == 0, f"expected sort_order base 0 for new exam, got {base}"
        key_id = insert_answer_key(
            exam_id,
            "Q1",
            "canonical model answer for smoke test",
            vec,
            4.0,
            "general",
            None,
            sort_order=base,
        )
        print(f"[ok] insert_answer_key -> {key_id}")

        keys = list_answer_keys(exam_id)
        assert len(keys) == 1, keys
        emb = keys[0]["embedding"]
        assert len(emb) == EMBEDDING_DIMS, f"embedding length {len(emb)} != {EMBEDDING_DIMS}"
        for i, (a, b) in enumerate(zip(emb, vec)):
            if abs(float(a) - float(b)) > 1e-5:
                raise AssertionError(f"embedding mismatch at index {i}: {a!r} vs {b!r}")
        print(f"[ok] list_answer_keys + vector round-trip ({len(emb)} dims)")

        questions = [
            {
                "question_id": "Q1",
                "student_answer": "smoke",
                "awarded_marks": 2.0,
                "max_marks": 4.0,
                "rubric_scores": {
                    "factual_accuracy": 2,
                    "conceptual_completeness": 2,
                    "reasoning": 2,
                    "domain_terminology": 2,
                },
                "feedback": "verify_postgres_db script",
                "grading_confidence": "high",
                "ocr_confidence": 0.85,
                "flags": [],
            }
        ]
        payload = {
            "sheet_id": sheet_id,
            "exam_id": exam_id,
            "questions": questions,
            "total_marks": 2.0,
            "max_total": 4.0,
            "confidence_flag": False,
            "grading_confidence": "high",
            "ocr_engine_used": "verify_script",
            "prompt_hash": "sha256:verify_postgres_db",
            "llm_model": "none",
            "flags": [],
        }
        result_id = insert_evaluation_result(payload)
        print(f"[ok] insert_evaluation_result -> {result_id}")

        row = get_result(result_id)
        assert row is not None
        assert row["exam_id"] == exam_id
        assert row["sheet_id"] == sheet_id
        assert len(row["questions"]) == 1
        assert row["questions"][0]["question_id"] == "Q1"
        print("[ok] get_result JSON round-trip")

    except AssertionError as e:
        print(f"[fail] assertion: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[fail] {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    if not args.no_cleanup:
        try:
            _cleanup_pg(exam_id, sheet_id, result_id)
            print("[ok] cleanup (evaluation_results, answer_keys, sheets)")
        except Exception as e:
            print(
                f"[warn] cleanup failed (remove rows manually for exam_id={exam_id!r}): {e}",
                file=sys.stderr,
            )

    print("\nAll Postgres DB checks passed.")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass
    raise SystemExit(main())
