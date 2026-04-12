#!/usr/bin/env python3
"""
Phase integration smoke script (Phase 6 helper).

Runs import checks and optional HTTP checks against a running API.

Usage:
  PYTHONPATH=src python scripts/verify_phases.py
  PYTHONPATH=src python scripts/verify_phases.py --url http://127.0.0.1:8000

Postgres / Supabase: PYTHONPATH=src python scripts/verify_postgres_db.py (needs GRADE_DATABASE_URL)
Answer key PDFs: PYTHONPATH=src python scripts/verify_key_pdfs.py
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def check_imports() -> bool:
    ok = True
    phases = [
        ("Phase 1 preprocessing", "autograder.preprocessing", "preprocess_pipeline"),
        ("Phase 2 OCR", "autograder.ocr", "ocr_patch"),
        ("Phase 3 DB", "autograder.db", "init_db"),
        ("Phase 3 embeddings", "autograder.embeddings", "embed_text_local"),
        ("Phase 3 scoring", "autograder.scoring", "score_answer_llm"),
        ("Phase 3 key PDF", "autograder.key_pdf", "text_to_upload_request"),
        ("Phase 4 API", "autograder.api", "app"),
        ("Phase 5 PDF", "autograder.report_pdf", "build_evaluation_pdf"),
    ]
    for label, mod, attr in phases:
        try:
            m = __import__(mod, fromlist=[attr])
            getattr(m, attr)
            print(f"[ok] {label}: {mod}.{attr}")
        except Exception as e:
            print(f"[fail] {label}: {e}")
            ok = False
    return ok


def check_http(base: str) -> bool:
    ok = True
    root = base.rstrip("/")
    for path in ("/health", "/api/integrations"):
        url = root + path
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                body = json.loads(r.read().decode())
            print(f"[ok] GET {path} -> keys: {list(body.keys())[:8]}...")
            if path == "/health" and body.get("status") != "ok":
                ok = False
            if path == "/api/integrations" and not body.get("phase_4_api", {}).get("ok"):
                ok = False
        except urllib.error.URLError as e:
            print(f"[fail] HTTP {url}: {e}")
            ok = False
    return ok


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=None, help="Base URL of running API (e.g. http://127.0.0.1:8000)")
    args = p.parse_args()

    if not check_imports():
        return 1
    if args.url and not check_http(args.url):
        return 1
    print("All import checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
