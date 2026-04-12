#!/usr/bin/env python3
"""
Parse answer-key PDFs with the same logic as POST /upload/key/file (pypdf + key_pdf).

Default files (repo root):
  docs/Key.pdf
  DocScanner*.pdf (e.g. phone scan; often has no text layer)

Usage:
  PYTHONPATH=src python scripts/verify_key_pdfs.py
  PYTHONPATH=src python scripts/verify_key_pdfs.py path/to/key.pdf

Exit 0 if at least one PDF parses successfully (text-based key).
Exit 1 if a path is missing, extractable text fails to parse, or no PDF produced any questions.
Image-only scans (no text layer) are reported but do not fail the run if another PDF succeeds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_paths(root: Path) -> list[Path]:
    out: list[Path] = []
    p = root / "docs" / "Key.pdf"
    if p.is_file():
        out.append(p)
    out.extend(sorted(root.glob("DocScanner*.pdf")))
    return out


def _try_parse(path: Path, exam_id: str, default_marks: float) -> int:
    from autograder.key_pdf import extract_text_from_pdf_bytes, pdf_bytes_to_upload_request

    data = path.read_bytes()
    raw = extract_text_from_pdf_bytes(data)
    if not raw.strip():
        print(f"  [skip] No extractable text (likely image-only scan). Use a text-based PDF or OCR first.")
        return -1
    req = pdf_bytes_to_upload_request(data, exam_id, default_marks)
    n = len(req.questions)
    print(f"  questions: {n}")
    for q in req.questions:
        preview = " ".join(q.expected_answer.split())[:140]
        if len(q.expected_answer) > 140:
            preview += "..."
        print(f"    {q.question_id} (max {q.max_marks}): {preview}")
    return n


def main() -> int:
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Verify answer-key PDF parsing (GRADE key_pdf)")
    parser.add_argument(
        "pdfs",
        nargs="*",
        type=Path,
        help="PDF paths (default: docs/Key.pdf and DocScanner*.pdf under repo root)",
    )
    parser.add_argument("--exam-id", default="verify-key-pdf", help="Synthetic exam_id for parsing")
    parser.add_argument("--default-marks", type=float, default=4.0)
    args = parser.parse_args()

    paths = [root / p if not p.is_absolute() else p for p in args.pdfs] if args.pdfs else _default_paths(root)
    if not paths:
        print("[fail] No default PDFs found (docs/Key.pdf or DocScanner*.pdf). Pass paths on the command line.", file=sys.stderr)
        return 1

    sys.path.insert(0, str(root / "src"))

    parsed_ok = 0
    for path in paths:
        if not path.is_file():
            print(f"[fail] Not found: {path}", file=sys.stderr)
            continue
        print("=" * 60)
        print(path)
        print("=" * 60)
        try:
            n = _try_parse(path, args.exam_id, args.default_marks)
        except Exception as e:
            print(f"  [fail] {type(e).__name__}: {e}", file=sys.stderr)
            continue
        if n == 0:
            print("  [fail] Parsed to zero questions.", file=sys.stderr)
            continue
        if n > 0:
            parsed_ok += 1

    if parsed_ok == 0:
        print("\n[fail] No PDF produced any questions (check paths, text layer, or scans).", file=sys.stderr)
        return 1
    print("\n[ok] At least one key PDF parsed successfully.")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except (AttributeError, OSError):
        pass
    raise SystemExit(main())
