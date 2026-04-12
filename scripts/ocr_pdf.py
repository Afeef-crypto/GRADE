#!/usr/bin/env python3
"""
Run GRADE preprocessing + OCR on a scanned sheet PDF.

PyMuPDF renders the page → segment → resize patches → ocr_patch (Google Vision upscales/enhances
cloud images unless disabled via env — see .env.example).

Usage (repo root):
  $env:PYTHONPATH = "src"
  python scripts/ocr_pdf.py "path/to/scan.pdf"
  python scripts/ocr_pdf.py "path/to/scan.pdf" --page 3 --patch-size 896 --pdf-zoom 3.5
  python scripts/ocr_pdf.py "path/to/scan.pdf" --page 3 --full-page -o notes.txt
  python scripts/ocr_pdf.py "path/to/scan.pdf" --page 3 --full-page --rag -o extraction.json

Optional:
  --page N             1-based PDF page (default: 1)
  --patch-size N       Segmentation patch square size (default: 768; API uses GRADE_OCR_PATCH_SIZE or 384)
  --pdf-zoom F         PyMuPDF render scale (default: 3.5)
  --inpaint-grey       Try to inpaint flat grey PDF redaction blocks (risky on lined paper)
  --full-page          OCR entire page in one Vision call (fixes clipped sentence endings)
  --bbox-padding F     Fraction to expand each contour box (default: 0.06; use 0 to disable)
  --expected-regions N hint for contour segmentation (same as preprocess_pipeline)
  --rag              Output JSON for RAG (chunked text + metadata; use -o file.json or stdout)
  --chunk-size N     RAG chunk character length (default: 512)
  --chunk-overlap N  RAG overlap (default: 96)
  --json-pretty      Indent JSON (with --rag)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    # override=True: repo .env wins over stale machine-level GOOGLE_APPLICATION_CREDENTIALS
    try:
        load_dotenv(_repo_root() / ".env", interpolate=False, override=True)
    except TypeError:
        load_dotenv(_repo_root() / ".env", override=True)
    load_dotenv(interpolate=False)


def main() -> int:
    _load_dotenv()
    # Helps PaddleOCR on Windows when protobuf is new (see protobuf 4 + generated _pb2 warnings).
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

    p = argparse.ArgumentParser(description="OCR test: PDF page → patches → ocr_patch")
    p.add_argument("pdf", type=Path, help="Path to PDF")
    p.add_argument(
        "--page",
        type=int,
        default=1,
        metavar="N",
        help="1-based PDF page number to OCR (default: 1)",
    )
    p.add_argument(
        "--expected-regions",
        type=int,
        default=None,
        metavar="N",
        help="Optional expected answer-box count (segmentation hint)",
    )
    p.add_argument(
        "--max-chars",
        type=int,
        default=0,
        metavar="N",
        help="Truncate each region's printed text to N chars (0 = print full text)",
    )
    p.add_argument(
        "--patch-size",
        type=int,
        default=768,
        metavar="N",
        help="Resize each region to N×N before OCR (default: 768; higher preserves handwriting detail)",
    )
    p.add_argument(
        "--pdf-zoom",
        type=float,
        default=3.5,
        metavar="F",
        help="PDF rasterization scale (default: 3.5; higher = sharper, slower)",
    )
    p.add_argument(
        "--inpaint-grey",
        action="store_true",
        help="Inpaint flat grey regions (PDF redactions); off by default — can harm lined paper",
    )
    p.add_argument(
        "--full-page",
        action="store_true",
        help="Run Vision once on the whole deskewed page (avoids cuts between regions)",
    )
    p.add_argument(
        "--bbox-padding",
        type=float,
        default=None,
        metavar="F",
        help="Contour bbox expansion fraction (default: 0.06; ignored with --full-page)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        metavar="FILE",
        help="Write output: plain .txt (default) or .json with --rag",
    )
    p.add_argument(
        "--rag",
        action="store_true",
        help="Emit RAG-ready JSON (normalized text, chunks with metadata for embeddings)",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        metavar="N",
        help="RAG chunk size in characters (default: 512)",
    )
    p.add_argument(
        "--chunk-overlap",
        type=int,
        default=96,
        metavar="N",
        help="RAG overlap between chunks (default: 96)",
    )
    p.add_argument(
        "--json-pretty",
        action="store_true",
        help="Pretty-print JSON (with --rag)",
    )
    args = p.parse_args()

    ps = int(args.patch_size)
    if ps < 128 or ps > 2048:
        print(
            "[fail] --patch-size must be between 128 and 2048 (typical: 768 or 1024). "
            "If you meant 1024, use --patch-size 1024",
            file=sys.stderr,
        )
        return 1

    pdf = args.pdf
    if not pdf.is_absolute():
        pdf = (_repo_root() / pdf).resolve()
    if not pdf.is_file():
        print(f"[fail] not a file: {pdf}", file=sys.stderr)
        return 1

    os.chdir(_repo_root())
    if not os.environ.get("PYTHONPATH"):
        sys.path.insert(0, str(_repo_root() / "src"))

    if args.inpaint_grey:
        os.environ["GRADE_OCR_INPAINT_GREY_REDACTIONS"] = "1"

    from autograder.ocr import ocr_patch
    from autograder.preprocessing import preprocess_pipeline

    if args.page < 1:
        print("[fail] --page must be >= 1", file=sys.stderr)
        return 1

    if args.rag:
        print(f"PDF: {pdf} (page {args.page})", file=sys.stderr, flush=True)
    else:
        print(f"PDF: {pdf} (page {args.page})", flush=True)
    z = float(args.pdf_zoom)
    if z < 1.0:
        z = 3.5
    pad_frac = 0.06 if args.bbox_padding is None else max(0.0, min(0.2, float(args.bbox_padding)))
    pad_px = 0 if pad_frac <= 0 else 20
    result = preprocess_pipeline(
        pdf,
        expected_num_regions=args.expected_regions,
        pdf_page_index=args.page - 1,
        patch_size=ps,
        pdf_render_scale=min(6.0, z),
        full_page=args.full_page,
        bbox_padding_frac=pad_frac,
        bbox_padding_px_min=pad_px,
    )
    d = result.diagnostics or {}
    if not args.rag:
        print(
            f"regions={len(result.patches)} fallback_grid={result.used_fallback_grid} "
            f"diagnostics={d}",
            flush=True,
        )

    regions_data: list[dict] = []
    out_chunks: list[str] = []
    for i, patch in enumerate(result.patches):
        rid = result.region_ids[i] if i < len(result.region_ids) else f"R{i+1}"
        res = ocr_patch(patch, retry_delay=0.0)
        body = res.text or ""
        if args.max_chars and len(body) > args.max_chars:
            body = body[: args.max_chars] + "…"
        regions_data.append(
            {
                "region_id": rid,
                "text": body,
                "ocr_engine": res.engine,
                "ocr_confidence": res.confidence,
                "flags": list(res.flags or []),
            }
        )
        if not args.rag:
            print(
                f"\n[{rid}] engine={res.engine!r} confidence={res.confidence:.3f} "
                f"low_confidence={res.low_confidence}",
                flush=True,
            )
            print("  extracted text:", flush=True)
            print(body, flush=True)
            if res.flags:
                print(f"  flags: {res.flags}", flush=True)
            if args.output is not None:
                out_chunks.append(f"[{rid}] (engine={res.engine}, conf={res.confidence:.3f})\n{body}")

    if args.rag:
        from autograder.rag_extract import build_rag_payload, rag_payload_to_json

        preprocess_meta = {
            "patch_size": ps,
            "pdf_render_scale": min(6.0, z),
            "pdf_page_index": args.page - 1,
            "full_page": args.full_page,
            "diagnostics": d,
        }
        payload = build_rag_payload(
            source_path=str(pdf.resolve()),
            page=args.page,
            regions=regions_data,
            preprocess=preprocess_meta,
            chunk_size=max(64, int(args.chunk_size)),
            chunk_overlap=max(0, int(args.chunk_overlap)),
        )
        js = rag_payload_to_json(payload, pretty=args.json_pretty)
        out_path_rag = args.output
        if out_path_rag is not None:
            out_p = (
                out_path_rag
                if out_path_rag.is_absolute()
                else (_repo_root() / out_path_rag).resolve()
            )
            out_p.write_text(js, encoding="utf-8")
            print(f"[wrote] {out_p}", file=sys.stderr, flush=True)
        else:
            print(js, flush=True)
        return 0

    if args.output is not None:
        out_path = args.output
        if not out_path.is_absolute():
            out_path = (_repo_root() / out_path).resolve()
        header = (
            f"# GRADE ocr_pdf.py\n# Source: {pdf.name}\n# Page: {args.page}\n\n"
        )
        out_path.write_text(header + "\n---\n\n".join(out_chunks), encoding="utf-8")
        print(f"\n[wrote] {out_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
