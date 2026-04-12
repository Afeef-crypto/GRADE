#!/usr/bin/env python3
"""
Load a RAG JSON payload (from ``ocr_pdf.py --rag``), embed chunks with ``embed_text_local``,
and upsert into PostgreSQL ``public.rag_chunks`` (pgvector).

Requires ``GRADE_DATABASE_URL`` and ``pip install -e ".[api]"`` (psycopg + pgvector).

Usage:
  PYTHONPATH=src python scripts/embed_rag_to_postgres.py extraction.json
  PYTHONPATH=src python scripts/embed_rag_to_postgres.py extraction.json --query "algorithm steps"

Optional:
  --dry-run   Print row counts only (no DB writes)
"""

from __future__ import annotations

import argparse
import json
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
    try:
        load_dotenv(_repo_root() / ".env", interpolate=False, override=True)
    except TypeError:
        load_dotenv(_repo_root() / ".env", override=True)


def main() -> int:
    _load_dotenv()
    p = argparse.ArgumentParser(description="Embed RAG JSON chunks and store in pgvector")
    p.add_argument("json_path", type=Path, help="Path to RAG JSON from ocr_pdf.py --rag")
    p.add_argument(
        "--query",
        type=str,
        default=None,
        help="After ingest, run a similarity search with this text (embed_text_local)",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of hits for --query (default: 3)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and embed only; do not write to Postgres",
    )
    args = p.parse_args()

    jp = args.json_path
    if not jp.is_absolute():
        jp = (_repo_root() / jp).resolve()
    if not jp.is_file():
        print(f"[fail] not a file: {jp}", file=sys.stderr)
        return 1

    os.chdir(_repo_root())
    if not os.environ.get("PYTHONPATH"):
        sys.path.insert(0, str(_repo_root() / "src"))

    raw = jp.read_text(encoding="utf-8")
    payload = json.loads(raw)

    from autograder.embeddings import embed_text_local
    from autograder.rag_extract import rag_rows_for_postgres

    rows = rag_rows_for_postgres(payload, embed_fn=embed_text_local)
    print(f"Chunks in file: {len(rows)}", flush=True)
    if not rows:
        print("[warn] no chunks to store", file=sys.stderr)
        return 0

    if args.dry_run:
        print("[dry-run] skipping database", flush=True)
    else:
        from autograder.db import init_db, upsert_rag_chunks_batch

        try:
            init_db()
            n = upsert_rag_chunks_batch(rows)
            print(f"[ok] upserted {n} rows into public.rag_chunks", flush=True)
        except OSError as e:
            print(
                "[fail] cannot reach database (network/DNS). Check internet, VPN, firewall, "
                "and that GRADE_DATABASE_URL host resolves (try: nslookup <host>). "
                f"Error: {e}",
                file=sys.stderr,
            )
            return 1
        except Exception as e:
            err = str(e).lower()
            if "getaddrinfo" in err or "could not translate host" in err:
                print(f"[fail] DNS / resolution ({e!r})", file=sys.stderr)
                print(
                    "  • If nslookup shows an IP but Python fails: set in .env or shell:\n"
                    "      GRADE_DATABASE_HOSTADDR=<IPv4 or IPv6 from nslookup>\n"
                    "    (TLS still uses the hostname in GRADE_DATABASE_URL.)\n"
                    "  • Run: PYTHONPATH=src python scripts/diagnose_pg_dns.py\n"
                    "  • Or use Supabase pooler URI (Dashboard → Connect), or local Docker:\n"
                    "      GRADE_DATABASE_URL=postgresql://grade:grade@127.0.0.1:5433/grade_test",
                    file=sys.stderr,
                )
            else:
                print(f"[fail] database error: {e}", file=sys.stderr)
            return 1

    if args.query:
        if args.dry_run:
            print("[warn] --query ignored with --dry-run (nothing stored)", file=sys.stderr)
        else:
            from autograder.db import search_rag_chunks

            qv = embed_text_local(args.query)
            hits = search_rag_chunks(
                qv,
                top_k=args.top_k,
                document_id=payload.get("document_id"),
            )
            print("\n--- search (cosine distance; lower = closer) ---", flush=True)
            for h in hits:
                dist = h.get("distance")
                tid = (h.get("chunk_id") or "")[:56]
                preview = (h.get("text_content") or "")[:120].replace("\n", " ")
                print(f"  distance={dist:.4f}  {tid}…  {preview!r}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
