"""
Helpers to turn OCR output into RAG-friendly payloads (chunked JSON for vector DB / LLM context).

Chunking is character-based with overlap; boundaries prefer whitespace when possible.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Dict, List, Optional


def normalize_ocr_text(text: str) -> str:
    """Collapse noisy OCR whitespace; keep paragraph breaks where possible."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in text.split("\n")]
    # Drop empty runs but keep single blank between paragraphs
    out: List[str] = []
    prev_empty = False
    for ln in lines:
        if not ln:
            if not prev_empty and out:
                prev_empty = True
            continue
        if prev_empty and out:
            out.append("")
        prev_empty = False
        out.append(ln)
    return "\n".join(out).strip()


def chunk_text(
    text: str,
    *,
    chunk_size: int = 512,
    overlap: int = 96,
) -> List[Tuple[str, int, int]]:
    """
    Split ``text`` into overlapping chunks for embedding / retrieval.

    Returns list of (chunk_text, char_start, char_end) in the **normalized** string.
    """
    t = text.strip()
    if not t:
        return []
    if chunk_size < 64:
        chunk_size = 64
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)

    chunks: List[Tuple[str, int, int]] = []
    start = 0
    n = len(t)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            window = t[start:end]
            best = -1
            for sep in ("\n\n", "\n", ". ", "; ", ", ", " "):
                pos = window.rfind(sep)
                if pos > chunk_size // 3:
                    best = max(best, pos + len(sep))
            if best > 0:
                end = start + best
        piece = t[start:end].strip()
        if piece:
            chunks.append((piece, start, end))
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_id_for_text(doc_id: str, index: int, text: str) -> str:
    """Stable short id for a chunk (not globally unique; include doc_id in metadata)."""
    h = hashlib.sha256(f"{doc_id}:{index}:{text[:200]}".encode("utf-8")).hexdigest()[:16]
    return f"{doc_id}_c{index}_{h}"


def build_rag_payload(
    *,
    source_path: str,
    page: int,
    regions: List[Dict[str, Any]],
    preprocess: Dict[str, Any],
    chunk_size: int = 512,
    chunk_overlap: int = 96,
    document_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a single JSON-serializable document for RAG ingestion.

    ``regions`` items: ``region_id``, ``text``, ``ocr_engine``, ``ocr_confidence``, ``flags`` (optional).
    """
    fname = source_path.replace("\\", "/").split("/")[-1]
    doc_id = document_id or hashlib.sha256(f"{source_path}:{page}".encode("utf-8")).hexdigest()[:12]

    combined_parts: List[str] = []
    all_chunks: List[Dict[str, Any]] = []
    chunk_index = 0

    for reg in regions:
        rid = reg.get("region_id", "R1")
        raw = reg.get("text") or ""
        normalized = normalize_ocr_text(raw)
        combined_parts.append(f"[{rid}]\n{normalized}")

        for piece, c0, c1 in chunk_text(
            normalized, chunk_size=chunk_size, overlap=chunk_overlap
        ):
            cid = chunk_id_for_text(doc_id, chunk_index, piece)
            all_chunks.append(
                {
                    "chunk_id": cid,
                    "index": chunk_index,
                    "text": piece,
                    "char_start": c0,
                    "char_end": c1,
                    "metadata": {
                        "region_id": rid,
                        "ocr_engine": reg.get("ocr_engine"),
                        "ocr_confidence": reg.get("ocr_confidence"),
                        "source_page": page,
                        "source_file": fname,
                        "source_path": source_path,
                    },
                }
            )
            chunk_index += 1

    full_text = normalize_ocr_text("\n\n".join(combined_parts))

    return {
        "schema_version": "1.0",
        "kind": "grade_ocr_extraction",
        "document_id": doc_id,
        "source": {
            "type": "application/pdf",
            "filename": fname,
            "path": source_path,
            "page": page,
        },
        "preprocess": preprocess,
        "full_text": full_text,
        "regions": regions,
        "chunks": all_chunks,
        "chunking": {
            "chunk_size": chunk_size,
            "overlap": chunk_overlap,
            "chunk_count": len(all_chunks),
        },
    }


def rag_payload_to_json(payload: Dict[str, Any], *, pretty: bool = False) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None)


def rag_rows_for_postgres(
    payload: Dict[str, Any],
    *,
    embed_fn: Callable[[str], List[float]],
) -> List[Dict[str, Any]]:
    """
    Build rows for :func:`autograder.db.upsert_rag_chunks_batch` from a RAG JSON payload.

    ``embed_fn`` is typically ``autograder.embeddings.embed_text_local`` (128-D).
    """
    doc_id = payload["document_id"]
    rows: List[Dict[str, Any]] = []
    for c in payload.get("chunks") or []:
        text = (c.get("text") or "").strip()
        vec = embed_fn(text)
        meta = dict(c.get("metadata") or {})
        meta.setdefault("payload_schema", payload.get("schema_version"))
        rows.append(
            {
                "chunk_id": c["chunk_id"],
                "document_id": doc_id,
                "chunk_index": int(c.get("index", 0)),
                "text_content": text,
                "embedding": vec,
                "meta": meta,
            }
        )
    return rows
