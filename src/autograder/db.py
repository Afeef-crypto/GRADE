"""
PostgreSQL + pgvector persistence for GRADE.

Requires ``GRADE_DATABASE_URL`` or ``DATABASE_URL`` and
``pip install -e ".[api]"`` or ``pip install 'psycopg[binary]' pgvector``.

Connections use ``_connection_dsn()`` which may set ``hostaddr`` from automatic DNS resolution,
``GRADE_DATABASE_HOSTADDR`` (manual IP when system DNS fails), or related env vars
(``GRADE_PG_SKIP_HOSTADDR_RESOLVE``, ``GRADE_PG_PREFER_ADDR_FAMILY``).
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import socket
import uuid
from typing import Any, Dict, List, Optional

import psycopg
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.rows import dict_row
from psycopg.types.json import Json

try:
    from pgvector.psycopg import Vector, register_vector
except ImportError:  # pragma: no cover - optional deps
    Vector = None  # type: ignore[misc, assignment]
    register_vector = None  # type: ignore[misc, assignment]


def _dsn() -> str:
    url = (os.environ.get("GRADE_DATABASE_URL") or os.environ.get("DATABASE_URL") or "").strip()
    if not url:
        raise RuntimeError(
            "GRADE_DATABASE_URL or DATABASE_URL must be set (PostgreSQL is required; SQLite is not supported)."
        )
    return url


def _host_is_numeric(host: str) -> bool:
    """True if host is already an IP literal (IPv4/IPv6), not a DNS name."""
    h = host.strip("[]")
    if "%" in h:
        h = h.split("%", 1)[0]
    try:
        ipaddress.ip_address(h)
        return True
    except ValueError:
        return False


def _parse_manual_hostaddr(raw: str) -> Optional[str]:
    """Normalize GRADE_DATABASE_HOSTADDR to a string libpq accepts (IPv4 or IPv6)."""
    s = raw.strip()
    if not s:
        return None
    s = s.strip("[]")
    if "%" in s:
        s = s.split("%", 1)[0]
    try:
        return str(ipaddress.ip_address(s))
    except ValueError:
        return None


def _connection_dsn() -> str:
    """
    Connection string for psycopg, with optional DNS pre-resolution.

    Resolves the hostname with Python's ``getaddrinfo`` and sets ``hostaddr`` so libpq
    connects by IP while ``host`` stays the original name (TLS/SNI). This often fixes
    Windows setups where libpq fails DNS but Python succeeds.

    Set ``GRADE_PG_SKIP_HOSTADDR_RESOLVE=1`` to disable automatic resolution.

    If **system DNS fails** but you can resolve the host elsewhere (e.g. ``nslookup``), set
    ``GRADE_DATABASE_HOSTADDR`` to that IPv4 or IPv6 literal; TLS still uses the hostname from the URL.

    Optional ``GRADE_PG_PREFER_ADDR_FAMILY`` may be ``ipv4`` (default) or ``ipv6`` when both exist.
    """
    raw = _dsn()
    flag = (os.environ.get("GRADE_PG_SKIP_HOSTADDR_RESOLVE") or "").strip().lower()
    if flag in ("1", "true", "yes"):
        return raw
    try:
        parts = conninfo_to_dict(raw)
    except Exception:
        return raw
    if parts.get("hostaddr"):
        return raw
    manual = _parse_manual_hostaddr(os.environ.get("GRADE_DATABASE_HOSTADDR") or "")
    if manual:
        merged = dict(parts)
        merged["hostaddr"] = manual
        try:
            return make_conninfo(**merged)
        except Exception:
            return raw
    host = parts.get("host")
    if not host or _host_is_numeric(host):
        return raw
    port_s = parts.get("port")
    try:
        port = int(port_s) if port_s is not None and str(port_s).strip() != "" else 5432
    except (TypeError, ValueError):
        port = 5432
    prefer = (os.environ.get("GRADE_PG_PREFER_ADDR_FAMILY") or "").strip().lower()
    try:
        infos = socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except OSError:
        return raw
    if not infos:
        return raw
    if prefer == "ipv6":
        infos.sort(key=lambda a: (0 if a[0] == socket.AF_INET6 else 1))
    elif prefer == "ipv4":
        infos.sort(key=lambda a: (0 if a[0] == socket.AF_INET else 1))
    else:
        infos.sort(key=lambda a: (0 if a[0] == socket.AF_INET else 1))
    ip = infos[0][4][0]
    merged = dict(parts)
    merged["hostaddr"] = ip
    try:
        return make_conninfo(**merged)
    except Exception:
        return raw


def _require_vector_deps() -> None:
    if Vector is None or register_vector is None:
        raise RuntimeError(
            "Install PostgreSQL dependencies: pip install 'psycopg[binary]' pgvector"
        )


def get_conn():
    """Return an open psycopg connection (caller must close)."""
    _require_vector_deps()
    conn = psycopg.connect(_connection_dsn(), row_factory=dict_row)
    register_vector(conn)
    return conn


# ---------------------------------------------------------------------------
# Natural-sort backfill (tests / one-off migration helpers)
# ---------------------------------------------------------------------------


def _natural_question_id_key(question_id: str) -> List[Any]:
    """Sort key so Q2 < Q10 (lexicographic order would place Q10 before Q2)."""
    parts = re.split(r"(\d+)", question_id)
    out: List[Any] = []
    for p in parts:
        if not p:
            continue
        out.append(int(p) if p.isdigit() else p.lower())
    return out


def _backfill_answer_keys_sort_order_natural(conn) -> None:
    """Reassign sort_order per exam_id by natural order of question_id (legacy data fix)."""
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT exam_id FROM public.answer_keys")
        exams = cur.fetchall()
        for (exam_id,) in exams:
            cur.execute(
                "SELECT id, question_id FROM public.answer_keys WHERE exam_id = %s",
                (exam_id,),
            )
            rows = cur.fetchall()
            ordered = sorted(rows, key=lambda r: _natural_question_id_key(r[1]))
            for i, (row_id, _) in enumerate(ordered):
                cur.execute(
                    "UPDATE public.answer_keys SET sort_order = %s WHERE id = %s",
                    (i, row_id),
                )
    conn.commit()


# Applied on init_db when tables may be missing (idempotent; matches supabase/migrations).
_SCHEMA_STATEMENTS: List[str] = [
    "CREATE EXTENSION IF NOT EXISTS vector",
    "COMMENT ON EXTENSION vector IS 'pgvector — embedding storage and similarity search for GRADE answer_keys'",
    """
CREATE TABLE IF NOT EXISTS public.sheets (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    path TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
""".strip(),
    "COMMENT ON TABLE public.sheets IS 'Scanned answer sheets; path is filesystem or storage URI resolved by the API'",
    """
CREATE TABLE IF NOT EXISTS public.answer_keys (
    id TEXT PRIMARY KEY,
    exam_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    expected_answer TEXT NOT NULL,
    embedding vector(128) NOT NULL,
    max_marks DOUBLE PRECISION NOT NULL CHECK (max_marks > 0),
    domain TEXT NOT NULL DEFAULT 'general',
    rubric_override JSONB,
    sort_order INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
""".strip(),
    "COMMENT ON TABLE public.answer_keys IS 'Model answers; sort_order preserves upload / sheet row order'",
    "COMMENT ON COLUMN public.answer_keys.embedding IS '128-D from embed_text_local; change dimension only with a coordinated migration + re-embed'",
    """
CREATE INDEX IF NOT EXISTS idx_answer_keys_exam_sort
    ON public.answer_keys (exam_id, sort_order ASC, question_id ASC)
""".strip(),
    """
CREATE INDEX IF NOT EXISTS idx_answer_keys_embedding_hnsw
    ON public.answer_keys
    USING hnsw (embedding vector_cosine_ops)
""".strip(),
    """
CREATE TABLE IF NOT EXISTS public.evaluation_results (
    id TEXT PRIMARY KEY,
    sheet_id TEXT NOT NULL,
    exam_id TEXT NOT NULL,
    per_question_scores JSONB NOT NULL,
    total_marks DOUBLE PRECISION NOT NULL,
    max_total DOUBLE PRECISION NOT NULL,
    confidence_flag BOOLEAN NOT NULL DEFAULT FALSE,
    grading_confidence TEXT,
    ocr_engine_used TEXT,
    prompt_hash TEXT,
    llm_model TEXT,
    flags JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_evaluation_results_sheet
        FOREIGN KEY (sheet_id) REFERENCES public.sheets (id) ON DELETE CASCADE
)
""".strip(),
    "COMMENT ON COLUMN public.evaluation_results.per_question_scores IS 'Array of question result objects (same shape as API questions[])'",
    "COMMENT ON COLUMN public.evaluation_results.flags IS 'JSON array of string flags'",
    """
CREATE INDEX IF NOT EXISTS idx_evaluation_results_sheet
    ON public.evaluation_results (sheet_id)
""".strip(),
    """
CREATE INDEX IF NOT EXISTS idx_evaluation_results_exam_created
    ON public.evaluation_results (exam_id, created_at DESC)
""".strip(),
    """
CREATE TABLE IF NOT EXISTS public.rag_chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text_content TEXT NOT NULL,
    embedding vector(128) NOT NULL,
    meta JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
""".strip(),
    "COMMENT ON TABLE public.rag_chunks IS 'OCR/RAG text chunks with embed_text_local(128-D) vectors for retrieval'",
    """
CREATE INDEX IF NOT EXISTS idx_rag_chunks_document_id
    ON public.rag_chunks (document_id)
""".strip(),
    """
CREATE INDEX IF NOT EXISTS idx_rag_chunks_embedding_hnsw
    ON public.rag_chunks
    USING hnsw (embedding vector_cosine_ops)
""".strip(),
]


def init_db() -> None:
    _require_vector_deps()
    with psycopg.connect(_connection_dsn(), autocommit=True) as conn:
        with conn.cursor() as cur:
            for stmt in _SCHEMA_STATEMENTS:
                try:
                    cur.execute(stmt)
                except psycopg.Error as e:
                    msg = str(e).lower()
                    if any(
                        s in msg
                        for s in (
                            "already exists",
                            "duplicate",
                            "permission denied",
                            "must be owner",
                            "only superuser",
                        )
                    ):
                        continue
                    raise


def insert_sheet(filename: str, path: str) -> str:
    sheet_id = str(uuid.uuid4())
    with psycopg.connect(_connection_dsn(), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO public.sheets (id, filename, path) VALUES (%s, %s, %s)",
                (sheet_id, filename, path),
            )
    return sheet_id


def get_sheet(sheet_id: str) -> Optional[Dict[str, Any]]:
    with psycopg.connect(_connection_dsn(), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM public.sheets WHERE id = %s", (sheet_id,))
            row = cur.fetchone()
    if not row:
        return None
    d = dict(row)
    if d.get("created_at") is not None:
        d["created_at"] = str(d["created_at"])
    return d


def insert_answer_key(
    exam_id: str,
    question_id: str,
    expected_answer: str,
    embedding: List[float],
    max_marks: float,
    domain: str,
    rubric_override: Optional[dict],
    sort_order: int = 0,
) -> str:
    _require_vector_deps()
    key_id = str(uuid.uuid4())
    rub = Json(rubric_override) if rubric_override is not None else None
    with psycopg.connect(_connection_dsn(), row_factory=dict_row) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public.answer_keys(
                    id, exam_id, question_id, expected_answer, embedding, max_marks,
                    domain, rubric_override, sort_order
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    key_id,
                    exam_id,
                    question_id,
                    expected_answer,
                    Vector(embedding),
                    max_marks,
                    domain,
                    rub,
                    sort_order,
                ),
            )
    return key_id


def list_answer_keys(exam_id: str) -> List[Dict[str, Any]]:
    _require_vector_deps()
    with psycopg.connect(_connection_dsn(), row_factory=dict_row) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, exam_id, question_id, expected_answer, embedding, max_marks,
                       domain, rubric_override, sort_order, created_at
                FROM public.answer_keys
                WHERE exam_id = %s
                ORDER BY sort_order ASC, question_id ASC
                """,
                (exam_id,),
            )
            rows = cur.fetchall()
    out: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        emb = item.get("embedding")
        if isinstance(emb, (list, tuple)):
            item["embedding"] = [float(x) for x in emb]
        elif emb is None:
            item["embedding"] = []
        else:
            item["embedding"] = list(emb)
        ro = item.get("rubric_override")
        if isinstance(ro, str):
            item["rubric_override"] = json.loads(ro) if ro else None
        if item.get("created_at") is not None:
            item["created_at"] = str(item["created_at"])
        out.append(item)
    return out


def next_answer_key_sort_order_start(exam_id: str) -> int:
    with psycopg.connect(_connection_dsn(), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COALESCE(MAX(sort_order), -1) AS m FROM public.answer_keys WHERE exam_id = %s",
                (exam_id,),
            )
            row = cur.fetchone()
    return int(row["m"]) + 1


def insert_evaluation_result(payload: Dict[str, Any]) -> str:
    result_id = str(uuid.uuid4())
    flags = payload.get("flags", [])
    with psycopg.connect(_connection_dsn(), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public.evaluation_results(
                    id, sheet_id, exam_id, per_question_scores, total_marks, max_total,
                    confidence_flag, grading_confidence, ocr_engine_used, prompt_hash,
                    llm_model, flags
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    result_id,
                    payload["sheet_id"],
                    payload["exam_id"],
                    Json(payload["questions"]),
                    payload["total_marks"],
                    payload["max_total"],
                    bool(payload["confidence_flag"]),
                    payload["grading_confidence"],
                    payload.get("ocr_engine_used", "mixed"),
                    payload["prompt_hash"],
                    payload["llm_model"],
                    Json(flags),
                ),
            )
    return result_id


def _coerce_json_field(val: Any) -> Any:
    if isinstance(val, (list, dict)):
        return val
    if isinstance(val, str):
        return json.loads(val)
    return val


def get_result(result_id: str) -> Optional[Dict[str, Any]]:
    with psycopg.connect(_connection_dsn(), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM public.evaluation_results WHERE id = %s", (result_id,))
            row = cur.fetchone()
    if not row:
        return None
    d = dict(row)
    d["questions"] = _coerce_json_field(d["per_question_scores"])
    d["flags"] = _coerce_json_field(d["flags"])
    if not isinstance(d["flags"], list):
        d["flags"] = []
    d["confidence_flag"] = bool(d["confidence_flag"])
    if d.get("created_at") is not None:
        d["created_at"] = str(d["created_at"])
    return d


def upsert_rag_chunks_batch(rows: List[Dict[str, Any]]) -> int:
    """
    Insert or update RAG chunks with 128-D embeddings (``embed_text_local``).

    Each row must include: ``chunk_id``, ``document_id``, ``chunk_index``, ``text_content``,
    ``embedding`` (128 floats), and optionally ``meta`` (dict, stored as JSONB).
    """
    if not rows:
        return 0
    _require_vector_deps()
    n = 0
    with psycopg.connect(_connection_dsn(), row_factory=dict_row) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            for r in rows:
                cid = r["chunk_id"]
                did = r["document_id"]
                idx = int(r["chunk_index"])
                txt = r["text_content"]
                emb = r["embedding"]
                meta = Json(r.get("meta") or {})
                cur.execute(
                    """
                    INSERT INTO public.rag_chunks (
                        chunk_id, document_id, chunk_index, text_content, embedding, meta
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        document_id = EXCLUDED.document_id,
                        chunk_index = EXCLUDED.chunk_index,
                        text_content = EXCLUDED.text_content,
                        embedding = EXCLUDED.embedding,
                        meta = EXCLUDED.meta
                    """,
                    (cid, did, idx, txt, Vector(emb), meta),
                )
                n += 1
        conn.commit()
    return n


def search_rag_chunks(
    embedding: List[float],
    *,
    top_k: int = 5,
    document_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Approximate nearest neighbors by cosine distance (``<=>``) on ``rag_chunks.embedding``.
    """
    _require_vector_deps()
    top_k = max(1, min(100, int(top_k)))
    q = Vector(embedding)
    with psycopg.connect(_connection_dsn(), row_factory=dict_row) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            if document_id:
                cur.execute(
                    """
                    SELECT chunk_id, document_id, chunk_index, text_content, meta,
                           (embedding <=> %s::vector) AS distance
                    FROM public.rag_chunks
                    WHERE document_id = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (q, document_id, q, top_k),
                )
            else:
                cur.execute(
                    """
                    SELECT chunk_id, document_id, chunk_index, text_content, meta,
                           (embedding <=> %s::vector) AS distance
                    FROM public.rag_chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (q, q, top_k),
                )
            rows = cur.fetchall()
    out: List[Dict[str, Any]] = []
    for row in rows:
        d = dict(row)
        m = d.get("meta")
        if isinstance(m, str):
            d["meta"] = json.loads(m) if m else {}
        out.append(d)
    return out


def delete_rag_chunks_for_document(document_id: str) -> int:
    """Remove all chunks for a document. Returns deleted row count."""
    with psycopg.connect(_connection_dsn(), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM public.rag_chunks WHERE document_id = %s",
                (document_id,),
            )
            n = cur.rowcount or 0
        conn.commit()
    return n
