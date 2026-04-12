"""RAG → Postgres integration (requires PostgreSQL + pgvector when not skipped)."""

from __future__ import annotations

import pytest

from autograder.embeddings import EMBEDDING_DIMS, embed_text_local
from autograder.rag_extract import rag_rows_for_postgres


def test_rag_rows_for_postgres_dimensions():
    payload = {
        "document_id": "test_doc_1",
        "schema_version": "1.0",
        "chunks": [
            {
                "chunk_id": "test_doc_1_c0",
                "index": 0,
                "text": "hello retrieval world",
                "metadata": {"region_id": "R1", "source_page": 1},
            }
        ],
    }
    rows = rag_rows_for_postgres(payload, embed_fn=embed_text_local)
    assert len(rows) == 1
    assert len(rows[0]["embedding"]) == EMBEDDING_DIMS
    assert rows[0]["chunk_id"] == "test_doc_1_c0"
    assert rows[0]["document_id"] == "test_doc_1"


def test_rag_pgvector_roundtrip():
    """Upsert + ANN search against live DB (see conftest GRADE_DATABASE_URL)."""
    pytest.importorskip("psycopg")
    from autograder.db import (
        delete_rag_chunks_for_document,
        init_db,
        search_rag_chunks,
        upsert_rag_chunks_batch,
    )

    doc_id = "__pytest_rag_doc__"
    try:
        init_db()
        delete_rag_chunks_for_document(doc_id)
        rows = [
            {
                "chunk_id": f"{doc_id}_a",
                "document_id": doc_id,
                "chunk_index": 0,
                "text_content": "the quick brown fox jumps",
                "embedding": embed_text_local("the quick brown fox jumps"),
                "meta": {"t": "a"},
            },
            {
                "chunk_id": f"{doc_id}_b",
                "document_id": doc_id,
                "chunk_index": 1,
                "text_content": "python asyncio networking",
                "embedding": embed_text_local("python asyncio networking"),
                "meta": {"t": "b"},
            },
        ]
        n = upsert_rag_chunks_batch(rows)
        assert n == 2

        q = embed_text_local("brown fox running")
        hits = search_rag_chunks(q, top_k=2, document_id=doc_id)
        assert len(hits) >= 1
        assert hits[0]["chunk_id"] == f"{doc_id}_a"
    except OSError as e:
        pytest.skip(f"database unavailable: {e}")
    except Exception as e:
        msg = str(e).lower()
        if "connection" in msg or "refused" in msg or "password" in msg:
            pytest.skip(f"database unavailable: {e}")
        raise
    finally:
        try:
            delete_rag_chunks_for_document(doc_id)
        except Exception:
            pass
