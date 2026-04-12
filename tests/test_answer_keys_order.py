"""Answer key ordering: region index alignment and Q2 vs Q10 numbering."""

from __future__ import annotations

import uuid

import pytest

pytest.importorskip("fastapi")


def test_list_answer_keys_natural_numeric_order():
    """After one-time migration backfill, keys sort by human-style question_id (Q2 before Q10)."""
    from autograder.db import _backfill_answer_keys_sort_order_natural, get_conn, init_db, insert_answer_key
    from autograder.embeddings import embed_text_local

    init_db()
    exam = f"nat-{uuid.uuid4().hex[:8]}"
    z = embed_text_local("x")
    for qid, so in (("Q10", 0), ("Q2", 1), ("Q1", 2)):
        insert_answer_key(exam, qid, "a", z, 1.0, "general", None, sort_order=so)

    conn = get_conn()
    _backfill_answer_keys_sort_order_natural(conn)
    conn.close()

    from autograder.db import list_answer_keys

    keys = list_answer_keys(exam)
    assert [k["question_id"] for k in keys] == ["Q1", "Q2", "Q10"]


def test_upload_key_preserves_json_array_order(client):
    """First patch on the sheet should align with the first entry in the uploaded key JSON."""
    exam_id = f"order-{uuid.uuid4().hex[:8]}"
    payload = {
        "exam_id": exam_id,
        "questions": [
            {"question_id": "Q10", "expected_answer": "ten", "max_marks": 1},
            {"question_id": "Q2", "expected_answer": "two", "max_marks": 1},
            {"question_id": "Q1", "expected_answer": "one", "max_marks": 1},
        ],
    }
    r = client.post("/upload/key", json=payload)
    assert r.status_code == 200
    from autograder.db import list_answer_keys

    keys = list_answer_keys(exam_id)
    assert [k["question_id"] for k in keys] == ["Q10", "Q2", "Q1"]
