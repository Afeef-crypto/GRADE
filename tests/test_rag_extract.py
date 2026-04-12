"""Tests for RAG-oriented OCR text normalization and chunking."""

from autograder.rag_extract import (
    build_rag_payload,
    chunk_text,
    normalize_ocr_text,
)


def test_normalize_ocr_text_collapses_noise():
    raw = "  hello  \n\n\n  world  "
    assert normalize_ocr_text(raw) == "hello\n\nworld"


def test_chunk_text_overlap():
    t = "word " * 200
    chunks = chunk_text(t, chunk_size=80, overlap=20)
    assert len(chunks) >= 2
    assert all(len(c[0]) <= 120 for c in chunks)


def test_build_rag_payload_has_chunks():
    p = build_rag_payload(
        source_path="/tmp/x.pdf",
        page=1,
        regions=[
            {
                "region_id": "R1",
                "text": "alpha beta " * 50,
                "ocr_engine": "google",
                "ocr_confidence": 0.9,
                "flags": [],
            }
        ],
        preprocess={"patch_size": 768},
        chunk_size=100,
        chunk_overlap=20,
    )
    assert p["schema_version"] == "1.0"
    assert "full_text" in p
    assert len(p["chunks"]) >= 1
    assert p["chunks"][0]["metadata"]["region_id"] == "R1"
    assert "chunk_id" in p["chunks"][0]
