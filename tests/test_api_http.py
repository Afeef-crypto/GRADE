"""HTTP-level smoke tests for the FastAPI app (Phase 4 wiring)."""

from __future__ import annotations

import json
import uuid

import pytest

pytest.importorskip("fastapi")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body.get("integrations", {}).get("preprocessing") is True


def test_integrations(client):
    r = client.get("/api/integrations")
    assert r.status_code == 200
    data = r.json()
    assert data["phase_1_preprocess"]["ok"] is True
    assert data["phase_4_api"]["ok"] is True


def test_upload_key_json(client):
    payload = {
        "exam_id": f"http-exam-{uuid.uuid4().hex[:8]}",
        "questions": [{"question_id": "Q1", "expected_answer": "alpha", "max_marks": 2}],
    }
    r = client.post("/upload/key", json=payload)
    assert r.status_code == 200
    out = r.json()
    assert len(out["key_ids"]) == 1


def test_upload_key_file(client):
    payload = {
        "exam_id": f"file-exam-{uuid.uuid4().hex[:8]}",
        "questions": [{"question_id": "Q1", "expected_answer": "beta", "max_marks": 2}],
    }
    raw = json.dumps(payload).encode("utf-8")
    r = client.post(
        "/upload/key/file",
        files={"file": ("key.json", raw, "application/json")},
    )
    assert r.status_code == 200
    assert len(r.json()["key_ids"]) == 1


def test_upload_sheet_and_evaluate_roundtrip(client, synthetic_binary_sheet, monkeypatch, tmp_path):
    """End-to-end HTTP: key file, sheet image, evaluate (OCR/scoring mocked)."""
    import cv2
    import numpy as np

    from autograder.preprocessing import PreprocessResult
    from autograder.schemas import EvaluationResult, RubricScores

    exam_id = f"roundtrip-{uuid.uuid4().hex[:8]}"
    key_payload = {
        "exam_id": exam_id,
        "questions": [
            {"question_id": "Q1", "expected_answer": "model one", "max_marks": 4},
            {"question_id": "Q2", "expected_answer": "model two", "max_marks": 4},
        ],
    }
    r0 = client.post(
        "/upload/key/file",
        files={"file": ("k.json", json.dumps(key_payload).encode("utf-8"), "application/json")},
    )
    assert r0.status_code == 200

    sheet_path = tmp_path / "s.png"
    cv2.imwrite(str(sheet_path), synthetic_binary_sheet)
    with sheet_path.open("rb") as fh:
        r1 = client.post("/upload/sheet", files={"file": ("s.png", fh, "image/png")})
    assert r1.status_code == 200
    sheet_id = r1.json()["sheet_id"]

    class _OCR:
        text = " student "
        confidence = 0.88
        engine = "mock-http"
        flags: list = []

    monkeypatch.setattr(
        "autograder.api.preprocess_pipeline",
        lambda *a, **k: PreprocessResult(
            patches=[np.zeros((64, 64), dtype=np.uint8), np.zeros((64, 64), dtype=np.uint8)],
            bboxes=[(0, 0, 10, 10), (0, 10, 10, 10)],
            region_ids=["R1", "R2"],
            used_fallback_grid=False,
            diagnostics={"num_regions": 2},
        ),
    )
    monkeypatch.setattr("autograder.api.ocr_patch", lambda _p: _OCR())
    monkeypatch.setattr(
        "autograder.api.score_answer_llm",
        lambda **kwargs: (
            EvaluationResult(
                awarded_marks=2.0,
                max_marks=4.0,
                rubric_scores=RubricScores(
                    factual_accuracy=2,
                    conceptual_completeness=2,
                    reasoning=2,
                    domain_terminology=2,
                ),
                feedback="mock",
                grading_confidence="high",
                flags=[],
            ),
            "mock-http-llm",
        ),
    )

    r2 = client.post(
        "/evaluate",
        json={"sheet_id": sheet_id, "exam_id": exam_id, "top_k": 3, "use_consensus_ocr": False},
    )
    assert r2.status_code == 200
    ev = r2.json()
    result_id = ev["result_id"]
    assert ev["sheet_id"] == sheet_id
    assert ev["exam_id"] == exam_id
    assert "flags" in ev
    assert len(ev["questions"]) == 2

    r3 = client.get(f"/result/{result_id}")
    assert r3.status_code == 200
    assert r3.json()["total_marks"] == ev["total_marks"]

    r4 = client.get(f"/result/{result_id}/rubric")
    assert r4.status_code == 200
    assert "dimension_totals" in r4.json()

    r5 = client.get(f"/sheet/{sheet_id}/file")
    assert r5.status_code == 200
    assert len(r5.content) > 0

    r6 = client.get(f"/report/{result_id}/pdf")
    assert r6.status_code == 200
    assert r6.headers.get("content-type", "").startswith("application/pdf")
    assert r6.content[:4] == b"%PDF"


def test_sheet_404(client):
    r = client.get("/sheet/not-a-real-uuid/file")
    assert r.status_code == 404


def test_result_404(client):
    r = client.get("/result/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 404
