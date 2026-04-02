from __future__ import annotations

import uuid

import numpy as np

from autograder.api import _aggregate_grading_confidence, evaluate
from autograder.schemas import EvaluateRequest


class _FakeOCR:
    text = "student answer"
    confidence = 0.9
    engine = "mock"
    flags = []


def test_confidence_aggregation():
    assert _aggregate_grading_confidence(["high", "high"]) == "high"
    assert _aggregate_grading_confidence(["high", "medium"]) == "medium"
    assert _aggregate_grading_confidence(["high", "low"]) == "low"


def test_evaluate_pipeline_smoke(monkeypatch):
    # fake sheet and keys
    monkeypatch.setattr("autograder.api.get_sheet", lambda _id: {"id": _id, "path": "fake.png"})
    monkeypatch.setattr(
        "autograder.api.list_answer_keys",
        lambda exam_id: [
            {
                "question_id": "Q1",
                "expected_answer": "ans1",
                "max_marks": 4.0,
                "domain": "general",
                "rubric_override": None,
                "embedding": [0.1] * 128,
            },
            {
                "question_id": "Q2",
                "expected_answer": "ans2",
                "max_marks": 4.0,
                "domain": "general",
                "rubric_override": None,
                "embedding": [0.2] * 128,
            },
        ],
    )

    from autograder.preprocessing import PreprocessResult

    monkeypatch.setattr(
        "autograder.api.preprocess_pipeline",
        lambda *args, **kwargs: PreprocessResult(
            patches=[np.zeros((384, 384), dtype=np.uint8), np.zeros((384, 384), dtype=np.uint8)],
            bboxes=[(0, 0, 10, 10), (0, 10, 10, 10)],
            region_ids=["R1", "R2"],
            used_fallback_grid=False,
            diagnostics={"num_regions": 2, "patch_size": 384},
        ),
    )
    monkeypatch.setattr("autograder.api.ocr_patch", lambda _patch: _FakeOCR())

    from autograder.schemas import EvaluationResult, RubricScores

    monkeypatch.setattr(
        "autograder.api.score_answer_llm",
        lambda **kwargs: (
            EvaluationResult(
                awarded_marks=3.0,
                max_marks=4.0,
                rubric_scores=RubricScores(
                    factual_accuracy=3,
                    conceptual_completeness=3,
                    reasoning=3,
                    domain_terminology=3,
                ),
                feedback="ok",
                grading_confidence="high",
                flags=[],
            ),
            "mock-llm",
        ),
    )

    captured = {}

    def _insert(payload):
        captured.update(payload)
        return str(uuid.uuid4())

    monkeypatch.setattr("autograder.api.insert_evaluation_result", _insert)

    req = EvaluateRequest(sheet_id="sheet-1", exam_id="exam-1")
    resp = evaluate(req)

    assert resp.total_marks == 6.0
    assert resp.max_total == 8.0
    assert len(resp.questions) == 2
    assert captured["grading_confidence"] == "high"
