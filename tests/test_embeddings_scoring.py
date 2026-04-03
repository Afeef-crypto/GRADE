"""Unit tests for local embeddings and scoring paths (no network, no API keys)."""

from __future__ import annotations

import json
import os

import pytest

from autograder.embeddings import EMBEDDING_DIMS, cosine_similarity, embed_text_local, retrieve_top_k
from autograder.scoring import _legacy_fallback_score, score_answer_llm


class TestEmbeddings:
    def test_embed_empty_is_zero_vector(self):
        v = embed_text_local("")
        assert len(v) == EMBEDDING_DIMS
        assert all(x == 0.0 for x in v)

    def test_embed_deterministic(self):
        a = embed_text_local("hello world")
        b = embed_text_local("hello world")
        assert a == b

    def test_cosine_identical_normalized(self):
        v = embed_text_local("alpha beta gamma")
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_cosine_orthogonal_empty(self):
        z = embed_text_local("")
        v = embed_text_local("something")
        assert cosine_similarity(z, v) == 0.0

    def test_retrieve_top_k_orders_by_similarity(self):
        emb_a = embed_text_local("photosynthesis chlorophyll")
        emb_b = embed_text_local("quantum mechanics wavefunction")
        candidates = [
            {"question_id": "q1", "expected_answer": "plants and light", "embedding": list(emb_b)},
            {"question_id": "q2", "expected_answer": "plant biology", "embedding": list(emb_a)},
        ]
        ranked = retrieve_top_k("photosynthesis in leaves", candidates, top_k=2)
        assert ranked[0][0]["question_id"] == "q2"
        assert ranked[0][1] >= ranked[1][1]


class TestScoringFallback:
    def test_legacy_fallback_identical_text_high_score(self):
        r = _legacy_fallback_score("same words here", "same words here", max_marks=10.0)
        assert r.awarded_marks == pytest.approx(10.0, abs=0.1)
        assert "llm_unavailable" in r.flags

    def test_legacy_fallback_empty_student_zero(self):
        r = _legacy_fallback_score("", "full model answer text", max_marks=5.0)
        assert r.awarded_marks == 0.0


class TestScoreAnswerLlmEnv:
    def test_llm_disabled_uses_fallback(self, monkeypatch):
        monkeypatch.delenv("GRADE_ENABLE_LLM", raising=False)
        monkeypatch.delenv("GRADE_GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GRADE_LLM_MOCK_RESPONSE", raising=False)
        res, label = score_answer_llm(
            student_answer="a b c",
            model_answer="a b c",
            max_marks=4.0,
            question_text="Q1",
            subject_domain="general",
            ocr_confidence=0.9,
        )
        assert label == "fallback-legacy"
        assert res.awarded_marks >= 0
        assert "llm_unavailable" in res.flags

    def test_mock_response_json_used_when_set(self, monkeypatch):
        monkeypatch.setenv("GRADE_ENABLE_LLM", "true")
        monkeypatch.setenv(
            "GRADE_LLM_MOCK_RESPONSE",
            json.dumps(
                {
                    "awarded_marks": 3.0,
                    "rubric_scores": {
                        "factual_accuracy": 3,
                        "conceptual_completeness": 3,
                        "reasoning": 3,
                        "domain_terminology": 3,
                    },
                    "feedback": "mock",
                    "grading_confidence": "high",
                    "flags": [],
                }
            ),
        )
        monkeypatch.delenv("GRADE_GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        res, _ = score_answer_llm(
            student_answer="x",
            model_answer="y",
            max_marks=4.0,
            question_text="Q1",
            subject_domain="general",
            ocr_confidence=0.5,
        )
        assert res.awarded_marks == 3.0
        assert res.feedback == "mock"

    def test_llm_on_no_key_sets_flag(self, monkeypatch):
        monkeypatch.setenv("GRADE_ENABLE_LLM", "true")
        monkeypatch.delenv("GRADE_GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GRADE_LLM_MOCK_RESPONSE", raising=False)
        res, _ = score_answer_llm(
            student_answer="answer",
            model_answer="model",
            max_marks=5.0,
            question_text="Q1",
            subject_domain="general",
            ocr_confidence=0.8,
        )
        assert "gemini_key_missing" in res.flags
        assert "Gemini API key" in res.feedback
