"""
AutoGrader — Automatic Handwritten Answer Sheet Evaluator.

Phase 1: Image preprocessing pipeline (ingest, segment, preprocess_patch).
"""

from autograder.preprocessing import (
    ingest,
    segment,
    preprocess_patch,
    preprocess_pipeline,
    PreprocessResult,
)

__all__ = [
    "ingest",
    "segment",
    "preprocess_patch",
    "preprocess_pipeline",
    "PreprocessResult",
]
