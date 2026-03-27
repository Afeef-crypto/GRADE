"""
GRADE — Automatic Handwritten Answer Sheet Evaluator.

Phase 1: Image preprocessing pipeline (ingest, segment, preprocess_patch).
"""

from autograder.preprocessing import (
    ingest,
    segment,
    preprocess_patch,
    preprocess_pipeline,
    PreprocessResult,
)
from autograder.ocr import (
    OCRResult,
    ocr_patch,
    ocr_patch_consensus,
    CONFIDENCE_FLAG_THRESHOLD,
)

__all__ = [
    "ingest",
    "segment",
    "preprocess_patch",
    "preprocess_pipeline",
    "PreprocessResult",
    "OCRResult",
    "ocr_patch",
    "ocr_patch_consensus",
    "CONFIDENCE_FLAG_THRESHOLD",
]
