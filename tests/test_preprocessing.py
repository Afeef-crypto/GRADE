"""Unit tests for AutoGrader image preprocessing (Phase 1)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from autograder.preprocessing import (
    PATCH_SIZE,
    ingest,
    preprocess_patch,
    preprocess_pipeline,
    segment,
    PreprocessResult,
)


class TestIngest:
    """Tests for ingest() — load, grayscale, adaptive threshold, optional deskew."""

    def test_ingest_accepts_ndarray_grayscale(self):
        img = np.random.randint(0, 256, (100, 200), dtype=np.uint8)
        out = ingest(img, do_deskew=False)
        assert out.ndim == 2
        assert out.shape == img.shape
        assert out.dtype == np.uint8
        assert np.issubdtype(out.dtype, np.integer)

    def test_ingest_accepts_ndarray_bgr(self):
        img = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        out = ingest(img, do_deskew=False)
        assert out.ndim == 2
        assert out.shape == (100, 200)

    def test_ingest_accepts_file_path(self, synthetic_binary_sheet):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            import cv2
            cv2.imwrite(path, synthetic_binary_sheet)
            out = ingest(path, do_deskew=False)
            assert out.ndim == 2
            assert out.shape == synthetic_binary_sheet.shape
        finally:
            Path(path).unlink(missing_ok=True)

    def test_ingest_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            ingest("/nonexistent/image.png")

    def test_ingest_deskew_does_not_crash(self, synthetic_binary_sheet):
        out = ingest(synthetic_binary_sheet, do_deskew=True)
        assert out.ndim == 2
        assert out.size == synthetic_binary_sheet.size


class TestSegment:
    """Tests for segment() — contour detection, filter, sort, grid fallback."""

    def test_segment_finds_rectangles(self, synthetic_binary_sheet):
        # Invert: segment expects dark regions; our boxes are 0, background 255.
        # segment() uses RETR_EXTERNAL and THRESH_BINARY_INV, so dark = foreground.
        patches, bboxes, used_fallback = segment(synthetic_binary_sheet)
        assert isinstance(patches, list)
        assert isinstance(bboxes, list)
        assert len(patches) == len(bboxes)
        # We have 3 black boxes; contour detection should find 3 (or possibly more if borders create extra contours)
        assert len(patches) >= 1
        for patch in patches:
            assert patch.ndim == 2
            assert patch.size > 0
        for (x, y, w, h) in bboxes:
            assert w > 0 and h > 0
            assert x >= 0 and y >= 0

    def test_segment_grid_fallback_when_count_mismatch(self, synthetic_binary_sheet):
        # Ask for 5 regions; contour count may not be 5 → fallback to 5-row grid
        patches, bboxes, used_fallback = segment(
            synthetic_binary_sheet, expected_num_regions=5
        )
        assert used_fallback is True
        assert len(patches) == 5
        assert len(bboxes) == 5
        for patch in patches:
            assert patch.ndim == 2

    def test_segment_sort_order_top_to_bottom(self, synthetic_binary_sheet):
        _, bboxes, _ = segment(synthetic_binary_sheet)
        if len(bboxes) >= 2:
            y_centers = [y + h // 2 for (x, y, w, h) in bboxes]
            assert y_centers == sorted(y_centers)


class TestPreprocessPatch:
    """Tests for preprocess_patch() — bilateral, morph, resize."""

    def test_preprocess_patch_output_shape(self, small_patch):
        out = preprocess_patch(small_patch, size=384)
        assert out.shape == (384, 384)
        assert out.dtype == np.uint8

    def test_preprocess_patch_default_size(self, small_patch):
        out = preprocess_patch(small_patch)
        assert out.shape == (PATCH_SIZE, PATCH_SIZE)

    def test_preprocess_patch_custom_size(self, small_patch):
        out = preprocess_patch(small_patch, size=256)
        assert out.shape == (256, 256)


class TestPreprocessPipeline:
    """Tests for full preprocess_pipeline() — ingest → segment → preprocess_patch."""

    def test_pipeline_returns_preprocess_result(self, synthetic_binary_sheet):
        result = preprocess_pipeline(synthetic_binary_sheet, do_deskew=False)
        assert isinstance(result, PreprocessResult)
        assert isinstance(result.patches, list)
        assert isinstance(result.bboxes, list)
        assert isinstance(result.used_fallback_grid, bool)
        assert len(result.patches) == len(result.bboxes)

    def test_pipeline_patches_are_384x384(self, synthetic_binary_sheet):
        result = preprocess_pipeline(
            synthetic_binary_sheet,
            expected_num_regions=3,
            do_deskew=False,
        )
        for patch in result.patches:
            assert patch.shape == (384, 384), "Each patch must be 384×384 for OCR input"

    def test_pipeline_with_grid_fallback(self, synthetic_binary_sheet):
        result = preprocess_pipeline(
            synthetic_binary_sheet,
            expected_num_regions=4,
            do_deskew=False,
        )
        assert result.used_fallback_grid is True
        assert len(result.patches) == 4
        for patch in result.patches:
            assert patch.shape == (384, 384)

    def test_pipeline_from_file(self, synthetic_binary_sheet):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            import cv2
            cv2.imwrite(path, synthetic_binary_sheet)
            result = preprocess_pipeline(path, expected_num_regions=3, do_deskew=False)
            assert len(result.patches) >= 1
            assert all(p.shape == (384, 384) for p in result.patches)
        finally:
            Path(path).unlink(missing_ok=True)


class TestPreprocessResult:
    """Tests for PreprocessResult and bbox consistency."""

    def test_bboxes_valid(self, synthetic_binary_sheet):
        result = preprocess_pipeline(synthetic_binary_sheet, do_deskew=False)
        h, w = synthetic_binary_sheet.shape[:2]
        for (x, y, bw, bh) in result.bboxes:
            assert 0 <= x < w and 0 <= y < h
            assert x + bw <= w and y + bh <= h
            assert bw > 0 and bh > 0
