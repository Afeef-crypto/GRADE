"""
Image preprocessing module for GRADE.

Stages 1–3: Ingest (load, grayscale, threshold, deskew), Segment (contour-based
answer regions), Preprocess (per-patch bilateral filter, morph close, resize 384×384).

Reference: [6] G. Bradski, "The OpenCV Library," Dr. Dobb's Journal, 2000.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Defaults for OCR input normalisation (proposal: 384×384)
PATCH_SIZE = 384

# Segment: min contour area as fraction of image area (filter tiny noise)
MIN_CONTOUR_AREA_FRAC = 0.002
MAX_CONTOUR_AREA_FRAC = 0.95
# Aspect ratio: expect roughly rectangular answer boxes (min width/height or height/width)
MIN_ASPECT_RATIO = 0.2
MAX_ASPECT_RATIO = 5.0


@dataclass
class PreprocessResult:
    """Result of full preprocessing pipeline: patches and their bounding boxes."""

    patches: List[np.ndarray]  # List of 384×384 grayscale patches
    bboxes: List[Tuple[int, int, int, int]]  # (x, y, w, h) per patch, on original image
    region_ids: List[str]  # Stable ordered ids: R1, R2, ...
    used_fallback_grid: bool  # True if contour count was wrong and grid fallback was used
    diagnostics: dict  # Lightweight debug/trace metadata


def ingest(
    image_input: Union[str, Path, np.ndarray],
    *,
    do_deskew: bool = True,
    block_size: int = 11,
    c: int = 2,
) -> np.ndarray:
    """
    Stage 1: Load image, grayscale, adaptive threshold, optional deskew.

    Args:
        image_input: File path (str/Path) or BGR image (ndarray).
        do_deskew: Whether to correct rotation via Hough line transform.
        block_size: Block size for adaptive threshold (odd).
        c: Constant subtracted from mean in adaptive threshold.

    Returns:
        Grayscale, thresholded (and optionally deskewed) image.
    """
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {image_input}")
    else:
        img = np.asarray(image_input)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif img.ndim != 3:
            raise ValueError("Image must be 2D (grayscale) or 3D (BGR/BGRA)")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if block_size % 2 == 0:
        block_size += 1
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        c,
    )

    if do_deskew:
        thresh = _deskew(thresh)

    return thresh


def _deskew(image: np.ndarray) -> np.ndarray:
    """Deskew image using Hough line transform to estimate rotation angle."""
    h, w = image.shape[:2]
    # Use edges to find dominant angle
    edges = cv2.Canny(image, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=min(w, h) // 4,
        minLineLength=min(w, h) // 4,
        maxLineGap=20,
    )
    if lines is None or len(lines) == 0:
        return image

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 != 0:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # We want small corrections (near 0 or ±90 for horizontal/vertical lines)
            if abs(angle) < 45:
                angles.append(angle)
            elif abs(angle - 90) < 45 or abs(angle + 90) < 45:
                angles.append(angle - 90 if angle > 0 else angle + 90)
    if not angles:
        return image

    median_angle = np.median(angles)
    if abs(median_angle) < 0.5:
        return image

    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def segment(
    image: np.ndarray,
    *,
    expected_num_regions: int | None = None,
    min_area_frac: float = MIN_CONTOUR_AREA_FRAC,
    max_area_frac: float = MAX_CONTOUR_AREA_FRAC,
    min_aspect: float = MIN_ASPECT_RATIO,
    max_aspect: float = MAX_ASPECT_RATIO,
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]], bool]:
    """
    Stage 2: Extract answer regions via contours; fallback to fixed grid if count wrong.

    Returns:
        (patches, bboxes, used_fallback_grid).
        patches: list of cropped grayscale regions (variable size).
        bboxes: list of (x, y, w, h) in original image coordinates.
        used_fallback_grid: True if expected_num_regions was set and contour count
            didn't match, so a simple N-row grid was used instead.
    """
    h, w = image.shape[:2]
    total_area = h * w
    min_area = int(total_area * min_area_frac)
    max_area = int(total_area * max_area_frac)

    # Contours on thresholded image (assume image is already binary or high-contrast)
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    if np.unique(gray).size > 2:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        # Dark (answer) regions should be foreground for findContours (white = object)
        binary = 255 - gray

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    rects: List[Tuple[int, int, int, int]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw <= 0 or ch <= 0:
            continue
        aspect = min(cw, ch) / max(cw, ch)
        if aspect < min_aspect or aspect > max_aspect:
            continue
        rects.append((x, y, cw, ch))

    # Sort top-to-bottom, then left-to-right (by center y, then center x)
    def sort_key(r: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x, y, cw, ch = r
        return (y + ch // 2, x + cw // 2)

    rects.sort(key=sort_key)

    used_fallback = False
    if expected_num_regions is not None and len(rects) != expected_num_regions:
        logger.warning(
            "Contour count %d != expected %d; using grid fallback.",
            len(rects),
            expected_num_regions,
        )
        patches, bboxes = _segment_grid_fallback(gray, expected_num_regions)
        used_fallback = True
    else:
        patches = []
        bboxes = []
        for (x, y, cw, ch) in rects:
            patch = gray[y : y + ch, x : x + cw]
            patches.append(patch)
            bboxes.append((x, y, cw, ch))

    return patches, bboxes, used_fallback


def _segment_grid_fallback(
    image: np.ndarray, num_rows: int
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    """Fallback: split image into num_rows equal horizontal strips."""
    h, w = image.shape[:2]
    patches = []
    bboxes = []
    step = h // num_rows
    for i in range(num_rows):
        y1 = i * step
        y2 = (h - 1) if i == num_rows - 1 else (i + 1) * step
        patch = image[y1:y2, :]
        patches.append(patch)
        bboxes.append((0, y1, w, y2 - y1))
    return patches, bboxes


def preprocess_patch(
    patch: np.ndarray,
    *,
    size: int = PATCH_SIZE,
    bilateral_d: int = 5,
    bilateral_sigma: float = 75.0,
    morph_kernel_size: Tuple[int, int] = (3, 3),
) -> np.ndarray:
    """
    Stage 3 (per-patch): Bilateral filter, morphological closing, resize to size×size.

    Args:
        patch: Grayscale patch (any size).
        size: Output width and height (default 384).
        bilateral_d, bilateral_sigma: Bilateral filter params.
        morph_kernel_size: Kernel for morphological closing.

    Returns:
        Grayscale image of shape (size, size).
    """
    if patch.ndim == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(patch, bilateral_d, bilateral_sigma, bilateral_sigma)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_size)
    closed = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)
    resized = cv2.resize(closed, (size, size), interpolation=cv2.INTER_LINEAR)
    return resized


def preprocess_pipeline(
    image_input: Union[str, Path, np.ndarray],
    *,
    expected_num_regions: int | None = None,
    patch_size: int = PATCH_SIZE,
    do_deskew: bool = True,
) -> PreprocessResult:
    """
    Run full preprocessing: ingest → segment → preprocess_patch for each region.

    Args:
        image_input: Path to scanned answer sheet (JPEG/PNG) or image array.
        expected_num_regions: If set, contour count is validated; on mismatch, grid fallback is used.
        patch_size: Output patch side length (default 384).
        do_deskew: Whether to deskew in ingest stage.

    Returns:
        PreprocessResult with list of size×size patches, bboxes, and fallback flag.
    """
    thresh = ingest(image_input, do_deskew=do_deskew)
    # For bbox consistency, segment the original (we need bboxes on original image)
    if isinstance(image_input, (str, Path)):
        orig = cv2.imread(str(image_input))
        if orig is None:
            orig = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    else:
        orig = (
            cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            if image_input.ndim == 2
            else image_input
        )
    orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    if do_deskew:
        orig_gray = _deskew(orig_gray)
    patches_raw, bboxes, used_fallback = segment(
        orig_gray, expected_num_regions=expected_num_regions
    )
    patches_384 = [preprocess_patch(p, size=patch_size) for p in patches_raw]
    region_ids = [f"R{i+1}" for i in range(len(patches_384))]
    return PreprocessResult(
        patches=patches_384,
        bboxes=bboxes,
        region_ids=region_ids,
        used_fallback_grid=used_fallback,
        diagnostics={
            "num_regions": len(patches_384),
            "expected_num_regions": expected_num_regions,
            "patch_size": patch_size,
        },
    )
