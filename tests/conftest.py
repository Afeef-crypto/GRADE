"""Pytest fixtures for AutoGrader tests."""

import numpy as np
import pytest


@pytest.fixture
def synthetic_binary_sheet():
    """
    Synthetic answer sheet: white image with 3 black rectangular "answer boxes".
    Simulates a simple layout for contour-based segmentation.
    """
    img = np.ones((600, 400), dtype=np.uint8) * 255
    # Three boxes: (x, y, w, h) - black (0) on white (255)
    boxes = [(50, 80, 300, 120), (50, 250, 300, 120), (50, 420, 300, 120)]
    for (x, y, w, h) in boxes:
        img[y : y + h, x : x + w] = 0
    return img


@pytest.fixture
def synthetic_grayscale_sheet():
    """Grayscale version with slight noise (still clear boxes)."""
    img = np.ones((600, 400), dtype=np.uint8) * 240
    boxes = [(50, 80, 300, 120), (50, 250, 300, 120), (50, 420, 300, 120)]
    for (x, y, w, h) in boxes:
        img[y : y + h, x : x + w] = 20
    return img


@pytest.fixture
def small_patch():
    """Small grayscale patch for preprocess_patch tests."""
    return np.random.randint(0, 256, (100, 150), dtype=np.uint8)
