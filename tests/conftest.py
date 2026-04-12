"""Pytest fixtures for GRADE."""

import os

# PostgreSQL is required. Tests use a dedicated DB — start Docker first: `docker compose up -d`
# Override with GRADE_TEST_DATABASE_URL for CI or a custom local instance.
os.environ["GRADE_DATABASE_URL"] = os.environ.get(
    "GRADE_TEST_DATABASE_URL",
    "postgresql://grade:grade@127.0.0.1:5433/grade_test",
)

import numpy as np
import pytest


@pytest.fixture
def client():
    """FastAPI TestClient (uses GRADE_DATABASE_URL from module-level env above)."""
    pytest.importorskip("fastapi")
    from starlette.testclient import TestClient

    from autograder.api import app

    with TestClient(app) as c:
        yield c


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
