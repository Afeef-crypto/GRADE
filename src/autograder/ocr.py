"""
OCR / HTR Layer for GRADE — Phase 2.

Three-tier orchestration: Primary (Cloud API) → Fallback 1 (PaddleOCR) → Fallback 2 (TrOCR).
Each tier returns (text, confidence, engine_name). Confidence is normalised to [0, 1].

References:
  [1]  Li et al., "TrOCR," arXiv:2109.10282, 2021.
  [3]  Google Cloud Vision API.
  [4]  Azure AI Vision Read API.
  [5]  PaddleOCR.
  [7]  IAM Handwriting Dataset, IJDAR 2002.
  [9]  HuggingFace Transformers.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Confidence below this threshold triggers a flag for examiner review (proposal §5).
CONFIDENCE_FLAG_THRESHOLD = 0.60

# Minimum recognised text length; below this we treat output as empty.
MIN_TEXT_LENGTH = 1

# Maximum tokens/chars fed to SBERT downstream; longer text is truncated.
MAX_TEXT_LENGTH = 2048

# TrOCR beam-search width (proposal §2.2).
TROCR_BEAM_WIDTH = 4

# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


@dataclass
class OCRResult:
    """Result from a single OCR tier call."""

    text: str
    confidence: float  # Normalised [0, 1]
    engine: str        # "google" | "azure" | "paddle" | "trocr"
    low_confidence: bool = False  # True if confidence < CONFIDENCE_FLAG_THRESHOLD
    flags: List[str] | None = None  # e.g. ["ocr_uncertainty", "review_required"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _image_to_bytes(image: np.ndarray, ext: str = ".png") -> bytes:
    """Encode numpy array to PNG/JPEG bytes."""
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise ValueError("Failed to encode image to bytes.")
    return buf.tobytes()


def _image_to_base64(image: np.ndarray) -> str:
    return base64.b64encode(_image_to_bytes(image)).decode("utf-8")


def _sanitise_text(text: str) -> str:
    """Strip leading/trailing whitespace; truncate to MAX_TEXT_LENGTH."""
    text = text.strip()
    if len(text) > MAX_TEXT_LENGTH:
        logger.warning("OCR text truncated from %d to %d chars.", len(text), MAX_TEXT_LENGTH)
        text = text[:MAX_TEXT_LENGTH]
    return text


def _normalize_confidence(confidence: float) -> float:
    """Clamp confidence into [0, 1] and coerce invalid inputs."""
    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        return 0.0
    if np.isnan(conf) or np.isinf(conf):
        return 0.0
    return max(0.0, min(1.0, conf))


def _make_result(text: str, confidence: float, engine: str) -> OCRResult:
    confidence = _normalize_confidence(confidence)
    text = _sanitise_text(text)
    low = confidence < CONFIDENCE_FLAG_THRESHOLD
    flags: List[str] = []
    if not text:
        flags.append("empty_text")
    if low:
        flags.extend(["ocr_uncertainty", "review_required"])
        logger.info("Engine '%s' returned low confidence %.3f.", engine, confidence)
    return OCRResult(
        text=text,
        confidence=confidence,
        engine=engine,
        low_confidence=low,
        flags=flags,
    )


# ---------------------------------------------------------------------------
# Tier 1 — Google Cloud Vision
# ---------------------------------------------------------------------------


def _ocr_google(image: np.ndarray) -> OCRResult:
    """
    Call Google Cloud Vision DOCUMENT_TEXT_DETECTION.
    Requires GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_CLOUD_VISION_API_KEY env var.

    Confidence: Google returns per-paragraph confidence. We average them.
    Raises RuntimeError on any failure (triggers fallback in orchestrator).
    """
    try:
        from google.cloud import vision  # type: ignore
    except ImportError as exc:
        raise RuntimeError("google-cloud-vision not installed.") from exc

    api_key = os.environ.get("GOOGLE_CLOUD_VISION_API_KEY")
    if api_key:
        client = vision.ImageAnnotatorClient(
            client_options={"api_key": api_key}
        )
    else:
        # Uses GOOGLE_APPLICATION_CREDENTIALS JSON file.
        client = vision.ImageAnnotatorClient()

    content = _image_to_bytes(image)
    gvision_image = vision.Image(content=content)

    response = client.document_text_detection(image=gvision_image)  # type: ignore

    if response.error.message:
        raise RuntimeError(f"Google Vision API error: {response.error.message}")

    annotation = response.full_text_annotation
    if not annotation or not annotation.text:
        return _make_result("", 0.0, "google")

    # Gather per-word confidences from the response hierarchy.
    confidences: List[float] = []
    for page in annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    confidences.append(word.confidence)

    avg_conf = float(np.mean(confidences)) if confidences else 0.5
    return _make_result(annotation.text, avg_conf, "google")


# ---------------------------------------------------------------------------
# Tier 1 (alt) — Azure AI Vision Read API
# ---------------------------------------------------------------------------


def _ocr_azure(image: np.ndarray) -> OCRResult:
    """
    Call Azure AI Vision Read API (handwriting mode).
    Requires AZURE_VISION_ENDPOINT and AZURE_VISION_KEY env vars.

    Confidence: Azure returns per-word confidence. We average them.
    Raises RuntimeError on any failure (triggers fallback in orchestrator).
    """
    try:
        from azure.ai.vision.imageanalysis import ImageAnalysisClient  # type: ignore
        from azure.ai.vision.imageanalysis.models import VisualFeatures  # type: ignore
        from azure.core.credentials import AzureKeyCredential  # type: ignore
    except ImportError as exc:
        raise RuntimeError("azure-ai-vision-imageanalysis not installed.") from exc

    endpoint = os.environ.get("AZURE_VISION_ENDPOINT")
    key = os.environ.get("AZURE_VISION_KEY")
    if not endpoint or not key:
        raise RuntimeError(
            "AZURE_VISION_ENDPOINT and AZURE_VISION_KEY must be set."
        )

    client = ImageAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    image_bytes = _image_to_bytes(image)
    result = client.analyze(
        image_data=image_bytes,
        visual_features=[VisualFeatures.READ],
    )

    if result.read is None or not result.read.blocks:
        return _make_result("", 0.0, "azure")

    lines_text: List[str] = []
    confidences: List[float] = []
    for block in result.read.blocks:
        for line in block.lines:
            lines_text.append(line.text)
            for word in line.words:
                confidences.append(word.confidence)

    text = "\n".join(lines_text)
    avg_conf = float(np.mean(confidences)) if confidences else 0.5
    return _make_result(text, avg_conf, "azure")


def _ocr_cloud(image: np.ndarray) -> OCRResult:
    """
    Try Google Vision first; if not configured, try Azure.
    The tier that is configured (env vars present) wins.
    Raises RuntimeError if neither is available/configured.
    """
    errors: List[str] = []

    # Prefer Google if credentials are present.
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.environ.get(
        "GOOGLE_CLOUD_VISION_API_KEY"
    ):
        try:
            return _ocr_google(image)
        except Exception as exc:
            logger.warning("Google Vision failed: %s", exc)
            errors.append(f"Google: {exc}")

    # Try Azure.
    if os.environ.get("AZURE_VISION_ENDPOINT") and os.environ.get("AZURE_VISION_KEY"):
        try:
            return _ocr_azure(image)
        except Exception as exc:
            logger.warning("Azure Vision failed: %s", exc)
            errors.append(f"Azure: {exc}")

    raise RuntimeError(
        "Cloud OCR unavailable or not configured. Errors: " + "; ".join(errors)
    )


# ---------------------------------------------------------------------------
# Tier 2 — PaddleOCR (local)
# ---------------------------------------------------------------------------


def _ocr_paddle(image: np.ndarray) -> OCRResult:
    """
    Run PaddleOCR locally on the patch.
    Requires paddleocr and paddlepaddle packages.

    Confidence: PaddleOCR returns per-line confidence. We average them.
    Raises RuntimeError on failure (triggers TrOCR fallback).

    Reference: [5] PaddlePaddle, PaddleOCR.
    """
    try:
        from paddleocr import PaddleOCR  # type: ignore
    except ImportError as exc:
        raise RuntimeError("paddleocr not installed.") from exc

    # lang='en' for English; PaddleOCR supports multi-language.
    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    result = ocr.ocr(image, cls=True)

    if not result or result == [None]:
        return _make_result("", 0.0, "paddle")

    lines_text: List[str] = []
    confidences: List[float] = []

    for line_group in result:
        if line_group is None:
            continue
        for detection in line_group:
            if detection is None:
                continue
            # detection: [[bbox_points], (text, confidence)]
            try:
                text_part = detection[1]
                lines_text.append(text_part[0])
                confidences.append(float(text_part[1]))
            except (IndexError, TypeError, ValueError):
                continue

    text = "\n".join(lines_text)
    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    return _make_result(text, avg_conf, "paddle")


# ---------------------------------------------------------------------------
# Tier 3 — TrOCR (HuggingFace)
# ---------------------------------------------------------------------------

# Module-level cache so the model loads only once per process.
_trocr_processor = None
_trocr_model = None
_trocr_model_name: str = ""


def _load_trocr(model_name: str = "microsoft/trocr-base-handwritten") -> None:
    """Load TrOCR processor and model once; cache at module level."""
    global _trocr_processor, _trocr_model, _trocr_model_name
    if _trocr_model is not None and _trocr_model_name == model_name:
        return
    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # type: ignore
    except ImportError as exc:
        raise RuntimeError("transformers not installed.") from exc

    logger.info("Loading TrOCR model '%s'…", model_name)
    _trocr_processor = TrOCRProcessor.from_pretrained(model_name)
    _trocr_model = VisionEncoderDecoderModel.from_pretrained(model_name)
    _trocr_model_name = model_name
    logger.info("TrOCR model loaded.")


def _aggregate_trocr_confidence(scores: list) -> float:
    """
    Aggregate per-token log-probabilities from beam-search sequences scores
    into a single [0, 1] confidence value.

    `scores` is a list of torch.Tensor (one per decoding step), each of shape
    (batch=1, vocab_size). We take the max log-prob at each step as the
    per-token confidence and average them.
    """
    try:
        import torch  # type: ignore

        token_confs: List[float] = []
        for step_scores in scores:
            # step_scores: (1, vocab_size) log-probs after log-softmax
            probs = torch.softmax(step_scores[0], dim=-1)
            max_prob = float(probs.max().item())
            token_confs.append(max_prob)
        if not token_confs:
            return 0.5
        return float(np.mean(token_confs))
    except Exception:
        return 0.5  # fallback if score extraction fails


def _ocr_trocr(
    image: np.ndarray,
    *,
    model_name: str = "microsoft/trocr-base-handwritten",
    beam_width: int = TROCR_BEAM_WIDTH,
) -> OCRResult:
    """
    Run TrOCR inference with beam-search decoding.

    The patch (grayscale or BGR, any size) is converted to PIL RGB before
    passing to the processor — TrOCR expects a 3-channel image.

    Confidence: averaged max-prob across beam-search decoding steps.

    References: [1] Li et al. 2021; [9] HuggingFace Transformers.
    Raises RuntimeError on failure.
    """
    try:
        from PIL import Image as PilImage  # type: ignore
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Pillow or torch not installed.") from exc

    _load_trocr(model_name)

    # Convert to 3-channel PIL image.
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = PilImage.fromarray(rgb)

    pixel_values = _trocr_processor(images=pil_image, return_tensors="pt").pixel_values

    with torch.no_grad():
        outputs = _trocr_model.generate(
            pixel_values,
            num_beams=beam_width,
            output_scores=True,
            return_dict_in_generate=True,
        )

    sequences = outputs.sequences
    generated_ids = sequences[0]
    generated_text = _trocr_processor.tokenizer.decode(
        generated_ids, skip_special_tokens=True
    )

    # Extract confidence from beam scores.
    scores = outputs.scores if hasattr(outputs, "scores") else []
    confidence = _aggregate_trocr_confidence(list(scores))

    return _make_result(generated_text, confidence, "trocr")


# ---------------------------------------------------------------------------
# Public API — orchestrator
# ---------------------------------------------------------------------------


def ocr_patch(
    image: np.ndarray,
    *,
    trocr_model_name: str = "microsoft/trocr-base-handwritten",
    trocr_beam_width: int = TROCR_BEAM_WIDTH,
    retry_delay: float = 1.0,
) -> OCRResult:
    """
    OCR a single preprocessed patch via the three-tier fallback chain.

    Tier order:
      1. Cloud API (Google Vision or Azure Vision)
      2. PaddleOCR  (local)
      3. TrOCR      (local, HuggingFace)

    Returns:
        OCRResult with (text, confidence, engine, low_confidence).
        text is empty string if all tiers fail; confidence = 0.0.

    Args:
        image: Preprocessed 384×384 grayscale patch (or any ndarray).
        trocr_model_name: HuggingFace model id for TrOCR.
        trocr_beam_width: Beam-search width for TrOCR (proposal default: 4).
        retry_delay: Seconds to wait between tier attempts.
    """
    tiers: List[Tuple[str, Callable[[], OCRResult]]] = [
        ("cloud", lambda: _ocr_cloud(image)),
        ("paddle", lambda: _ocr_paddle(image)),
        (
            "trocr",
            lambda: _ocr_trocr(
                image, model_name=trocr_model_name, beam_width=trocr_beam_width
            ),
        ),
    ]

    last_error: Optional[Exception] = None
    for tier_name, tier_fn in tiers:
        try:
            result = tier_fn()
            if result.text and len(result.text.strip()) >= MIN_TEXT_LENGTH:
                logger.info(
                    "OCR tier '%s' succeeded (conf=%.3f, engine=%s).",
                    tier_name,
                    result.confidence,
                    result.engine,
                )
                return result
            else:
                logger.warning(
                    "OCR tier '%s' returned empty/short text; trying next tier.",
                    tier_name,
                )
                # Don't immediately retry; treat empty text as a soft failure.
                last_error = ValueError(f"Tier '{tier_name}' produced empty text.")
        except Exception as exc:
            logger.warning("OCR tier '%s' raised: %s", tier_name, exc)
            last_error = exc
            if retry_delay > 0:
                time.sleep(retry_delay)

    # All tiers exhausted — return empty result flagged as low-confidence.
    logger.error("All OCR tiers failed. Last error: %s", last_error)
    return OCRResult(
        text="",
        confidence=0.0,
        engine="none",
        low_confidence=True,
        flags=["ocr_uncertainty", "review_required", "all_tiers_failed"],
    )


def ocr_patch_consensus(
    image: np.ndarray,
    *,
    trocr_model_name: str = "microsoft/trocr-base-handwritten",
    trocr_beam_width: int = TROCR_BEAM_WIDTH,
    retry_delay: float = 1.0,
) -> OCRResult:
    """
    Optional multi-pass OCR consensus mode (v2.1-ready):
    - Try cloud OCR and PaddleOCR.
    - If both succeed with non-empty text, choose higher-confidence output.
    - If only one succeeds, use it.
    - Else fall back to TrOCR via orchestrator.

    This preserves Phase 2 architecture while preparing for consensus upgrades.
    """
    cloud_result: OCRResult | None = None
    paddle_result: OCRResult | None = None

    try:
        cloud_result = _ocr_cloud(image)
    except Exception as exc:
        logger.warning("Consensus cloud pass failed: %s", exc)

    try:
        paddle_result = _ocr_paddle(image)
    except Exception as exc:
        logger.warning("Consensus paddle pass failed: %s", exc)

    valid = [
        r
        for r in (cloud_result, paddle_result)
        if r is not None and r.text and len(r.text.strip()) >= MIN_TEXT_LENGTH
    ]
    if len(valid) == 2:
        best = max(valid, key=lambda r: r.confidence)
        if best.flags is None:
            best.flags = []
        best.flags.append("consensus_mode")
        return best
    if len(valid) == 1:
        best = valid[0]
        if best.flags is None:
            best.flags = []
        best.flags.append("consensus_mode")
        return best

    # Delegate to normal orchestrator to include TrOCR fallback path.
    fallback = ocr_patch(
        image,
        trocr_model_name=trocr_model_name,
        trocr_beam_width=trocr_beam_width,
        retry_delay=retry_delay,
    )
    if fallback.flags is None:
        fallback.flags = []
    fallback.flags.append("consensus_mode")
    return fallback
