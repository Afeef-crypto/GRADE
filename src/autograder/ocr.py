"""
OCR / HTR Layer for GRADE — Phase 2.

Three-tier orchestration: Primary **Google Cloud Vision** (``document_text_detection``) or Azure,
then PaddleOCR, then TrOCR — unless ``GRADE_OCR_CLOUD_ONLY`` is set (cloud APIs only).
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
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Confidence below this threshold triggers a flag for examiner review (proposal §5).
CONFIDENCE_FLAG_THRESHOLD = 0.60

# Minimum recognised text length; below this we treat output as empty.
MIN_TEXT_LENGTH = 1

# Maximum OCR transcript length (full-page handwriting can exceed 2k chars).
MAX_TEXT_LENGTH = 16384

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


def _vision_vertex_sort_key(block) -> Tuple[float, float]:
    """Sort blocks top-to-bottom, left-to-right using bbox min y then min x."""
    try:
        bb = block.bounding_box
        if not bb or not bb.vertices:
            return (0.0, 0.0)
        ys = [float(v.y) for v in bb.vertices]
        xs = [float(v.x) for v in bb.vertices]
        return (min(ys), min(xs))
    except (AttributeError, TypeError, ValueError, IndexError):
        return (0.0, 0.0)


def _vision_word_string(word) -> str:
    try:
        if getattr(word, "symbols", None):
            return "".join(s.text for s in word.symbols if getattr(s, "text", None))
        return getattr(word, "text", "") or ""
    except (AttributeError, TypeError):
        return ""


def _text_and_confidences_from_full_annotation(annotation) -> Optional[Tuple[str, List[float]]]:
    """
    Reconstruct transcript by sorting layout blocks in reading order.

    Google's ``annotation.text`` can interleave columns; geometry order usually reads
    more like the page (fewer \"missing\" tails when lines wrap oddly).
    """
    if not annotation or not annotation.pages:
        return None
    try:
        all_confidences: List[float] = []
        page_texts: List[str] = []
        for page in annotation.pages:
            blocks = list(page.blocks)
            blocks.sort(key=_vision_vertex_sort_key)
            block_chunks: List[str] = []
            for block in blocks:
                for para in block.paragraphs:
                    words_out: List[str] = []
                    for word in para.words:
                        wtxt = _vision_word_string(word)
                        if wtxt:
                            words_out.append(wtxt)
                        try:
                            all_confidences.append(float(word.confidence))
                        except (TypeError, ValueError, AttributeError):
                            all_confidences.append(0.0)
                    if words_out:
                        block_chunks.append(" ".join(words_out))
            if block_chunks:
                page_texts.append("\n".join(block_chunks))
        if not page_texts:
            return None
        return ("\n\n".join(page_texts), all_confidences)
    except (AttributeError, TypeError, ValueError) as e:
        logger.debug("Vision geometry transcript skipped: %s", e)
        return None


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

    prepared = prepare_image_for_cloud_ocr(image)
    content = _image_to_bytes(prepared)
    gvision_image = vision.Image(content=content)

    response = client.document_text_detection(image=gvision_image)  # type: ignore

    if response.error.message:
        raise RuntimeError(f"Google Vision API error: {response.error.message}")

    annotation = response.full_text_annotation
    if not annotation:
        return _make_result("", 0.0, "google")

    geo = _text_and_confidences_from_full_annotation(annotation)
    use_plain = _env_truthy("GRADE_OCR_VISION_USE_PLAIN_TEXT")

    if geo and not use_plain:
        text_body, confidences = geo
        if not (text_body or "").strip():
            text_body = annotation.text or ""
    else:
        text_body = annotation.text or ""
        confidences = []
        for page in annotation.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        try:
                            confidences.append(float(word.confidence))
                        except (TypeError, ValueError, AttributeError):
                            confidences.append(0.0)

    if not (text_body or "").strip():
        return _make_result("", 0.0, "google")

    avg_conf = float(np.mean(confidences)) if confidences else 0.5
    return _make_result(text_body, avg_conf, "google")


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

    prepared = prepare_image_for_cloud_ocr(image)
    image_bytes = _image_to_bytes(prepared)
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


_google_adc_available: Optional[bool] = None


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_falsy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"0", "false", "no", "off"}


def _parse_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _cloud_clahe_enabled() -> bool:
    """CLAHE on LAB luminance before Google/Azure unless GRADE_OCR_ENHANCE_CLOUD=0."""
    return not _env_falsy("GRADE_OCR_ENHANCE_CLOUD")


def _cloud_upscale_enabled() -> bool:
    """Upscale small patches so Vision sees enough pixels (default on)."""
    return not _env_falsy("GRADE_OCR_CLOUD_UPSCALE")


def _cloud_inpaint_grey_enabled() -> bool:
    """Inpaint flat grey blocks (PDF redactions) if GRADE_OCR_INPAINT_GREY_REDACTIONS=1."""
    return _env_truthy("GRADE_OCR_INPAINT_GREY_REDACTIONS")


def _inpaint_flat_grey_regions(bgr: np.ndarray) -> np.ndarray:
    """
    Heuristic: large, low-texture, mid-grey regions (common redaction overlays).
    Conservative — skips if mask too small.
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    b_ch, g_ch, r_ch = cv2.split(bgr)
    diff = np.maximum(
        np.abs(b_ch.astype(np.int16) - g_ch),
        np.maximum(np.abs(g_ch.astype(np.int16) - r_ch), np.abs(b_ch.astype(np.int16) - r_ch)),
    )
    uniform = diff < 30
    mid = (gray > 90) & (gray < 218)
    mask = ((uniform & mid).astype(np.uint8)) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    min_area = max(1200, (h * w) // 150)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    mask_pixels = np.count_nonzero(out)
    if mask_pixels < min_area:
        return bgr
    # Avoid wiping a large fraction of the page (lined paper / faint ink can look "grey")
    if mask_pixels > (h * w) * 0.18:
        return bgr
    return cv2.inpaint(bgr, out, 8, cv2.INPAINT_NS)


def _clahe_bgr(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l2 = clahe.apply(l_ch)
    merged = cv2.merge((l2, a_ch, b_ch))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _upscale_for_cloud(image: np.ndarray) -> np.ndarray:
    """Ensure the longer side is at least MIN_MAX_DIM (Vision reads small 384² patches poorly)."""
    min_max = _parse_positive_int("GRADE_OCR_CLOUD_MIN_MAX_DIM", 1600)
    cap_max = _parse_positive_int("GRADE_OCR_CLOUD_MAX_MAX_DIM", 4096)
    hh, ww = image.shape[:2]
    if hh < 1 or ww < 1:
        return image
    cur = max(hh, ww)
    if cur >= min_max:
        if cur > cap_max:
            scale = cap_max / cur
            return cv2.resize(
                image,
                (max(1, int(round(ww * scale))), max(1, int(round(hh * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        return image
    scale = min_max / cur
    nw = int(round(ww * scale))
    nh = int(round(hh * scale))
    if max(nw, nh) > cap_max:
        scale2 = cap_max / max(nw, nh)
        nw = max(1, int(round(nw * scale2)))
        nh = max(1, int(round(nh * scale2)))
    return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)


def prepare_image_for_cloud_ocr(image: np.ndarray) -> np.ndarray:
    """
    Normalize to BGR uint8, optionally remove grey redaction blocks, CLAHE, upscale for Vision/Azure.

    Controlled by env:
      GRADE_OCR_ENHANCE_CLOUD (default on) — CLAHE + min resolution
      GRADE_OCR_INPAINT_GREY_REDACTIONS — conservative inpaint (off unless set)
      GRADE_OCR_CLOUD_MIN_MAX_DIM / GRADE_OCR_CLOUD_MAX_MAX_DIM
    """
    img = np.asarray(image)
    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim == 3 and img.shape[2] == 3:
        bgr = img.copy()
    else:
        raise ValueError("OCR image must be HxW grayscale or HxWx3/4 BGR/RGBA.")

    if img.dtype != np.uint8:
        bgr = np.clip(bgr, 0, 255).astype(np.uint8)

    if _cloud_inpaint_grey_enabled():
        bgr = _inpaint_flat_grey_regions(bgr)

    if _cloud_clahe_enabled():
        bgr = _clahe_bgr(bgr)

    if _cloud_upscale_enabled():
        bgr = _upscale_for_cloud(bgr)

    return bgr


def _google_vision_adc_available() -> bool:
    """True if Application Default Credentials can be obtained (optional opt-in via env)."""
    global _google_adc_available
    if _google_adc_available is not None:
        return _google_adc_available
    try:
        from google.auth import default as google_auth_default
        from google.auth.exceptions import DefaultCredentialsError

        google_auth_default(scopes=["https://www.googleapis.com/auth/cloud-vision"])
        _google_adc_available = True
    except Exception:
        _google_adc_available = False
    return _google_adc_available


def google_vision_configured() -> bool:
    """
    Whether the app will attempt Google Cloud Vision (env key, service-account JSON path, or ADC).

    - ``GOOGLE_CLOUD_VISION_API_KEY`` — REST API key.
    - ``GOOGLE_APPLICATION_CREDENTIALS`` — path to service account JSON (invalid paths fail at API call).
    - ``GRADE_GOOGLE_VISION_USE_ADC=1`` — use ``gcloud auth application-default login`` (or GCE metadata).
    """
    if os.environ.get("GOOGLE_CLOUD_VISION_API_KEY", "").strip():
        return True
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip():
        return True
    if _env_truthy("GRADE_GOOGLE_VISION_USE_ADC"):
        return _google_vision_adc_available()
    return False


def google_vision_credentials_file_ok() -> bool:
    """True if ``GOOGLE_APPLICATION_CREDENTIALS`` points to an existing file."""
    raw = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if not raw:
        return False
    try:
        return Path(raw).expanduser().is_file()
    except OSError:
        return False


def _ocr_cloud_google_only() -> bool:
    """When set, do not fall back to Azure after Google errors (``GRADE_OCR_GOOGLE_ONLY``)."""
    return _env_truthy("GRADE_OCR_GOOGLE_ONLY")


def _ocr_cloud(image: np.ndarray) -> OCRResult:
    """
    Try Google Cloud Vision first when configured; otherwise Azure (unless Google-only mode).

    Google is used when any of: API key, ``GOOGLE_APPLICATION_CREDENTIALS``,
    or ``GRADE_GOOGLE_VISION_USE_ADC=1`` with working ADC.

    Set ``GRADE_OCR_GOOGLE_ONLY=1`` to use only Google for cloud OCR (no Azure fallback).
    """
    errors: List[str] = []
    google_only = _ocr_cloud_google_only()

    if google_only and not google_vision_configured():
        raise RuntimeError(
            "GRADE_OCR_GOOGLE_ONLY is set but Google Vision is not configured "
            "(set GOOGLE_CLOUD_VISION_API_KEY, GOOGLE_APPLICATION_CREDENTIALS, or "
            "GRADE_GOOGLE_VISION_USE_ADC=1 with working Application Default Credentials)."
        )

    if google_vision_configured():
        try:
            return _ocr_google(image)
        except Exception as exc:
            logger.warning("Google Vision failed: %s", exc)
            errors.append(f"Google: {exc}")
            if google_only:
                raise RuntimeError(
                    "Google Vision failed and GRADE_OCR_GOOGLE_ONLY is set (Azure fallback disabled). "
                    + "; ".join(errors)
                ) from exc

    if not google_only and os.environ.get("AZURE_VISION_ENDPOINT") and os.environ.get(
        "AZURE_VISION_KEY"
    ):
        try:
            return _ocr_azure(image)
        except Exception as exc:
            logger.warning("Azure Vision failed: %s", exc)
            errors.append(f"Azure: {exc}")

    if not errors:
        raise RuntimeError(
            "Cloud OCR not configured. Set GOOGLE_APPLICATION_CREDENTIALS (path to JSON), "
            "or GOOGLE_CLOUD_VISION_API_KEY, or GRADE_GOOGLE_VISION_USE_ADC=1 after "
            "`gcloud auth application-default login`. Optional: Azure with AZURE_VISION_*."
        )

    msg = "Cloud OCR unavailable or failed. Errors: " + "; ".join(errors)
    if "ACCOUNT_STATE_INVALID" in msg:
        msg += (
            " — For ACCOUNT_STATE_INVALID: use a **standard** Google Cloud project where **Cloud Vision API** "
            "is enabled (IAM → service account → new JSON key). Credentials from `gen-lang-client-*` / "
            "AI-only projects often cannot call Cloud Vision; create a separate project or use "
            "`GOOGLE_CLOUD_VISION_API_KEY` (APIs & Services → Credentials → API key, restrict to Vision)."
        )
    raise RuntimeError(msg)


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
    # Some PaddleOCR builds reject show_log.
    try:
        ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    except Exception as e:
        if "show_log" not in str(e).lower():
            raise
        ocr = PaddleOCR(use_angle_cls=True, lang="en")
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


def _ocr_cloud_only_mode() -> bool:
    """If true, only run cloud OCR (Google/Azure); no PaddleOCR or TrOCR (``GRADE_OCR_CLOUD_ONLY``)."""
    return _env_truthy("GRADE_OCR_CLOUD_ONLY")


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
      1. Cloud API (Google Cloud Vision preferred, then Azure unless GRADE_OCR_GOOGLE_ONLY)
      2. PaddleOCR  (local; skipped if GRADE_OCR_CLOUD_ONLY)
      3. TrOCR      (local; skipped if GRADE_OCR_CLOUD_ONLY)

    Set ``GRADE_OCR_CLOUD_ONLY=1`` to use only cloud APIs (no local OCR fallbacks).

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
    ]
    if not _ocr_cloud_only_mode():
        tiers.extend(
            [
                ("paddle", lambda: _ocr_paddle(image)),
                (
                    "trocr",
                    lambda: _ocr_trocr(
                        image, model_name=trocr_model_name, beam_width=trocr_beam_width
                    ),
                ),
            ]
        )

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
