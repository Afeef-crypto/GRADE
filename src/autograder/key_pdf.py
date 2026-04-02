"""
Answer key ingestion from text-based PDFs.

Extracts text with pypdf, then splits on numbered sections (1. 2. Q1:, etc.).
"""

from __future__ import annotations

import re
from io import BytesIO
from typing import List, Tuple

from autograder.schemas import AnswerKeyItemIn, UploadKeyRequest


def extract_text_from_pdf_bytes(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise RuntimeError("Install PyPDF to use PDF answer keys: pip install pypdf") from e

    reader = PdfReader(BytesIO(data))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise ValueError("Encrypted PDFs are not supported; provide an unlocked key PDF.") from exc
    chunks: List[str] = []
    for page in reader.pages:
        chunks.append(page.extract_text() or "")
    return "\n\n".join(chunks).strip()


def _parse_max_marks(body: str, default: float) -> Tuple[str, float]:
    for pat in (
        r"(?im)^\s*max\s*marks?\s*[:=]\s*(\d+(?:\.\d+)?)\s*$",
        r"(?im)max\s*marks?\s*[:=]\s*(\d+(?:\.\d+)?)",
    ):
        m = re.search(pat, body)
        if m:
            val = float(m.group(1))
            body = re.sub(pat, "", body, count=1).strip()
            return body, val
    return body, default


# Line-start: "1. ..." / "2) ..." / "Q3: ..." / "Question 4."
_SECTION_START = re.compile(
    r"(?m)^\s*(?:Q(?:uestion)?\s*(\d+)|(\d+))\s*[.:)]\s*",
    re.IGNORECASE,
)


def text_to_upload_request(text: str, exam_id: str, default_max_marks: float = 4.0) -> UploadKeyRequest:
    """
    Turn plain text (from PDF or else) into UploadKeyRequest.

    Splits on lines that start a new item: ``1.``, ``2)``, ``Q3:``, etc.
    If no numbered blocks are found, the entire text becomes a single question ``Q1``.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise ValueError("Answer key PDF contained no extractable text (try a text-based PDF, not a scan).")

    matches = list(_SECTION_START.finditer(text))
    if not matches:
        questions = [
            AnswerKeyItemIn(
                question_id="Q1",
                expected_answer=text,
                max_marks=default_max_marks,
            )
        ]
        return UploadKeyRequest(exam_id=exam_id, questions=questions)

    spans: List[Tuple[int, int, str]] = []
    for i, m in enumerate(matches):
        num = m.group(1) or m.group(2)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        spans.append((m.start(), end, num))

    preamble = text[: spans[0][0]].strip()
    items: List[AnswerKeyItemIn] = []
    for idx, (start, end, num) in enumerate(spans):
        segment = text[start:end].strip()
        body = _SECTION_START.sub("", segment, count=1).strip()
        if idx == 0 and preamble:
            body = f"{preamble}\n\n{body}".strip()
        body, mx = _parse_max_marks(body, default_max_marks)
        items.append(
            AnswerKeyItemIn(
                question_id=f"Q{num}",
                expected_answer=body,
                max_marks=mx,
            )
        )
    return UploadKeyRequest(exam_id=exam_id, questions=items)


def pdf_bytes_to_upload_request(
    data: bytes,
    exam_id: str,
    default_max_marks: float = 4.0,
) -> UploadKeyRequest:
    raw = extract_text_from_pdf_bytes(data)
    return text_to_upload_request(raw, exam_id, default_max_marks)
