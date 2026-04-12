"""Answer key text / PDF parsing helpers."""

from __future__ import annotations

import pytest

from autograder.key_pdf import pdf_bytes_to_upload_request, text_to_upload_request


def test_text_single_block_becomes_q1():
    r = text_to_upload_request("Paris is the capital of France.", "exam-a", 5.0)
    assert r.exam_id == "exam-a"
    assert len(r.questions) == 1
    assert r.questions[0].question_id == "Q1"
    assert r.questions[0].max_marks == 5.0
    assert "Paris" in r.questions[0].expected_answer


def test_text_numbered_sections():
    raw = "Preamble before.\n\n1. First model answer.\n\n2. Second model answer.\n"
    r = text_to_upload_request(raw, "exam-b", 4.0)
    assert len(r.questions) == 2
    assert r.questions[0].question_id == "Q1"
    assert "Preamble" in r.questions[0].expected_answer
    assert "First model" in r.questions[0].expected_answer
    assert r.questions[1].question_id == "Q2"
    assert "Second model" in r.questions[1].expected_answer


def test_max_marks_inline():
    t = "1. Alpha answer.\nMax marks: 10\n\n2. Beta.\n"
    r = text_to_upload_request(t, "e", 4.0)
    assert r.questions[0].max_marks == 10.0
    assert r.questions[1].max_marks == 4.0


def test_empty_text_raises():
    with pytest.raises(ValueError, match="no extractable"):
        text_to_upload_request("", "e", 4.0)


def test_section_marker_does_not_span_lines():
    """Avoid treating subscript-like line breaks (a\\n2\\n.) as a numbered section 2."""
    raw = (
        "1. First answer block.\n"
        "formula (a1,\n"
        "a\n"
        "2\n"
        ".,an)\n\n"
        "2. Second answer block.\n"
    )
    r = text_to_upload_request(raw, "e", 4.0)
    assert len(r.questions) == 2
    assert r.questions[0].question_id == "Q1"
    assert "First answer" in r.questions[0].expected_answer
    assert r.questions[1].question_id == "Q2"
    assert "Second answer" in r.questions[1].expected_answer


def test_pdf_via_reportlab_extracts_sections():
    pytest.importorskip("reportlab")
    pytest.importorskip("pypdf")
    from io import BytesIO

    from reportlab.pdfgen import canvas

    buf = BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(72, 720, "1. First model answer for exam.")
    c.drawString(72, 680, "2. Second model answer here.")
    c.save()
    data = buf.getvalue()
    r = pdf_bytes_to_upload_request(data, "rlexam", 3.0)
    assert r.exam_id == "rlexam"
    assert len(r.questions) == 2
    assert "First model" in r.questions[0].expected_answer
    assert "Second model" in r.questions[1].expected_answer
