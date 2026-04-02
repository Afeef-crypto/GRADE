"""Generate evaluation PDF reports (Phase 5)."""

from __future__ import annotations

import io
from typing import Any, Dict, List


def build_evaluation_pdf(result_row: Dict[str, Any]) -> bytes:
    """
    Build a PDF bytes buffer from a DB result row (as returned by get_result).
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as e:
        raise RuntimeError("reportlab is required for PDF reports: pip install reportlab") from e

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, title="GRADE Evaluation Report")
    styles = getSampleStyleSheet()
    story: List[Any] = []

    story.append(Paragraph("<b>GRADE — Evaluation Report</b>", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(
        Paragraph(
            f"Result ID: {result_row.get('id', '')}<br/>"
            f"Exam: {result_row.get('exam_id', '')}<br/>"
            f"Total: <b>{result_row.get('total_marks')}</b> / {result_row.get('max_total')}<br/>"
            f"Confidence flag: {result_row.get('confidence_flag')}<br/>"
            f"Grading confidence: {result_row.get('grading_confidence')}<br/>"
            f"LLM model: {result_row.get('llm_model')}<br/>"
            f"Prompt hash: {result_row.get('prompt_hash', '')[:48]}…",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 18))

    questions: List[Dict[str, Any]] = result_row.get("questions", [])
    table_data = [
        [
            "Q",
            "Awarded",
            "Max",
            "FA",
            "CC",
            "RE",
            "DT",
            "OCR",
            "GC",
            "Flags",
        ]
    ]
    for q in questions:
        rs = q.get("rubric_scores", {})
        flags = ", ".join(q.get("flags") or [])[:40]
        table_data.append(
            [
                str(q.get("question_id", "")),
                str(q.get("awarded_marks", "")),
                str(q.get("max_marks", "")),
                str(rs.get("factual_accuracy", "")),
                str(rs.get("conceptual_completeness", "")),
                str(rs.get("reasoning", "")),
                str(rs.get("domain_terminology", "")),
                f"{float(q.get('ocr_confidence', 0)):.2f}",
                str(q.get("grading_confidence", "")),
                flags,
            ]
        )

    t = Table(table_data, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 18))

    story.append(Paragraph("<b>Per-question feedback</b>", styles["Heading2"]))
    for q in questions:
        story.append(
            Paragraph(
                f"<b>{q.get('question_id')}</b> — OCR text: {q.get('student_answer', '')[:500]}",
                styles["Normal"],
            )
        )
        story.append(Paragraph(q.get("feedback", ""), styles["Italic"]))
        story.append(Spacer(1, 8))

    doc.build(story)
    return buf.getvalue()
