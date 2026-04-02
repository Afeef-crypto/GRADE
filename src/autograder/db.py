from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = Path("data/grade.db")


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sheets (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            path TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS answer_keys (
            id TEXT PRIMARY KEY,
            exam_id TEXT NOT NULL,
            question_id TEXT NOT NULL,
            expected_answer TEXT NOT NULL,
            embedding TEXT,
            max_marks REAL NOT NULL,
            domain TEXT DEFAULT 'general',
            rubric_override TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS evaluation_results (
            id TEXT PRIMARY KEY,
            sheet_id TEXT NOT NULL,
            exam_id TEXT NOT NULL,
            per_question_scores TEXT NOT NULL,
            total_marks REAL NOT NULL,
            max_total REAL NOT NULL,
            confidence_flag INTEGER DEFAULT 0,
            grading_confidence TEXT,
            ocr_engine_used TEXT,
            prompt_hash TEXT,
            llm_model TEXT,
            flags TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def insert_sheet(filename: str, path: str) -> str:
    sheet_id = str(uuid.uuid4())
    conn = get_conn()
    conn.execute(
        "INSERT INTO sheets(id, filename, path) VALUES (?, ?, ?)",
        (sheet_id, filename, path),
    )
    conn.commit()
    conn.close()
    return sheet_id


def get_sheet(sheet_id: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    row = conn.execute("SELECT * FROM sheets WHERE id = ?", (sheet_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def insert_answer_key(
    exam_id: str,
    question_id: str,
    expected_answer: str,
    embedding: List[float],
    max_marks: float,
    domain: str,
    rubric_override: Optional[dict],
) -> str:
    key_id = str(uuid.uuid4())
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO answer_keys(id, exam_id, question_id, expected_answer, embedding, max_marks, domain, rubric_override)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            key_id,
            exam_id,
            question_id,
            expected_answer,
            json.dumps(embedding),
            max_marks,
            domain,
            json.dumps(rubric_override) if rubric_override is not None else None,
        ),
    )
    conn.commit()
    conn.close()
    return key_id


def list_answer_keys(exam_id: str) -> List[Dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM answer_keys WHERE exam_id = ? ORDER BY question_id ASC", (exam_id,)
    ).fetchall()
    conn.close()
    out: List[Dict[str, Any]] = []
    for r in rows:
        item = dict(r)
        item["embedding"] = json.loads(item["embedding"]) if item.get("embedding") else []
        item["rubric_override"] = (
            json.loads(item["rubric_override"]) if item.get("rubric_override") else None
        )
        out.append(item)
    return out


def insert_evaluation_result(payload: Dict[str, Any]) -> str:
    result_id = str(uuid.uuid4())
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO evaluation_results(
            id, sheet_id, exam_id, per_question_scores, total_marks, max_total,
            confidence_flag, grading_confidence, ocr_engine_used, prompt_hash,
            llm_model, flags
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result_id,
            payload["sheet_id"],
            payload["exam_id"],
            json.dumps(payload["questions"]),
            payload["total_marks"],
            payload["max_total"],
            1 if payload["confidence_flag"] else 0,
            payload["grading_confidence"],
            payload.get("ocr_engine_used", "mixed"),
            payload["prompt_hash"],
            payload["llm_model"],
            json.dumps(payload.get("flags", [])),
        ),
    )
    conn.commit()
    conn.close()
    return result_id


def get_result(result_id: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    row = conn.execute("SELECT * FROM evaluation_results WHERE id = ?", (result_id,)).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    d["questions"] = json.loads(d["per_question_scores"])
    d["flags"] = json.loads(d["flags"] or "[]")
    d["confidence_flag"] = bool(d["confidence_flag"])
    return d
