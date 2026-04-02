from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class AnswerKeyItemIn(BaseModel):
    question_id: str
    expected_answer: str
    max_marks: float = Field(gt=0)
    domain: str = "general"
    rubric_override: Optional[dict] = None


class UploadKeyRequest(BaseModel):
    exam_id: str
    questions: List[AnswerKeyItemIn]


class UploadKeyResponse(BaseModel):
    key_ids: List[str]


class UploadSheetResponse(BaseModel):
    sheet_id: str
    filename: str


class EvaluateRequest(BaseModel):
    sheet_id: str
    exam_id: str
    top_k: int = Field(default=3, ge=1, le=10)
    expected_num_regions: Optional[int] = None
    use_consensus_ocr: bool = False


class RubricScores(BaseModel):
    factual_accuracy: int = Field(ge=0, le=4)
    conceptual_completeness: int = Field(ge=0, le=4)
    reasoning: int = Field(ge=0, le=4)
    domain_terminology: int = Field(ge=0, le=4)


class EvaluationResult(BaseModel):
    awarded_marks: float = Field(ge=0)
    max_marks: float = Field(ge=0)
    rubric_scores: RubricScores
    feedback: str
    grading_confidence: str
    flags: List[str] = Field(default_factory=list)

    @field_validator("grading_confidence")
    @classmethod
    def validate_confidence(cls, v: str) -> str:
        if v not in {"high", "medium", "low"}:
            raise ValueError("grading_confidence must be high|medium|low")
        return v


class QuestionResult(BaseModel):
    question_id: str
    student_answer: str
    awarded_marks: float
    max_marks: float
    rubric_scores: RubricScores
    feedback: str
    grading_confidence: str
    ocr_confidence: float
    flags: List[str] = Field(default_factory=list)


class EvaluateResponse(BaseModel):
    result_id: str
    sheet_id: str = ""
    exam_id: str = ""
    total_marks: float
    max_total: float
    confidence_flag: bool
    grading_confidence: str
    prompt_hash: str
    llm_model: str
    flags: List[str] = Field(default_factory=list)
    questions: List[QuestionResult]


class RubricBreakdownResponse(BaseModel):
    result_id: str
    dimension_totals: Dict[str, int]
    questions: List[QuestionResult]
