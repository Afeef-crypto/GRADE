# GRADE — Implementation Plan (v2.0)

**Version:** 2.0  
**Project:** GRADE — Automatic Handwritten Answer Sheet Evaluator  
**Purpose:** Replace SBERT-only scoring with an LLM rubric-based evaluation core while preserving the OCR pipeline and improving auditability, reliability, and testing rigor.

---

## Summary of v2 Changes

| Area | v1.0 | v2.0 | Status |
|---|---|---|---|
| Scoring core | Sentence-BERT cosine similarity | LLM rubric evaluator (structured JSON) | Replaced |
| Key storage | Text answers only | Text + embeddings (`pgvector`) + `rubric_override` + `domain` | Enhanced |
| Feedback | Similarity + short note | Multi-metric rubric feedback | Enhanced |
| Aggregation | OCR confidence only | OCR confidence + LLM grading confidence | Enhanced |
| Testing | Accuracy/correlation focus | Adds Cohen's kappa, JSON validity, fallback rate | Enhanced |
| Dependencies | SBERT required | SBERT optional fallback, LLM + pgvector required | Updated |

---

## Architecture Direction (v2)

Pipeline remains 8-stage, but Stage 6 changes:

1. **Ingest / Segment / Preprocess** (unchanged): OpenCV pipeline.  
2. **OCR/HTR** (unchanged core): Cloud -> PaddleOCR -> TrOCR with confidence + flags.  
3. **Answer key retrieval** (enhanced): pgvector-backed semantic retrieval for model answers/variants.  
4. **LLM scoring** (new core): rubric-governed scoring via system prompt and strict JSON schema.  
5. **Aggregation/reporting** (enhanced): include rubric breakdown + grading confidence + prompt hash.

---

## Section 2 — LLM Evaluation Prompt Spec

The LLM system prompt is the governance layer for deterministic and auditable grading.

### 2.1 Design Principles

- **Role isolation:** LLM is evaluator only, not tutor.
- **Metric boundedness:** rubric scores constrained to fixed ranges.
- **Structured output only:** strict JSON, no free-form text outside schema.
- **Hallucination control:** low confidence must trigger review flags.
- **Semantic fairness:** paraphrases and clear OCR artifacts should still get credit.

### 2.2 Prompt Template Contract

Store versioned prompt templates in `prompts/grading_v2.txt` and inject runtime fields:

- `{subject_domain}`
- `{question_text}`
- `{max_marks}`
- `{model_answer}`
- `{student_answer}`
- `{ocr_confidence}`

### 2.3 Rubric Dimensions and Weights

| Dimension | Weight | Scale |
|---|---:|---|
| Factual accuracy | 40% | 0-4 |
| Conceptual completeness | 30% | 0-4 |
| Reasoning & explanation | 20% | 0-4 |
| Domain terminology | 10% | 0-4 |

Suggested calculation:

`awarded_marks = round(((fa*0.40 + cc*0.30 + re*0.20 + dt*0.10) / 4.0) * max_marks, 1)`

### 2.4 Required Output Schema

```json
{
  "awarded_marks": 0.0,
  "max_marks": 0.0,
  "rubric_scores": {
    "factual_accuracy": 0,
    "conceptual_completeness": 0,
    "reasoning": 0,
    "domain_terminology": 0
  },
  "feedback": "1-3 sentence explanation",
  "grading_confidence": "high|medium|low",
  "flags": ["ocr_uncertainty", "off_topic", "ambiguous_question", "review_required"]
}
```

### 2.5 Prompt Versioning

- Store prompt files by version (`grading_v2.txt`, future `grading_v2_1.txt`).
- Save `prompt_hash` in results for audit/reproducibility.
- Allow optional per-question `rubric_override` from key JSON.

---

## Phase 1 — Foundation & Image Preprocessing (mostly unchanged)

**Goal:** Reliable patch extraction for downstream OCR and rubric scoring.

### Deliverables

- OpenCV ingest, deskew, segmentation, patch preprocessing.
- Ordered region mapping (`region_ids`) and `bboxes`.
- Preprocessing diagnostics for traceability.

### Extra v2 Requirement

- Ensure Stage 1 output is auditable and stable across runs:
  - deterministic region ordering,
  - persistent region IDs,
  - fallback metadata (`used_fallback_grid`).

### Roadblocks & Mitigations

- **Layout drift:** support template config and controlled fallback.
- **Contour failures:** retry with relaxed params then grid fallback.
- **Region-question mismatch:** preserve region ordering and IDs for Phase 4/5 overlays.

---

## Phase 2 — OCR/HTR Three-Tier Layer (kept, but confidence semantics tightened)

**Goal:** Per-region OCR with robust fallback and normalized confidence for LLM-aware review.

### Deliverables

- Tier chain: Cloud (Google/Azure) -> PaddleOCR -> TrOCR.
- `OCRResult` with text, confidence `[0,1]`, engine, and flags.
- Optional consensus mode (cloud + paddle) for higher reliability.

### v2 Expectations

- Normalize confidence from all engines into `[0,1]`.
- Propagate uncertainty flags (`ocr_uncertainty`, `review_required`) when needed.
- Keep TrOCR fallback and beam-search configuration.

### Roadblocks & Mitigations

- **Quota/API failures:** retry/backoff + fallback chain.
- **CPU latency:** async per patch and optional GPU.
- **Empty OCR text:** fallback to next tier and mark for review.

---

## Phase 3 (Revised) — Answer Key Storage, Embeddings, and LLM Scoring

**Goal:** Replace SBERT-only scoring with retrieval + LLM rubric evaluation.

### 3.1 Deliverables

- PostgreSQL with `pgvector` enabled.
- Embedding ingestion on key upload.
- Semantic retrieval (`top_k`) for answer variants/context.
- `score_answer_llm()` with strict schema validation.
- Retry + fallback logic.

### 3.2 Database Changes

```sql
CREATE EXTENSION IF NOT EXISTS vector;

-- answer_keys additions:
-- embedding vector(1536), domain text, rubric_override jsonb

-- evaluation_results additions:
-- rubric_scores in per-question JSON,
-- grading_confidence text,
-- prompt_hash text,
-- llm_model text,
-- flags text[]
```

### 3.3 Tasks

| # | Task | Notes |
|---|---|---|
| 1 | Add pgvector migration | PostgreSQL 14+ preferred |
| 2 | Embed `expected_answer` on upload | cache repeated text |
| 3 | Implement semantic retrieval | use vector distance operator |
| 4 | Add `score_answer_llm()` | async HTTP, timeout, retries |
| 5 | Validate strict JSON schema | pydantic model |
| 6 | Add fallback path when LLM unavailable | legacy cosine/SBERT, with flags |
| 7 | Store rubric details in result JSON | for dashboard/report |
| 8 | Unit tests for retries/fallback/schema | mock LLM/API responses |

### 3.4 Roadblocks & Mitigations

| Roadblock | Mitigation |
|---|---|
| LLM JSON invalid | Retry up to 3x, then fallback and flag |
| Out-of-range rubric values | Schema validation + retry |
| Embedding API outage | Local embedding fallback path |
| Migration risk | Nullable columns first, backfill later |

### 3.5 Acceptance Criteria

- LLM scoring returns valid schema >99% (with retries).
- Fallback path handles outages without user-facing hard failure.
- Per-question results include rubric scores and confidence fields.

---

## Phase 4 (Updated) — Aggregation, Storage, and API

**Goal:** Aggregate rubric-aware results and expose enriched API contracts.

### v2 Changes

- `confidence_flag = (ocr_confidence < 0.60) OR (grading_confidence == "low")`.
- Save `prompt_hash` and `llm_model` in result record.
- Add endpoint: `GET /result/{id}/rubric` for per-dimension breakdown.

### Updated Result Structure (example)

```json
{
  "result_id": "uuid",
  "total_marks": 17.5,
  "max_total": 20.0,
  "confidence_flag": false,
  "grading_confidence": "high",
  "prompt_hash": "sha256:...",
  "llm_model": "claude-sonnet-4-20250514",
  "questions": [
    {
      "question_id": "Q1",
      "awarded_marks": 3.5,
      "max_marks": 4.0,
      "rubric_scores": {
        "factual_accuracy": 4,
        "conceptual_completeness": 3,
        "reasoning": 3,
        "domain_terminology": 4
      },
      "feedback": "Accurate and clear, with minor omission.",
      "grading_confidence": "high",
      "ocr_confidence": 0.91,
      "flags": []
    }
  ]
}
```

---

## Phase 5 — Dashboard and Reporting (enhanced)

**Goal:** Present rubric-level explainability in UI and PDF.

### Deliverables

- Annotated sheet with per-question marks.
- Rubric table in result view and PDF report.
- Confidence and flag visualization (OCR + LLM confidence).

### New v2 UI requirements

- Show per-dimension rubric bars/scores.
- Highlight review-required answers.
- Show prompt/model metadata for admin/audit view.

---

## Phase 6 (Updated) — Testing, Metrics, and Ablation

### 6.1 Metrics

| Metric | Target | Notes |
|---|---:|---|
| HTR WER | < 12% | unchanged |
| End-to-end latency | < 15s | relaxed for LLM overhead |
| Overall grading accuracy | > 85% | unchanged |
| Pearson correlation | > 0.82 | unchanged |
| Cohen's kappa (rubric dims) | > 0.70 | new |
| LLM JSON validity rate | > 99% | new |
| Fallback rate | < 5% | new |

### 6.2 Tasks

- Add rubric-level agreement experiments against human grading.
- Track JSON validity and fallback rates in evaluation logs.
- Report latency per stage (preprocess/OCR/retrieval/LLM/report).

### 6.3 Roadblocks & Mitigations

- **Human rubric labels unavailable:** run subset study first.
- **LLM cost concerns:** cache by answer/prompt hash.
- **Long-tail latency spikes:** async workers + queued jobs.

---

## Implementation Priorities

### High Priority (v2.1)

1. Multi-pass OCR consensus before LLM scoring.
2. LLM self-consistency check on low-confidence answers.
3. Structured answer key format with per-question rubric weights.

### Medium Priority (v2.2)

1. Diagram/formula detection path.
2. Examiner override feedback loop (active learning).
3. Multilingual support for OCR + evaluation.
4. Async queue orchestration (Celery + Redis).

### Backlog

1. On-prem LLM option.
2. Real-time grading progress streaming.
3. Near-duplicate answer detection using embeddings.

---

## References

[1] M. Li et al., TrOCR: Transformer-based OCR with Pre-trained Models, arXiv:2109.10282, 2021.  
[2] N. Reimers and I. Gurevych, Sentence-BERT, EMNLP, 2019.  
[3] Google Cloud Vision API — Handwriting Recognition.  
[4] Microsoft Azure AI Vision — Read API.  
[5] PaddleOCR (PaddlePaddle).  
[6] G. Bradski, The OpenCV Library, 2000.  
[7] IAM Handwriting Dataset, IJDAR, 2002.  
[8] A. Devlin et al., BERT, NAACL, 2019.  
[9] HuggingFace Transformers documentation.  
[10] pgvector for PostgreSQL.  
[11] OpenAI Embeddings guide.  
[12] J. Cohen, A Coefficient of Agreement for Nominal Scales, 1960.
