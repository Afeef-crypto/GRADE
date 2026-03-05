# AutoGrader — Implementation Plan

**Version:** 1.0  
**Project:** AutoGrader — Automatic Handwritten Answer Sheet Evaluator  
**Purpose:** Phased, citation-backed plan to implement the full system from proposal to deployment.

---

## Overview

This plan breaks implementation into **6 phases** aligned with the system architecture and the 8-stage pipeline. Each phase includes:

- **Deliverables** and **tasks** with notes  
- **Acceptance criteria**  
- **Citations** for design and technology choices  
- **Roadblocks & mitigations** — likely blockers and how to avoid or resolve them  

**Dependencies:** Phases are sequential. Phase 4 depends on 1–3; Phase 5 depends on 4; Phase 6 (testing) runs after 5. Phase 3 (DB + scoring) can start in parallel with Phase 2 (OCR) once the schema and key format are fixed.

---

## Phase 1: Foundation & Image Preprocessing

**Goal:** Set up project structure, dependencies, and implement the image ingestion, segmentation, and patch preprocessing pipeline (Stages 1–3).

### 1.1 Deliverables

- Python project layout (core pipeline, API placeholder).
- OpenCV-based ingest: load image → grayscale → adaptive Gaussian thresholding → deskew (Hough line transform).
- Contour-based segmentation: extract answer regions (template grid or dynamic bounding boxes); output ordered list of cropped patches.
- Per-patch preprocessing: bilateral filter, morphological closing, resize to 384×384.
- Unit tests on sample images (deskew, segment, patch count).

### 1.2 Tasks

| # | Task | Notes |
|---|------|--------|
| 1 | Create repo structure: `src/`, `tests/`, `docs/`, `requirements.txt` | — |
| 2 | Implement `ingest()`: load, grayscale, `cv2.adaptiveThreshold`, deskew | Use OpenCV 4.x |
| 3 | Implement `segment()`: contour detection, filter by area/aspect, sort by position | Support configurable template or dynamic detection |
| 4 | Implement `preprocess_patch()`: bilateral filter, morph close, resize 384×384 | Single patch in, single patch out |
| 5 | Integrate into a single `preprocess_pipeline(image_path) -> List[ndarray]` | Return list of 384×384 patches |
| 6 | Add tests with 2–3 sample answer-sheet images | Verify patch count and dimensions |

### 1.3 Acceptance Criteria

- Given a scanned JPEG/PNG, pipeline returns N patches (N = number of answer regions), each 384×384.
- Deskew corrects obvious rotation; contours isolate answer boxes without major overlap.

### 1.4 Roadblocks & Mitigations

| Roadblock | Mitigation |
|-----------|------------|
| **Answer-sheet layout varies** (different institutions, no standard grid) | Decide early: support one fixed template (e.g. numbered boxes) or a configurable template (JSON/YAML defining regions). For MVP, use a single predefined layout and document it; add config later. |
| **Contour detection fails** (noise, merged boxes, skewed scan) | Add fallback: if contour count ≠ expected N, retry with relaxed params or fall back to a fixed grid split (e.g. N equal rows). Log failures for manual review. |
| **Empty or tiny regions** | Filter contours by min area and aspect ratio; reject or flag patches that are too small. Return patch index so downstream can align with question_id. |
| **Patch–question ordering ambiguity** | Always sort regions by position (e.g. top-to-bottom, left-to-right) and return ordered list; store 1:1 mapping patch_index → question_id in config or key. |

### 1.5 Citations

| Topic | Citation |
|-------|----------|
| OpenCV usage, image ops, Hough transform | **[6]** G. Bradski, "The OpenCV Library," Dr. Dobb's Journal of Software Tools, 2000. https://opencv.org |
| Adaptive thresholding, morphological ops | OpenCV official docs (built on [6]) |
| Contour-based layout analysis | Standard CV practice; segmentation approach consistent with document analysis literature |

---

## Phase 2: OCR / HTR Three-Tier Layer

**Goal:** Implement Stage 4 — primary cloud OCR (Google Vision or Azure Read API), fallback to PaddleOCR, then TrOCR; return text + confidence + engine used.

### 2.1 Deliverables

- Wrapper for Google Cloud Vision API or Azure AI Vision Read API (DOCUMENT_TEXT_DETECTION / handwriting mode).
- PaddleOCR integration (local); call when cloud fails or is rate-limited.
- TrOCR (HuggingFace) integration, optional fine-tuning on IAM Handwriting Dataset; beam-search width=4.
- Orchestrator: try Primary → Fallback 1 → Fallback 2; log confidence and which engine responded.
- Per-patch function: `ocr_patch(patch_image) -> (text, confidence, engine_name)`.

### 2.2 Tasks

| # | Task | Notes |
|---|------|--------|
| 1 | Implement Google Vision or Azure Read API client; DOCUMENT_TEXT_DETECTION | Use official Python SDK; handle auth (env vars) |
| 2 | Add retry/backoff and rate-limit handling; on failure, return control to orchestrator | — |
| 3 | Integrate PaddleOCR; run locally on patch (after resize/normalisation) | [5] |
| 4 | Integrate TrOCR (HuggingFace); load pre-trained or IAM-fine-tuned model; beam_search width=4 | [1], [9] |
| 5 | Implement `ocr_patch()` with try Primary → PaddleOCR → TrOCR | Log which engine succeeded |
| 6 | Return confidence: from API response, PaddleOCR score, or TrOCR character-level aggregation | Raise confidence flag if &lt; 0.60 |

### 2.3 Roadblocks & Mitigations

| Roadblock | Mitigation |
|-----------|------------|
| **Cloud API keys / quota** | Use env vars for keys; document free tiers (Google 1k/mo, Azure 5k/mo). Implement rate limiting and exponential backoff; switch to fallback when quota exceeded or API returns 429. |
| **Confidence scores not comparable** across engines | Normalise to [0, 1]: map each engine’s raw score to a common scale and document the mapping; use normalised value for confidence_flag (e.g. &lt; 0.60). |
| **PaddleOCR / TrOCR slow on CPU** | For &lt; 12 s/sheet target: run OCR on GPU if available; batch patches where possible; or accept longer latency for fallback and document. Consider async per-patch with a timeout. |
| **TrOCR fine-tuning** (IAM dataset, compute) | MVP: use pre-trained TrOCR without fine-tuning. If WER is high, plan a separate sprint for IAM download, fine-tuning script, and model versioning; optional for Phase 6. |
| **Empty or garbage OCR output** | If recognised text is empty or length &lt; threshold, retry with next tier or assign 0 marks and set confidence_flag; never pass empty string to SBERT without handling. |

### 2.4 Acceptance Criteria

- Each patch gets exactly one OCR result (text + confidence + engine).
- If cloud is unavailable, PaddleOCR or TrOCR returns a valid result.
- Confidence values are in [0, 1] and logged for downstream confidence flag.

### 2.5 Citations

| Topic | Citation |
|-------|----------|
| TrOCR architecture and pre-trained models | **[1]** M. Li et al., "TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models," arXiv:2109.10282, 2021. |
| Google Cloud Vision for handwriting | **[3]** Google Cloud, "Cloud Vision API – Handwriting Recognition," https://cloud.google.com/vision |
| Azure Read API for handwritten documents | **[4]** Microsoft Azure, "Azure AI Vision – Read API for Handwritten Documents," https://azure.microsoft.com/en-us/products/ai-services/ai-vision |
| PaddleOCR multi-language OCR | **[5]** PaddlePaddle, "PaddleOCR: Awesome Multilingual OCR Toolkits," https://github.com/PaddlePaddle/PaddleOCR |
| IAM Handwriting Dataset for TrOCR fine-tuning | **[7]** U. Marti and H. Bunke, "The IAM-database: An English Sentence Database for Offline Handwriting Recognition," IJDAR, vol. 5, pp. 39–46, 2002. |
| HuggingFace Transformers (TrOCR implementation) | **[9]** HuggingFace, "Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX," 2023. https://huggingface.co/transformers |

---

## Phase 3: Answer Key Storage & Semantic Scoring

**Goal:** Implement answer key storage (Stage 5) and semantic similarity scoring (Stage 6) using Sentence-BERT.

### 3.1 Deliverables

- PostgreSQL schema: `answer_keys` (question_id, expected_answer, max_marks), `evaluation_results`, `sheets` (if needed).
- API or internal API to load key: question_id → expected_answer, max_marks.
- Sentence-BERT encoding: `all-mpnet-base-v2`; 768-dim embeddings for student and model answers.
- Cosine similarity and mapping to marks: `awarded_marks = round(cosine_sim × max_marks, 1)`; configurable zero-mark threshold (default 0.25).
- Function: `score_answer(recognised_text, model_answer, max_marks) -> (awarded_marks, similarity, justification)`.

### 3.2 Tasks

| # | Task | Notes |
|---|------|--------|
| 1 | Define and migrate PostgreSQL schema (keys, results, optional sheets) | Use Alembic or raw SQL |
| 2 | Implement key upload/parsing (e.g. JSON); store in `answer_keys` | — |
| 3 | Integrate `sentence-transformers`; load `all-mpnet-base-v2` | [2] |
| 4 | Implement encode(student_text), encode(model_answer); compute cosine_sim | [2], [8] |
| 5 | Implement mark mapping with threshold (e.g. sim &lt; 0.25 → 0 marks) | Config in config file or env |
| 6 | Add optional justification string (e.g. "Partial credit: similarity 0.72") for report | — |

### 3.3 Roadblocks & Mitigations

| Roadblock | Mitigation |
|-----------|------------|
| **PostgreSQL not available** (local dev, CI) | Use Docker Compose or a cloud instance; document one-line setup. For dev-only, consider SQLite as an optional fallback and keep schema compatible (e.g. same column names). |
| **Sentence-BERT first-load slow / large** | Model download on first run; cache in standard HuggingFace cache. Consider lazy loading at first scoring request and show “Loading model…” in API if needed. |
| **Empty or very long text** | If student or model answer is empty: award 0 and log. If length &gt; model max (e.g. 512 tokens): truncate or split and aggregate (e.g. max similarity); document behaviour. |
| **Key–patch count mismatch** | Validate key size vs. number of patches before scoring; return clear error (e.g. “Key has 5 questions, sheet produced 4 patches”). Require explicit question_id in key to allow optional questions. |

### 3.4 Acceptance Criteria

- Key is stored and retrievable by question_id.
- For identical (or near-identical) text, similarity ≈ 1.0 and full marks awarded.
- Paraphrased correct answers receive partial credit; irrelevant text receives low similarity and low/zero marks.

### 3.5 Citations

| Topic | Citation |
|-------|----------|
| Sentence-BERT and sentence embeddings | **[2]** N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," EMNLP, 2019. |
| BERT foundation (used by Sentence-BERT) | **[8]** A. Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," NAACL, 2019. |

---

## Phase 4: Aggregation, Result Storage & Backend API

**Goal:** Implement Stage 7 (aggregate scores, confidence flag, persist result) and expose a REST API for the frontend (FastAPI).

### 4.1 Deliverables

- Aggregate per-question scores into total marks; set confidence_flag if any OCR confidence &lt; 0.60.
- Persist result JSON to PostgreSQL (`evaluation_results`).
- FastAPI app: `POST /upload/sheet`, `POST /upload/key`, `POST /evaluate`, `GET /result/{id}`, `GET /report/{id}/pdf` (stub or redirect to Phase 5).
- Error handling, request validation, and basic logging.

### 4.2 Tasks

| # | Task | Notes |
|---|------|--------|
| 1 | Implement aggregate(scores_list, ocr_confidences) -> total_marks, confidence_flag | — |
| 2 | Write result record to DB (per-question scores, total, flag, ocr_engine_used) | — |
| 3 | Create FastAPI app; define Pydantic models for uploads and responses | — |
| 4 | Implement upload endpoints (multipart for sheet; JSON for key) | — |
| 5 | Implement evaluate: run full pipeline (preprocess → OCR → load key → score → aggregate → save) | Reuse Phase 1–3 code |
| 6 | Implement GET /result/{id} and placeholder GET /report/{id}/pdf | PDF in Phase 5 |

### 4.3 Roadblocks & Mitigations

| Roadblock | Mitigation |
|-----------|------------|
| **Evaluate blocks API** (long-running pipeline) | Run evaluation in a background task (e.g. FastAPI BackgroundTasks, Celery, or async worker). Return job_id immediately; provide GET /result/{id} that returns 202 + “processing” until done, then 200 + body. |
| **Key must exist before evaluate** | Enforce ordering: require key_id or key payload in evaluate request; return 400 if key missing. Document flow: upload key → upload sheet → evaluate(sheet_id, key_id). |
| **Duplicate submissions** | Use idempotency key (e.g. hash of sheet + key_id) or allow duplicate and always create new result; document policy. Optionally add GET /results?sheet_id= for listing. |
| **Large payload / timeouts** | Limit upload size (e.g. 10 MB); stream or save to temp file and process asynchronously; set reasonable timeouts for external OCR calls. |

### 4.4 Acceptance Criteria

- One evaluation produces one result row with correct total and confidence flag.
- API accepts sheet + key and returns result ID; GET /result/{id} returns full result JSON.

### 4.5 Citations

| Topic | Citation |
|-------|----------|
| REST API design, FastAPI | FastAPI docs; general REST best practices (no specific proposal ref; implementation standard). |
| Data persistence | PostgreSQL documentation; schema designed per ARCHITECTURE.md. |

---

## Phase 5: Report Generation & Dashboard (Stage 8)

**Goal:** Implement PDF report generation and React dashboard with annotated sheet and downloadable report.

### 5.1 Deliverables

- PDF report (e.g. ReportLab): cover info, per-question table (question_id, recognised text, similarity, awarded_marks, max_marks), total, confidence warning if flag set.
- FastAPI endpoint `GET /report/{id}/pdf` returns PDF bytes (or file response).
- React app: upload sheet and key (or select existing); trigger evaluate; view result (annotated image with score badges, total summary); download PDF.

### 5.2 Tasks

| # | Task | Notes |
|---|------|--------|
| 1 | Implement PDF generation: result JSON + optional metadata → PDF with table and totals | ReportLab or WeasyPrint |
| 2 | Serve PDF via FastAPI; set Content-Disposition for download | — |
| 3 | React: upload UI, call POST /upload/sheet, POST /upload/key, POST /evaluate | — |
| 4 | React: fetch GET /result/{id}; overlay score badges on sheet image at region coordinates | Store bbox per question if needed |
| 5 | React: total marks summary and "Download report" button → GET /report/{id}/pdf | — |
| 6 | Optional: store original image reference for overlay; pass bboxes from pipeline | — |

### 5.3 Roadblocks & Mitigations

| Roadblock | Mitigation |
|-----------|------------|
| **Bounding boxes for overlay missing** | Pipeline must return bbox per question (e.g. from Phase 1 segment step). Store in result JSON (e.g. `regions: [{ question_id, bbox: [x,y,w,h] }]`) so frontend can overlay badges. Add this to Phase 1/4 if not already present. |
| **CORS / separate frontend origin** | Enable CORS in FastAPI for frontend origin; use env var for allowed origins. For local dev, allow localhost. |
| **Storing full sheet image** | Decide: store file path or blob in DB vs. object storage. For MVP, store path or base64 in result for report/dashboard; add cleanup policy for temp files. |
| **PDF generation failures** | Validate result JSON before generating; catch ReportLab errors and return 500 with clear message; optional: queue PDF generation and serve when ready. |

### 5.4 Acceptance Criteria

- PDF contains all per-question data and total; confidence warning when flag is set.
- Dashboard shows annotated sheet with scores and allows PDF download.

### 5.5 Citations

| Topic | Citation |
|-------|----------|
| Report generation, layout | ReportLab / WeasyPrint documentation; proposal Section 4 (Result Generation Dashboard). |
| Frontend integration | React and REST integration best practices; proposal Section 4. |

---

## Phase 6: Testing, Metrics & Ablation

**Goal:** Meet proposal’s testing strategy and target metrics; run ablation study.

### 6.1 Deliverables

- Dataset: 200 handwritten scripts, 3 domains (CS, Mathematics, General Science), 800 QA pairs with ground-truth marks.
- Ablation: Baseline (Tesseract + keyword match) vs. Cloud + SBERT vs. PaddleOCR + SBERT vs. TrOCR + SBERT.
- Metrics: HTR Word Error Rate &lt; 12%; Grading Accuracy &gt; 85%; Pearson correlation with human evaluator &gt; 0.82; end-to-end latency per sheet &lt; 12 s.
- Report or spreadsheet with results and short conclusions.

### 6.2 Tasks

| # | Task | Notes |
|---|------|--------|
| 1 | Collect or curate 200 scripts + ground-truth marks (800 QA pairs) | Per proposal |
| 2 | Implement WER computation (optional: use library); measure for each OCR engine | — |
| 3 | Run grading with each configuration; compute grading accuracy vs. ground truth | — |
| 4 | Compute Pearson correlation between system marks and human marks (subset) | — |
| 5 | Measure end-to-end latency (upload to result); optimize if &gt; 12 s | — |
| 6 | Document ablation and metrics in `docs/TESTING_RESULTS.md` or similar | — |

### 6.3 Roadblocks & Mitigations

| Roadblock | Mitigation |
|-----------|------------|
| **200-script dataset not available** | Start with a smaller set (e.g. 20–50) for initial ablation; use synthetic or publicly available handwriting (e.g. IAM subset) if needed. Document “target 200” and report on actual N. |
| **Human grader correlation** (Pearson &gt; 0.82) | Requires double-marking by humans on a subset. Plan: get 1–2 human graders to mark a sample (e.g. 30–50 scripts); compute Pearson between system and human. If human data unavailable, report “N/A” and use grading accuracy vs. ground truth only. |
| **WER needs ground-truth transcript** | WER is per-OCR engine vs. manually transcribed answers. Either transcribe a subset (e.g. 50 patches) or use existing IAM test set for OCR-only WER; report “grading accuracy” as primary metric if full transcripts are scarce. |
| **Latency &gt; 12 s** | Profile: measure time per stage (preprocess, OCR, scoring). Optimise heaviest (often OCR): GPU, batching, or lower-resolution TrOCR. If still over, document and set “target for v2.” |

### 6.4 Acceptance Criteria

- All four ablation configurations run and are compared.
- Target metrics documented; either targets met or gaps explained with next steps.

### 6.5 Citations

| Topic | Citation |
|-------|----------|
| Testing strategy and metrics | Proposal Section 5 (Testing Strategy); target metrics and ablation design from proposal. |
| WER and evaluation of OCR | Standard NLP/OCR evaluation; IAM benchmark [7] for HTR. |

---

## Common Roadblocks Across Phases

| Risk | Where it appears | Mitigation summary |
|------|-------------------|---------------------|
| **Layout / format assumptions** | Phase 1, 5 | Fix one answer-sheet format for MVP; document it; add bbox output for overlay. |
| **External services (cloud OCR, quota)** | Phase 2 | Env-based config; retry/backoff; fallback chain; normalise confidence. |
| **Long-running pipeline blocking API** | Phase 4 | Background jobs; return job_id; poll or webhook for result. |
| **Data dependencies** (key before evaluate, patches ↔ questions) | Phase 3, 4 | Validate key vs. patch count; enforce upload order in API contract. |
| **Dataset / human labels for testing** | Phase 6 | Start small; use IAM or synthetic data; document “target 200” and human correlation as optional if unavailable. |

---

## Summary Table: Phases, Roadblocks, and Citations

| Phase | Focus | Key roadblocks | Primary citations |
|-------|--------|----------------|-------------------|
| 1 | Image preprocessing (OpenCV) | Layout variance; contour failures; patch–question ordering | [6] OpenCV |
| 2 | OCR/HTR three-tier | API quota; confidence normalisation; CPU latency; TrOCR fine-tuning | [1], [3], [4], [5], [7], [9] |
| 3 | Answer key & semantic scoring | DB setup; SBERT load time; empty/long text; key–patch mismatch | [2], [8] |
| 4 | Aggregation, DB, FastAPI | Blocking evaluate; key-before-evaluate; idempotency; timeouts | Architecture; PostgreSQL; FastAPI |
| 5 | PDF & dashboard | Missing bboxes; CORS; image storage; PDF errors | Proposal §4; ReportLab/React |
| 6 | Testing & ablation | Dataset size; human correlation; WER ground truth; latency | Proposal §5; [7] |

---

## Reference List (Proposal)

[1] M. Li et al., "TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models," arXiv:2109.10282, 2021.  
[2] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," EMNLP, 2019.  
[3] Google Cloud, "Cloud Vision API – Handwriting Recognition," https://cloud.google.com/vision  
[4] Microsoft Azure, "Azure AI Vision – Read API for Handwritten Documents," https://azure.microsoft.com/en-us/products/ai-services/ai-vision  
[5] PaddlePaddle, "PaddleOCR: Awesome Multilingual OCR Toolkits," https://github.com/PaddlePaddle/PaddleOCR  
[6] G. Bradski, "The OpenCV Library," Dr. Dobb's Journal of Software Tools, 2000. https://opencv.org  
[7] U. Marti and H. Bunke, "The IAM-database: An English Sentence Database for Offline Handwriting Recognition," IJDAR, vol. 5, pp. 39–46, 2002.  
[8] A. Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," NAACL, 2019.  
[9] HuggingFace, "Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX," 2023. https://huggingface.co/transformers
