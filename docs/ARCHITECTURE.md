# AutoGrader — System Architecture

**Version:** 1.0  
**Project:** AutoGrader — Automatic Handwritten Answer Sheet Evaluator  
**Stack:** OpenCV · Google Vision / Azure AI Vision · PaddleOCR · TrOCR · Sentence-BERT · FastAPI · React · PostgreSQL

---

## 1. High-Level Architecture Overview

AutoGrader is a four-module pipeline that ingests scanned handwritten answer sheets, segments answer regions, performs Handwritten Text Recognition (HTR), scores answers via semantic similarity, and delivers structured feedback via a web dashboard.

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                    AutoGrader System                                     │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐    ┌──────────────────┐  │
│  │   Frontend  │───▶│  FastAPI    │───▶│  Image Preprocess   │───▶│  OCR/HTR Layer   │  │
│  │  (React)    │    │  Backend    │    │  (OpenCV)           │    │  (3-tier)        │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘    └────────┬─────────┘  │
│        ▲                    │                    ▲                         │            │
│        │                    │                    │                         ▼            │
│        │                    │                    │              ┌──────────────────┐   │
│        │                    │                    │              │ Semantic Scoring  │   │
│        │                    │                    │              │ (Sentence-BERT)  │   │
│        │                    │                    │              └────────┬─────────┘   │
│        │                    │                    │                         │            │
│        │                    │                    ▼                         ▼            │
│        │                    │           ┌─────────────────┐      ┌──────────────────┐   │
│        │                    └──────────▶│   PostgreSQL    │◀─────│  Result & Report  │   │
│        │                                 │  (Keys, Results)│      │  Generation       │   │
│        │                                 └─────────────────┘      └──────────────────┘   │
│        └────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Definitions

### 2.1 Image Preprocessing Module (OpenCV)

**Responsibility:** Normalise and segment the scanned answer sheet so each question–answer region is a clean, isolated image patch suitable for OCR.

| Sub-component        | Function |
|----------------------|----------|
| **Ingest**           | Load JPEG/PNG via OpenCV; convert to grayscale; apply adaptive Gaussian thresholding; deskew using Hough line transform [6]. |
| **Segment**          | Contour-based detection to isolate answer boxes (template grid or dynamic bounding boxes); crop and persist each region as a patch. |
| **Patch preprocess** | Per-patch: bilateral filtering (noise removal); morphological closing (connect broken strokes); resize to 384×384 for OCR input normalisation. |

**Inputs:** Single scanned answer sheet image (JPEG/PNG).  
**Outputs:** Ordered list of image patches (one per question region).  
**Citation:** [6] Bradski, "The OpenCV Library," Dr. Dobb's Journal of Software Tools, 2000.

---

### 2.2 OCR / HTR Layer (Three-Tier)

**Responsibility:** Convert each answer-region image into text with confidence scores. Tier order: Cloud API → PaddleOCR → TrOCR.

| Tier   | Engine                    | When used | Output |
|--------|---------------------------|-----------|--------|
| **Primary** | Google Cloud Vision API or Azure AI Vision (Read API) | Default; DOCUMENT_TEXT_DETECTION / dense handwriting mode | Text + confidence |
| **Fallback 1** | PaddleOCR (PaddlePaddle) | Cloud unavailable or rate-limited | Text + confidence |
| **Fallback 2** | TrOCR (HuggingFace), fine-tuned on IAM Handwriting Dataset | When PaddleOCR fails or is unavailable | Text + character-level confidence |

**Interfaces:**
- **Primary:** REST API (Google/Azure Python SDK); 1,000 (Google) / 5,000 (Azure) free transactions/month [3], [4].
- **PaddleOCR:** Local inference; multi-language handwriting support [5].
- **TrOCR:** Vision encoder + language decoder transformer; beam-search (width=4) for decoding [1], [7], [9].

**Inputs:** 384×384 preprocessed patch (per question).  
**Outputs:** Recognised text string; confidence score; engine used (for logging/confidence flags).  
**Citations:** [1] Li et al., "TrOCR: Transformer-based OCR," arXiv:2109.10282, 2021; [3] Google Cloud Vision; [4] Azure AI Vision; [5] PaddleOCR; [7] IAM-database, IJDAR 2002; [9] HuggingFace Transformers.

---

### 2.3 Semantic Similarity Scoring Engine (Sentence-BERT)

**Responsibility:** Compare recognised student answer to model answer and assign marks using semantic similarity with configurable partial credit.

| Step | Action |
|------|--------|
| **Encode** | Encode student answer and model answer to 768-dimensional vectors using Sentence-BERT (`all-mpnet-base-v2`) [2], [8]. |
| **Similarity** | `cosine_sim = dot(v1, v2) / (||v1|| × ||v2||)`. |
| **Mapping** | `awarded_marks = round(cosine_sim × max_marks, 1)`; configurable zero-mark cutoff (default similarity threshold 0.25). |

**Inputs:** Recognised text (from OCR); model answer and `max_marks` per question (from answer key).  
**Outputs:** Per-question awarded marks; similarity score; optional justification for feedback.  
**Citations:** [2] Reimers & Gurevych, "Sentence-BERT," EMNLP 2019; [8] Devlin et al., "BERT," NAACL 2019.

---

### 2.4 Result Generation & Dashboard

**Responsibility:** Persist results, generate feedback report, and present an interactive dashboard with score overlays and downloadable PDF.

| Sub-component   | Function |
|-----------------|----------|
| **Storage**     | Save answer key (question_id → expected_answer, max_marks) and evaluation result JSON (per-question scores, OCR confidence, total) in PostgreSQL. |
| **API**         | FastAPI endpoints: upload sheet, upload key, run pipeline, get result, get report. |
| **Report**      | ReportLab (or equivalent): PDF with per-question breakdown table, similarity scores, OCR confidence indicator. |
| **Dashboard**   | React frontend: annotated sheet image with score badges per region; total marks summary; download PDF. |

**Inputs:** Result JSON; original sheet image; answer key metadata.  
**Outputs:** Stored records; API response; PDF file; dashboard UI.  
**Citations:** (Implementation-specific; no single reference in proposal — best practices for REST APIs and report generation.)

---

## 3. Data Flow (8-Stage Pipeline)

| Stage | Name        | Input | Output | Component |
|-------|-------------|--------|--------|-----------|
| 1 | **INGEST**  | JPEG/PNG file | Grayscale, thresholded, deskewed image | Preprocessing (OpenCV) |
| 2 | **SEGMENT** | Preprocessed full image | List of cropped patches (one per question) | Preprocessing (OpenCV) |
| 3 | **PREPROCESS** | Each patch | 384×384 normalised patch | Preprocessing (OpenCV) |
| 4 | **OCR**     | Each patch | Text + confidence + engine used | OCR/HTR (3-tier) |
| 5 | **LOAD KEY**| Examiner upload | question_id → expected_answer, max_marks in DB | Backend + PostgreSQL |
| 6 | **SCORE**   | Recognised text + model answer + max_marks | awarded_marks, similarity, justification | Semantic engine (SBERT) |
| 7 | **AGGREGATE**| Per-question scores | Total marks; confidence flag if any OCR &lt; 0.60; result JSON in DB | Backend + PostgreSQL |
| 8 | **REPORT**  | Result JSON + original image | Annotated image + PDF report + API response | Backend + React + ReportLab |

---

## 4. Interfaces & Contracts

### 4.1 Backend API (FastAPI)

- `POST /upload/sheet` — Upload answer sheet image; return job/session ID.
- `POST /upload/key` — Upload model answer key (JSON); store in PostgreSQL.
- `POST /evaluate` — Run pipeline for given sheet + key; return result ID.
- `GET /result/{id}` — Return result JSON (scores, confidence, metadata).
- `GET /report/{id}/pdf` — Return generated PDF.

### 4.2 Database Schema (PostgreSQL)

- **answer_keys:** id, exam/session_id, question_id, expected_answer, max_marks, created_at.
- **evaluation_results:** id, sheet_id, key_id, per_question_scores (JSONB), total_marks, confidence_flag, ocr_engine_used, created_at.
- **sheets:** id, file_path_or_blob_ref, upload_time (optional, for traceability).

### 4.3 Frontend–Backend

- REST/JSON over HTTPS; dashboard consumes `/result/{id}` and `/report/{id}/pdf`; uploads via multipart/form-data.

---

## 5. Non-Functional Considerations

- **Latency:** End-to-end target &lt; 12 s per sheet (proposal).
- **Resilience:** OCR tier fallbacks (Cloud → PaddleOCR → TrOCR) for availability and rate limits.
- **Confidence:** Flag result when any OCR confidence &lt; 0.60 for examiner review.
- **Security:** Validate uploads (type, size); avoid storing raw images long-term if not required; secure API and DB access.

---

## 6. References (Proposal)

[1] M. Li et al., "TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models," arXiv:2109.10282, 2021.  
[2] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," EMNLP, 2019.  
[3] Google Cloud, "Cloud Vision API – Handwriting Recognition," https://cloud.google.com/vision  
[4] Microsoft Azure, "Azure AI Vision – Read API for Handwritten Documents," https://azure.microsoft.com/en-us/products/ai-services/ai-vision  
[5] PaddlePaddle, "PaddleOCR: Awesome Multilingual OCR Toolkits," https://github.com/PaddlePaddle/PaddleOCR  
[6] G. Bradski, "The OpenCV Library," Dr. Dobb's Journal of Software Tools, 2000. https://opencv.org  
[7] U. Marti and H. Bunke, "The IAM-database: An English Sentence Database for Offline Handwriting Recognition," IJDAR, vol. 5, pp. 39–46, 2002.  
[8] A. Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," NAACL, 2019.  
[9] HuggingFace, "Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX," 2023. https://huggingface.co/transformers
