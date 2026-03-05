# GRADE (AutoGrader)

**Automatic Handwritten Answer Sheet Evaluator** — uses Computer Vision and Natural Language Processing to scan, recognise, and grade handwritten exam answer sheets with minimal human intervention.

---

## About

AutoGrader processes scanned answer sheet images, segments per-question regions, runs Handwritten Text Recognition (HTR) via a three-tier OCR stack (Cloud → PaddleOCR → TrOCR), and scores answers using **Sentence-BERT** semantic similarity for partial credit and paraphrasing tolerance. Results are shown on an interactive dashboard with score overlays and a downloadable PDF report.

- **Objective:** Reliable automated grading for open-ended handwritten responses (not just MCQ/OMR).
- **Target metrics:** HTR WER &lt; 12%, grading accuracy &gt; 85%, &lt; 12 s per sheet.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Preprocessing** | OpenCV 4.x (thresholding, deskew, contour segmentation) |
| **OCR / HTR** | Google Cloud Vision / Azure AI Vision (primary), PaddleOCR (fallback 1), TrOCR (fallback 2) |
| **Scoring** | Sentence-BERT (`sentence-transformers`, all-mpnet-base-v2) |
| **Backend** | FastAPI, PostgreSQL |
| **Frontend** | React (dashboard, uploads, annotated results, PDF download) |
| **Report** | ReportLab (or equivalent) for PDF generation |

---

## Repository Structure

```
GRADE/
├── docs/
│   ├── ARCHITECTURE.md        # System design, components, data flow
│   └── IMPLEMENTATION_PLAN.md # Phased plan with roadblocks & citations
├── src/
│   └── autograder/
│       ├── __init__.py        # Package exports
│       ├── preprocessing.py   # Ingest, segment, preprocess_patch, preprocess_pipeline
│       └── api.py             # FastAPI placeholder (Phase 4)
├── tests/
│   ├── conftest.py            # Fixtures (synthetic sheets)
│   └── test_preprocessing.py  # Phase 1 unit tests
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Documentation

| Document | Description |
|----------|-------------|
| **[System Architecture](docs/ARCHITECTURE.md)** | Component definitions, 8-stage pipeline, interfaces, DB schema, references. |
| **[Implementation Plan](docs/IMPLEMENTATION_PLAN.md)** | 6-phase implementation with tasks, acceptance criteria, **roadblocks & mitigations**, and citations. |

---

## Team
 
- Md Afeeduddin  
- Syed Liyaqat  

*B.E. Computer Science Engineering — Academic Mini Project (AutoGrader v1)*

---

## Getting Started

1. **Clone:** `git clone https://github.com/Afeef-crypto/GRADE.git`
2. **Install:** `pip install -r requirements.txt` (or `pip install -e ".[dev]"` from repo root for editable install + tests).
3. **Run preprocessing (Phase 1):**
   ```bash
   # From repo root, with PYTHONPATH=src
   python -c "
   from autograder import preprocess_pipeline
   result = preprocess_pipeline('path/to/answer_sheet.png', expected_num_regions=5)
   print(len(result.patches), 'patches', result.patches[0].shape)
   "
   ```
4. **Run tests:** `pytest` (with `src` on PYTHONPATH, or after `pip install -e .`).
5. Review [ARCHITECTURE.md](docs/ARCHITECTURE.md) and [IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for full pipeline and phases.

---

## License

Academic project — see institution guidelines.
