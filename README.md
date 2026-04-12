# GRADE (Generalized Recognition and Automated Document Evaluator)

**Automatic Handwritten Answer Sheet Evaluator** — scans, recognises, and grades handwritten exam answers with minimal human intervention.

---

## About

GRADE segments scanned sheets (OpenCV), runs **OCR** with a tiered stack (**Google Cloud Vision** `document_text_detection` preferred when configured, then Azure, PaddleOCR, TrOCR), and **scores** answers with an **LLM rubric** when enabled or a **legacy fallback** otherwise. Results are exposed via **FastAPI** and a PDF report.

- **Objective:** Reliable grading for open-ended handwritten responses.
- **Implementation plan:** See [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md).

---

## Tech stack (as implemented)

| Layer | Technology |
|-------|------------|
| **Preprocessing** | OpenCV (segmentation, patches) |
| **OCR / HTR** | **Google Cloud Vision** (tier 1 when `GOOGLE_APPLICATION_CREDENTIALS` or `GOOGLE_CLOUD_VISION_API_KEY` is set), Azure, PaddleOCR, TrOCR |
| **Embeddings / retrieval** | Local hash embeddings (`embed_text_local`, 128-D); optional **pgvector** on Postgres |
| **Scoring** | Gemini when `GRADE_ENABLE_LLM` + key; else legacy fallback |
| **Backend** | FastAPI |
| **Storage** | **PostgreSQL + pgvector** (`GRADE_DATABASE_URL` required) |
| **Report** | ReportLab PDF |

Install extras as needed: `pip install -e ".[api]"` (includes Postgres drivers), `pip install -e ".[ocr_google]"`, `pip install -e ".[llm_gemini]"`.

---

## Repository structure (high level)

```
GRADE/
├── docs/                    # ARCHITECTURE.md, IMPLEMENTATION_PLAN.md
├── supabase/migrations/     # Postgres + pgvector schema (Supabase-compatible)
├── scripts/
│   ├── verify_phases.py     # Import smoke test
│   ├── verify_postgres_db.py   # DB + pgvector integration (needs GRADE_DATABASE_URL)
│   └── verify_key_pdfs.py   # Answer-key PDF parsing (docs/Key.pdf, DocScanner*.pdf)
├── src/autograder/          # preprocessing, ocr, db, api, scoring, …
├── tests/
├── pyproject.toml
└── README.md
```

---

## Configuration (`.env`)

Copy [`.env.example`](.env.example) to `.env` in the repo root.

| Variable | Purpose |
|----------|---------|
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to GCP service account JSON (**recommended for Vision OCR**) |
| `GOOGLE_CLOUD_VISION_API_KEY` | Alternative to JSON for Vision API |
| `GRADE_OCR_GOOGLE_ONLY` | Optional: cloud tier uses only Google (no Azure fallback) |
| `GRADE_OCR_CLOUD_ONLY` | Optional: no local PaddleOCR/TrOCR fallback |
| `GRADE_DATABASE_URL` | **Required** for the API: Postgres URI (see `docker-compose.yml` for local dev) |
| `GRADE_ENABLE_LLM` / `GRADE_GEMINI_API_KEY` | Optional LLM grading |

Passwords with **`$`** in URLs: loaders use `interpolate=False` so `.env` values are not mangled.

---

## Getting started

1. **Clone** and **install:**
   ```bash
   pip install -e ".[api,ocr_google]"
   ```
2. **Database:** use **Docker** (below) or any PostgreSQL 16+ with pgvector. Set `GRADE_DATABASE_URL` in `.env`. The API runs `init_db()` on startup and creates the schema (no manual SQL required for a fresh local DB). For **Supabase**, apply `supabase/migrations/` via the SQL editor or CLI instead.
3. **Configure** `.env` (Vision credentials for OCR; `GRADE_DATABASE_URL`; optional Gemini).
4. **Run API** (from repo root):
   ```powershell
   $env:PYTHONPATH = "src"
   python -m uvicorn autograder.api:app --host 127.0.0.1 --port 8000
   ```
5. **Tests:** `python -m pytest` needs Postgres (default matches Docker below; override with `GRADE_TEST_DATABASE_URL`).

### Docker (local PostgreSQL + pgvector)

1. Install **[Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install/)** and enable the **WSL 2** backend if the installer asks. Reboot if required.
2. **Start Docker Desktop** and wait until the whale icon shows **Running** (if `docker version` errors with `dockerDesktopLinuxEngine` / “pipe … not found”, the engine is not started).
3. From the repo root:
   ```powershell
   docker compose up -d
   docker compose ps
   ```
   You should see `postgres` **healthy** on host port **5433** (avoids clashing with a local Postgres on 5432).
4. Point the app at the container (for local-only dev, in `.env`):
   ```text
   GRADE_DATABASE_URL=postgresql://grade:grade@127.0.0.1:5433/grade_test
   ```
   To keep **Supabase** for the API but run **tests** against Docker, leave `GRADE_DATABASE_URL` as-is and set `GRADE_TEST_DATABASE_URL` to the URL above (see `tests/conftest.py`).
5. Smoke test:
   ```powershell
   $env:PYTHONPATH = "src"
   $env:GRADE_DATABASE_URL = "postgresql://grade:grade@127.0.0.1:5433/grade_test"
   python scripts/verify_postgres_db.py
   ```

### Verification scripts

```powershell
$env:PYTHONPATH = "src"
python scripts/verify_phases.py
python scripts/verify_key_pdfs.py          # parses docs/Key.pdf + DocScanner*.pdf if present
python scripts/verify_postgres_db.py     # needs GRADE_DATABASE_URL (pip install -e ".[api]" includes drivers)
```

- **`docs/Key.pdf`**: text-based key → expect multiple `Q1…` sections after parsing.
- **`DocScanner … am.pdf`**: typical **phone scan** has no text layer → script reports “no extractable text” until you use a text PDF or OCR the scan elsewhere.

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design (note: diagram may describe earlier SBERT/Postgres-only assumptions). |
| [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) | Phased plan and v2 direction. |

---

## Team

- Md Afeeduddin  
- Syed Liyaqat  

*B.E. Computer Science Engineering — Academic Mini Project (GRADE v1)*

---

## License

Academic project — see institution guidelines.
