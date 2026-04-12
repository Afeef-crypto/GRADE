-- GRADE core schema (Postgres / Supabase).
-- Application schema for GRADE (PostgreSQL + pgvector); mirrors src/autograder/db.py init_db().
--
-- Embedding width 128 matches autograder.embeddings.EMBEDDING_DIMS (embed_text_local).
-- If you switch to OpenAI-style embeddings (e.g. 1536 dims), add a new migration to
-- ALTER COLUMN embedding TYPE vector(1536) and re-embed all keys.

-- ---------------------------------------------------------------------------
-- Sheets (uploaded answer-sheet files; paths remain app-local or object storage URIs)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.sheets (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    path TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE public.sheets IS 'Scanned answer sheets; path is filesystem or storage URI resolved by the API';

-- ---------------------------------------------------------------------------
-- Answer keys (per exam); embedding for semantic retrieval / top_k
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.answer_keys (
    id TEXT PRIMARY KEY,
    exam_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    expected_answer TEXT NOT NULL,
    embedding vector(128) NOT NULL,
    max_marks DOUBLE PRECISION NOT NULL CHECK (max_marks > 0),
    domain TEXT NOT NULL DEFAULT 'general',
    rubric_override JSONB,
    sort_order INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE public.answer_keys IS 'Model answers; sort_order preserves upload / sheet row order';
COMMENT ON COLUMN public.answer_keys.embedding IS '128-D from embed_text_local; change dimension only with a coordinated migration + re-embed';

CREATE INDEX IF NOT EXISTS idx_answer_keys_exam_sort
    ON public.answer_keys (exam_id, sort_order ASC, question_id ASC);

-- Approximate nearest neighbor (cosine). Tune lists / build after bulk load if you use IVFFlat.
-- HNSW works well from small N upward on current pgvector (Supabase).
CREATE INDEX IF NOT EXISTS idx_answer_keys_embedding_hnsw
    ON public.answer_keys
    USING hnsw (embedding vector_cosine_ops);

-- ---------------------------------------------------------------------------
-- Evaluation results (aggregated run per sheet + exam)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.evaluation_results (
    id TEXT PRIMARY KEY,
    sheet_id TEXT NOT NULL,
    exam_id TEXT NOT NULL,
    per_question_scores JSONB NOT NULL,
    total_marks DOUBLE PRECISION NOT NULL,
    max_total DOUBLE PRECISION NOT NULL,
    confidence_flag BOOLEAN NOT NULL DEFAULT FALSE,
    grading_confidence TEXT,
    ocr_engine_used TEXT,
    prompt_hash TEXT,
    llm_model TEXT,
    flags JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_evaluation_results_sheet
        FOREIGN KEY (sheet_id) REFERENCES public.sheets (id) ON DELETE CASCADE
);

COMMENT ON COLUMN public.evaluation_results.per_question_scores IS 'Array of question result objects (same shape as API questions[])';
COMMENT ON COLUMN public.evaluation_results.flags IS 'JSON array of string flags';

CREATE INDEX IF NOT EXISTS idx_evaluation_results_sheet
    ON public.evaluation_results (sheet_id);

CREATE INDEX IF NOT EXISTS idx_evaluation_results_exam_created
    ON public.evaluation_results (exam_id, created_at DESC);
