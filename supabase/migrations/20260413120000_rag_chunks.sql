-- RAG / OCR chunk storage with embed_text_local(128) vectors (same dim as answer_keys).

CREATE TABLE IF NOT EXISTS public.rag_chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text_content TEXT NOT NULL,
    embedding vector(128) NOT NULL,
    meta JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE public.rag_chunks IS 'OCR/RAG text chunks with 128-D hash embeddings for retrieval';

CREATE INDEX IF NOT EXISTS idx_rag_chunks_document_id
    ON public.rag_chunks (document_id);

CREATE INDEX IF NOT EXISTS idx_rag_chunks_embedding_hnsw
    ON public.rag_chunks
    USING hnsw (embedding vector_cosine_ops);
