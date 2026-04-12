-- GRADE: enable pgvector for answer-key embeddings.
-- Supabase: you can also toggle "vector" under Database → Extensions; this migration keeps the repo reproducible.
-- Requires: PostgreSQL with pgvector (Supabase includes it).

CREATE EXTENSION IF NOT EXISTS vector;

COMMENT ON EXTENSION vector IS 'pgvector — embedding storage and similarity search for GRADE answer_keys';
