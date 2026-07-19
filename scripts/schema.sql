-- KB chunk store for the kb-rag-mcp server. Run once against the target database
-- (Neon: paste into the SQL editor, or: psql "$DATABASE_URL" -f scripts/schema.sql)

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS kb_chunks (
    id              bigserial PRIMARY KEY,
    kb_number       text NOT NULL,
    kb_sys_id       text NOT NULL,
    title           text,
    chunk_index     int NOT NULL,
    chunk_text      text NOT NULL,
    embedding       vector(1536),
    category        text,
    workflow_state  text DEFAULT 'published',
    sys_updated_on  timestamptz,
    created_at      timestamptz NOT NULL DEFAULT now(),
    tsv             tsvector GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED,
    UNIQUE (kb_sys_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS kb_chunks_tsv_idx ON kb_chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS kb_chunks_kb_number_idx ON kb_chunks (kb_number);

-- No HNSW vector index yet: sequential scan is fast below ~50k rows.
-- When the table grows past that:
--   CREATE INDEX kb_chunks_embedding_hnsw_idx ON kb_chunks USING hnsw (embedding vector_cosine_ops);
