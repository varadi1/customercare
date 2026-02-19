-- Obsidian RAG schema for PostgreSQL + pgvector
-- Target DB: neu-docs-db (port 5433, user klara, db neu_docs)

CREATE EXTENSION IF NOT EXISTS vector;

-- Main chunks table
CREATE TABLE IF NOT EXISTS obsidian_chunks (
    id SERIAL PRIMARY KEY,
    chunk_id VARCHAR(64) NOT NULL UNIQUE,
    file_path VARCHAR(1024) NOT NULL,
    file_name VARCHAR(512) NOT NULL,
    folder VARCHAR(64) NOT NULL,   -- inbox, projects, areas, resources, archive, tags, tasknotes, other
    chunk_index INTEGER NOT NULL DEFAULT 0,
    content TEXT NOT NULL,
    embedding vector(1024),
    tsv tsvector GENERATED ALWAYS AS (
        to_tsvector('simple', content)
    ) STORED,
    file_hash VARCHAR(64),
    modified_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    context_prefix TEXT,           -- LLM-generated contextual summary
    original_content TEXT          -- original content before enrichment
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_obsidian_chunks_embedding_hnsw ON obsidian_chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200);
CREATE INDEX IF NOT EXISTS idx_obsidian_chunks_tsv ON obsidian_chunks USING GIN (tsv);
CREATE INDEX IF NOT EXISTS idx_obsidian_chunks_file_path ON obsidian_chunks (file_path);
CREATE INDEX IF NOT EXISTS idx_obsidian_chunks_folder ON obsidian_chunks (folder);
CREATE INDEX IF NOT EXISTS idx_obsidian_chunks_file_hash ON obsidian_chunks (file_hash);

-- Sync state table (replaces JSON hash file)
CREATE TABLE IF NOT EXISTS obsidian_sync_state (
    file_path VARCHAR(1024) PRIMARY KEY,
    file_hash VARCHAR(64) NOT NULL,
    chunk_count INTEGER DEFAULT 0,
    last_synced TIMESTAMPTZ DEFAULT NOW()
);

-- Sync log table
CREATE TABLE IF NOT EXISTS obsidian_sync_log (
    id SERIAL PRIMARY KEY,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    total_files INTEGER DEFAULT 0,
    processed_files INTEGER DEFAULT 0,
    skipped_files INTEGER DEFAULT 0,
    total_chunks INTEGER DEFAULT 0,
    new_files TEXT[] DEFAULT '{}',
    changed_files TEXT[] DEFAULT '{}',
    errors TEXT[] DEFAULT '{}'
);
