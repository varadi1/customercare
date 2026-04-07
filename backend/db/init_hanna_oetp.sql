-- Hanna OETP RAG Database Schema
-- Run against PostgreSQL (neu-docs-db, port 5433, user: klara)
-- CREATE DATABASE hanna_oetp OWNER klara;
-- \c hanna_oetp

-- Extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;

-- Main chunks table
CREATE TABLE IF NOT EXISTS chunks (
    id VARCHAR(32) PRIMARY KEY,
    doc_id VARCHAR NOT NULL,
    doc_type VARCHAR,
    program VARCHAR DEFAULT 'OETP',
    chunk_index INT,
    title VARCHAR,
    content TEXT,
    content_enriched TEXT,
    embedding vector(1024),
    content_tsvector tsvector,
    metadata JSONB DEFAULT '{}',
    authority_score FLOAT DEFAULT 0.5,
    source_date TIMESTAMP,
    content_hash VARCHAR(16),
    phases TEXT[],
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_type ON chunks(doc_type);
CREATE INDEX IF NOT EXISTS idx_chunks_program ON chunks(program);
CREATE INDEX IF NOT EXISTS idx_chunks_created ON chunks(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chunks_content_hash ON chunks(content_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_tsvector ON chunks USING gin(content_tsvector);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);

-- Auto-generate tsvector on insert/update
CREATE OR REPLACE FUNCTION chunks_tsvector_trigger() RETURNS trigger AS $$
BEGIN
    NEW.content_tsvector := to_tsvector('hungarian', COALESCE(NEW.content, ''));
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS tsvector_update ON chunks;
CREATE TRIGGER tsvector_update BEFORE INSERT OR UPDATE OF content ON chunks
    FOR EACH ROW EXECUTE FUNCTION chunks_tsvector_trigger();

-- KG entities
CREATE TABLE IF NOT EXISTS kg_entities (
    id SERIAL PRIMARY KEY,
    name VARCHAR NOT NULL,
    type VARCHAR,
    aliases TEXT[],
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(name, type)
);
CREATE INDEX IF NOT EXISTS idx_kg_entities_type ON kg_entities(type);
CREATE INDEX IF NOT EXISTS idx_kg_entities_name ON kg_entities(name);

-- KG relations
CREATE TABLE IF NOT EXISTS kg_relations (
    id SERIAL PRIMARY KEY,
    source_id INT REFERENCES kg_entities(id),
    target_id INT REFERENCES kg_entities(id),
    relation_type VARCHAR,
    source_chunk_id VARCHAR(32) REFERENCES chunks(id),
    weight FLOAT DEFAULT 1.0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(source_id, target_id, relation_type)
);
CREATE INDEX IF NOT EXISTS idx_kg_relations_source ON kg_relations(source_id);
CREATE INDEX IF NOT EXISTS idx_kg_relations_target ON kg_relations(target_id);
CREATE INDEX IF NOT EXISTS idx_kg_relations_type ON kg_relations(relation_type);

-- KG entity-chunk links
CREATE TABLE IF NOT EXISTS kg_entity_chunks (
    entity_id INT REFERENCES kg_entities(id),
    chunk_id VARCHAR(32) REFERENCES chunks(id),
    confidence FLOAT DEFAULT 0.85,
    extraction_method VARCHAR DEFAULT 'deterministic',
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (entity_id, chunk_id)
);
CREATE INDEX IF NOT EXISTS idx_kg_ec_chunk ON kg_entity_chunks(chunk_id);

-- Reasoning traces
CREATE TABLE IF NOT EXISTS reasoning_traces (
    id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    query_embedding vector(1024),
    email_message_id VARCHAR,
    sender_name VARCHAR,
    sender_email VARCHAR,
    category VARCHAR,
    program VARCHAR DEFAULT 'OETP',
    phases TEXT[],
    confidence VARCHAR,
    draft_text TEXT,
    sent_text TEXT,
    outcome VARCHAR DEFAULT 'PENDING',
    similarity_score FLOAT,
    top_chunks JSONB,
    rag_scores JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    resolved_at TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_traces_outcome ON reasoning_traces(outcome);
CREATE INDEX IF NOT EXISTS idx_traces_category ON reasoning_traces(category);
CREATE INDEX IF NOT EXISTS idx_traces_program ON reasoning_traces(program);
CREATE INDEX IF NOT EXISTS idx_traces_created ON reasoning_traces(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_traces_email ON reasoning_traces(email_message_id);
CREATE INDEX IF NOT EXISTS idx_traces_embedding ON reasoning_traces USING hnsw (query_embedding vector_cosine_ops);

-- Feedback analytics (learning from draft-vs-sent differences)
CREATE TABLE IF NOT EXISTS feedback_analytics (
    id SERIAL PRIMARY KEY,
    trace_id INT REFERENCES reasoning_traces(id),
    change_types TEXT[],
    lesson TEXT,
    added_content TEXT,
    removed_content TEXT,
    chunk_survival JSONB DEFAULT '[]',
    gap_topics TEXT[],
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_fa_trace ON feedback_analytics(trace_id);
CREATE INDEX IF NOT EXISTS idx_fa_change_types ON feedback_analytics USING gin(change_types);
CREATE INDEX IF NOT EXISTS idx_fa_created ON feedback_analytics(created_at DESC);

-- Chunk survival tracking columns
DO $$ BEGIN
    ALTER TABLE chunks ADD COLUMN IF NOT EXISTS survival_rate FLOAT DEFAULT NULL;
    ALTER TABLE chunks ADD COLUMN IF NOT EXISTS survival_count INT DEFAULT 0;
    ALTER TABLE chunks ADD COLUMN IF NOT EXISTS appearance_count INT DEFAULT 0;
EXCEPTION WHEN duplicate_column THEN NULL;
END $$;
CREATE INDEX IF NOT EXISTS idx_chunks_survival ON chunks(survival_rate) WHERE survival_rate IS NOT NULL;

-- Canonical entities (cross-RAG)
CREATE TABLE IF NOT EXISTS canonical_entities (
    id SERIAL PRIMARY KEY,
    canonical_name VARCHAR NOT NULL,
    canonical_type VARCHAR,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(canonical_name, canonical_type)
);
CREATE INDEX IF NOT EXISTS idx_canonical_type ON canonical_entities(canonical_type);

-- Entity links (cross-RAG)
CREATE TABLE IF NOT EXISTS entity_links (
    id SERIAL PRIMARY KEY,
    canonical_id INT REFERENCES canonical_entities(id),
    source_db VARCHAR,
    entity_id VARCHAR,
    entity_name VARCHAR,
    confidence FLOAT DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(canonical_id, source_db, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_el_canonical ON entity_links(canonical_id);
CREATE INDEX IF NOT EXISTS idx_el_source_db ON entity_links(source_db);
