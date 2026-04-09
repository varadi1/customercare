#!/usr/bin/env python3
"""Migration: creates reasoning_traces table + chunks.phases column.
Safe to re-run (IF NOT EXISTS). Usage: python3 scripts/migrate_reasoning.py"""
import asyncio, asyncpg

PG_DSN = "postgresql://klara:klara_docs_2026@localhost:5433/customercare"

async def migrate():
    conn = await asyncpg.connect(PG_DSN)
    try:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS reasoning_traces (
                id SERIAL PRIMARY KEY, query_text TEXT NOT NULL,
                query_embedding vector(1024), email_message_id VARCHAR,
                sender_name VARCHAR, sender_email VARCHAR, category VARCHAR,
                program VARCHAR DEFAULT 'OETP', phases TEXT[], confidence VARCHAR,
                draft_text TEXT, sent_text TEXT, outcome VARCHAR DEFAULT 'PENDING',
                similarity_score FLOAT, top_chunks JSONB, rag_scores JSONB,
                created_at TIMESTAMP DEFAULT NOW(), resolved_at TIMESTAMP)""")
        print("[OK] reasoning_traces table")
        for idx in [
            "CREATE INDEX IF NOT EXISTS idx_traces_embedding ON reasoning_traces USING hnsw (query_embedding vector_cosine_ops)",
            "CREATE INDEX IF NOT EXISTS idx_traces_outcome ON reasoning_traces (outcome)",
            "CREATE INDEX IF NOT EXISTS idx_traces_category ON reasoning_traces (category)",
            "CREATE INDEX IF NOT EXISTS idx_traces_program ON reasoning_traces (program)",
            "CREATE INDEX IF NOT EXISTS idx_traces_created ON reasoning_traces (created_at DESC)",
        ]:
            await conn.execute(idx)
        print("[OK] indexes")
        await conn.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS phases TEXT[] DEFAULT NULL")
        print("[OK] chunks.phases")
        count = await conn.fetchval("SELECT COUNT(*) FROM reasoning_traces")
        print(f"\nDone. reasoning_traces: {count} rows.")
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(migrate())
