"""Migrate Hanna OETP data from ChromaDB to PostgreSQL.

Usage (inside Docker container):
    python3 /app/scripts/migrate_to_pg.py [--dry-run]

Or from host:
    docker exec cc-backend python3 /app/scripts/migrate_to_pg.py
"""

import asyncio
import hashlib
import json
import sys
from datetime import datetime

import asyncpg
import chromadb

# --- Config ---
CHROMA_HOST = "chromadb"
CHROMA_PORT = 8000
CHROMA_COLLECTION = "hanna_knowledge_bge_m3"

import os
PG_DSN = os.environ.get("HANNA_PG_DSN", "postgresql://klara:klara_docs_2026@cc-db:5432/customercare")

# chunk_type → doc_type mapping
DOC_TYPE_MAP = {
    "palyazat_felhivas": "felhívás",
    "palyazat_melleklet": "melléklet",
    "kozlemeny": "közlemény",
    "gyik": "gyik",
    "segedlet": "segédlet",
    "email_reply": "email_reply",
    "email_question": "email_question",
    "document": "dokumentum",
}

# doc_type → authority_score
AUTHORITY_MAP = {
    "felhívás": 0.95,
    "melléklet": 0.90,
    "közlemény": 0.85,
    "gyik": 0.80,
    "segédlet": 0.75,
    "dokumentum": 0.60,
    "email_question": 0.35,
    "email_reply": 0.40,
}

# category → program mapping
PROGRAM_MAP = {
    "oetp": "OETP",
    "felhívás": "OETP",
    "távhő": "Távhő",
    "NPP is vagy RRF6.2": "NPP2/RRF",
}

DRY_RUN = "--dry-run" in sys.argv


def resolve_doc_type(chunk_type: str) -> str:
    return DOC_TYPE_MAP.get(chunk_type, "dokumentum")


def resolve_program(category: str) -> str | None:
    return PROGRAM_MAP.get(category, "OETP")  # default OETP


def resolve_authority(doc_type: str) -> float:
    return AUTHORITY_MAP.get(doc_type, 0.5)


def make_content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


async def migrate():
    # Connect to ChromaDB
    print(f"Connecting to ChromaDB ({CHROMA_HOST}:{CHROMA_PORT})...")
    chroma = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    col = chroma.get_or_create_collection(CHROMA_COLLECTION)

    total = col.count()
    print(f"ChromaDB collection: {CHROMA_COLLECTION}, chunks: {total}")

    if DRY_RUN:
        print("DRY RUN — no data will be written to PostgreSQL")

    # Get ALL data from ChromaDB (batch)
    print("Fetching all data from ChromaDB...")
    batch_size = 1000
    all_ids = []
    all_documents = []
    all_metadatas = []
    all_embeddings = []

    for offset in range(0, total, batch_size):
        batch = col.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )
        all_ids.extend(batch["ids"])
        all_documents.extend(batch["documents"])
        all_metadatas.extend(batch["metadatas"])
        all_embeddings.extend(batch["embeddings"])
        print(f"  Fetched {len(all_ids)}/{total}")

    print(f"Total fetched: {len(all_ids)} chunks")

    if DRY_RUN:
        # Print sample
        for i in range(min(5, len(all_ids))):
            m = all_metadatas[i]
            dt = resolve_doc_type(m.get("chunk_type", ""))
            print(f"  [{i}] id={all_ids[i][:30]} doc_type={dt} "
                  f"authority={resolve_authority(dt)} "
                  f"program={resolve_program(m.get('category', ''))}")
        print(f"\nDRY RUN complete. {len(all_ids)} chunks would be migrated.")
        return

    # Connect to PostgreSQL
    print(f"\nConnecting to PostgreSQL...")
    pool = await asyncpg.create_pool(PG_DSN, min_size=2, max_size=10)

    # Migrate in batches
    inserted = 0
    skipped = 0
    errors = 0

    async with pool.acquire() as conn:
        for i in range(len(all_ids)):
            chunk_id = all_ids[i]
            content = all_documents[i]
            meta = all_metadatas[i]
            embedding = all_embeddings[i]

            if not content or not content.strip():
                skipped += 1
                continue

            doc_type = resolve_doc_type(meta.get("chunk_type", ""))
            program = resolve_program(meta.get("category", ""))
            authority = resolve_authority(doc_type)

            # Parse dates
            source_date = None
            vf = meta.get("valid_from", "")
            if vf:
                try:
                    source_date = datetime.fromisoformat(vf)
                except (ValueError, TypeError):
                    pass

            try:
                await conn.execute(
                    """
                    INSERT INTO chunks (
                        id, doc_id, doc_type, program, chunk_index, title,
                        content, embedding, metadata, authority_score,
                        source_date, content_hash
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8::vector, $9, $10, $11, $12)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    chunk_id,
                    meta.get("source", "unknown"),
                    doc_type,
                    program,
                    meta.get("chunk_index", 0),
                    meta.get("source", ""),
                    content,
                    "[" + ",".join(str(float(x)) for x in embedding) + "]",
                    json.dumps({
                        "category": meta.get("category", ""),
                        "chunk_type": meta.get("chunk_type", ""),
                        "version": meta.get("version", 1),
                        "supersedes": meta.get("supersedes", ""),
                        "valid_from": meta.get("valid_from", ""),
                        "valid_to": meta.get("valid_to", ""),
                        "indexed_at": meta.get("indexed_at", ""),
                    }),
                    authority,
                    source_date,
                    make_content_hash(content),
                )
                inserted += 1
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  ERROR inserting {chunk_id}: {e}")

            if (i + 1) % 500 == 0:
                print(f"  Progress: {i + 1}/{len(all_ids)} (inserted={inserted}, skip={skipped}, err={errors})")

    await pool.close()

    print(f"\n=== Migration complete ===")
    print(f"Total: {len(all_ids)}")
    print(f"Inserted: {inserted}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {errors}")

    # Verify
    pool2 = await asyncpg.create_pool(PG_DSN, min_size=1, max_size=2)
    async with pool2.acquire() as conn:
        count = await conn.fetchval("SELECT count(*) FROM chunks")
        print(f"\nPostgreSQL chunks count: {count}")
        
        # Distribution
        rows = await conn.fetch(
            "SELECT doc_type, count(*), avg(authority_score)::numeric(3,2) "
            "FROM chunks GROUP BY doc_type ORDER BY count DESC"
        )
        print("\nDoc type distribution:")
        for r in rows:
            print(f"  {r['doc_type']:20s} {r['count']:6d}  authority={r['avg']}")
    await pool2.close()


if __name__ == "__main__":
    asyncio.run(migrate())
