#!/usr/bin/env python3
"""Backfill KG extraction for chunks that don't have KG entities yet.

Usage inside cc-backend container:
    python3 /app/scripts/kg_backfill_new.py
    python3 /app/scripts/kg_backfill_new.py --doc-id '%260407%'  # specific doc
"""

import argparse
import asyncio
import sys

sys.path.insert(0, "/app")

import asyncpg


async def main(doc_id_filter: str | None = None):
    dsn = "postgresql://klara:klara_docs_2026@cc-db:5432/customercare"
    pool = await asyncpg.create_pool(dsn, min_size=2, max_size=5)

    # Find chunks without KG entities
    async with pool.acquire() as conn:
        if doc_id_filter:
            rows = await conn.fetch(
                "SELECT c.id, c.doc_id, c.doc_type FROM chunks c "
                "LEFT JOIN kg_entity_chunks ec ON c.id = ec.chunk_id "
                "WHERE c.doc_id ILIKE $1 AND ec.entity_id IS NULL",
                doc_id_filter,
            )
        else:
            # All high-value chunks without KG
            rows = await conn.fetch(
                "SELECT c.id, c.doc_id, c.doc_type FROM chunks c "
                "LEFT JOIN kg_entity_chunks ec ON c.id = ec.chunk_id "
                "WHERE c.doc_type IN ('felhívás','melléklet','közlemény','gyik','segédlet','dokumentum') "
                "AND ec.entity_id IS NULL"
            )

    chunk_ids = [r["id"] for r in rows]
    print(f"Chunks without KG entities: {len(chunk_ids)}")

    if not chunk_ids:
        print("Nothing to do.")
        await pool.close()
        return

    # Show doc breakdown
    from collections import Counter
    doc_counts = Counter(r["doc_id"] for r in rows)
    for doc, cnt in doc_counts.most_common(10):
        print(f"  {cnt:4d} chunks | {doc[:70]}")

    from app.rag.kg_extract import extract_and_store

    total_ent = 0
    total_rel = 0
    for i, row in enumerate(rows):
        async with pool.acquire() as conn:
            chunk = await conn.fetchval("SELECT content FROM chunks WHERE id = $1", row["id"])
            ent, rel = await extract_and_store(conn, row["id"], chunk, row["doc_type"], row["doc_id"])
            total_ent += ent
            total_rel += rel

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(rows)}] entities={total_ent}, relations={total_rel}")

    print(f"\nDone: {total_ent} entities, {total_rel} relations from {len(rows)} chunks")

    await pool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-id", help="Filter by doc_id ILIKE pattern")
    args = parser.parse_args()
    asyncio.run(main(args.doc_id))
