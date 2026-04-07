"""Re-embed chunks using enriched content.

Updates the embedding vector in the chunks table using content_enriched
(contextual summary) concatenated with original content for richer semantic search.

Usage: python3 re_embed_enriched.py [--batch-size 50] [--dry-run]
"""

import asyncio
import sys
import time
import argparse
import httpx
import asyncpg
import json
import struct

DB_URL = os.environ.get("HANNA_PG_DSN", "postgresql://klara:klara_docs_2026@hanna-db:5432/hanna_oetp")
BGE_URL = "http://host.docker.internal:8104"


async def get_embedding(client: httpx.AsyncClient, text: str) -> list[float]:
    """Get embedding from BGE-M3 service."""
    resp = await client.post(f"{BGE_URL}/embed", json={"texts": [text]})
    resp.raise_for_status()
    data = resp.json()
    return data["embeddings"][0]


def embedding_to_pgvector(embedding: list[float]) -> str:
    """Convert embedding list to pgvector string format."""
    return "[" + ",".join(f"{x:.6f}" for x in embedding) + "]"


async def main(batch_size: int = 50, dry_run: bool = False, since: str = None):
    conn = await asyncpg.connect(DB_URL)

    # Count chunks that need re-embedding
    if since:
        # Only re-embed chunks updated after a given timestamp
        total = await conn.fetchval(
            "SELECT count(*) FROM chunks WHERE content_enriched IS NOT NULL AND updated_at >= $1::timestamptz",
            since,
        )
        print(f"Chunks with enriched content updated since {since}: {total}")
    else:
        total = await conn.fetchval(
            "SELECT count(*) FROM chunks WHERE content_enriched IS NOT NULL"
        )
        print(f"Total chunks with enriched content: {total}")
        print(f"TIP: Use --since '2026-03-15' to only re-embed recently changed chunks")
    
    if dry_run:
        sample = await conn.fetchrow(
            "SELECT id, content_enriched, content FROM chunks WHERE content_enriched IS NOT NULL LIMIT 1"
        )
        if sample:
            combined = f"{sample['content_enriched']}\n\n{sample['content']}"
            print(f"Sample combined text ({len(combined)} chars):")
            print(combined[:300] + "...")
        print(f"\nDry run — would re-embed {total} chunks in batches of {batch_size}")
        await conn.close()
        return
    
    # Process in batches
    offset = 0
    updated = 0
    errors = 0
    start_time = time.time()
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        while offset < total:
            if since:
                rows = await conn.fetch(
                    """SELECT id, content_enriched, content
                       FROM chunks
                       WHERE content_enriched IS NOT NULL
                         AND updated_at >= $3::timestamptz
                       ORDER BY id
                       LIMIT $1 OFFSET $2""",
                    batch_size, offset, since,
                )
            else:
                rows = await conn.fetch(
                    """SELECT id, content_enriched, content
                       FROM chunks
                       WHERE content_enriched IS NOT NULL
                       ORDER BY id
                       LIMIT $1 OFFSET $2""",
                    batch_size, offset,
                )
            
            if not rows:
                break
            
            for row in rows:
                try:
                    # Combine enriched summary + original content for embedding
                    combined = f"{row['content_enriched']}\n\n{row['content']}"
                    
                    # Get new embedding
                    embedding = await get_embedding(client, combined)
                    
                    # Update in DB
                    vec_str = embedding_to_pgvector(embedding)
                    await conn.execute(
                        "UPDATE chunks SET embedding = $1::vector WHERE id = $2",
                        vec_str, row['id']
                    )
                    updated += 1
                    
                except Exception as e:
                    errors += 1
                    print(f"  Error for {row['id']}: {e}")
            
            offset += batch_size
            elapsed = time.time() - start_time
            rate = updated / elapsed if elapsed > 0 else 0
            eta = (total - updated) / rate if rate > 0 else 0
            print(f"  Progress: {updated}/{total} ({updated*100//total}%) | {rate:.1f} chunks/s | ETA: {eta:.0f}s | Errors: {errors}")
    
    elapsed = time.time() - start_time
    print(f"\n=== Re-embedding Complete ===")
    print(f"Updated: {updated}/{total}")
    print(f"Errors: {errors}")
    print(f"Time: {elapsed:.1f}s ({updated/elapsed:.1f} chunks/s)")
    
    await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--since", type=str, default=None,
                        help="Only re-embed chunks updated after this timestamp (e.g. '2026-03-15')")
    args = parser.parse_args()

    asyncio.run(main(batch_size=args.batch_size, dry_run=args.dry_run, since=args.since))
