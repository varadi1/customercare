#!/usr/bin/env python3
"""Re-ingest existing chunks with contextual embeddings.

Reads all chunks from ChromaDB, enriches them with context prefix,
re-embeds, and updates in-place. Does NOT change the stored text,
only the embedding vectors.

Usage (inside Docker container):
    python /app/scripts/reingest_contextual.py [--batch-size 200] [--dry-run]
"""

import sys
import os
import argparse
import time

# Add app to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.config import settings
from app.rag.search import get_collection
from app.rag.contextual import enrich_chunk
from app.rag.embeddings import embed_texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    collection = get_collection()
    total = collection.count()
    print(f"[reingest] Collection: {settings.chroma_collection}, total chunks: {total}")

    if args.dry_run:
        # Show sample
        sample = collection.get(limit=5, include=["documents", "metadatas"])
        for i, doc_id in enumerate(sample["ids"]):
            meta = sample["metadatas"][i]
            text = sample["documents"][i][:100]
            enriched = enrich_chunk(
                text=sample["documents"][i],
                chunk_type=meta.get("chunk_type", "general"),
                source=meta.get("source", "unknown"),
                category=meta.get("category", ""),
            )
            print(f"\n--- Chunk {doc_id} ---")
            print(f"  Type: {meta.get('chunk_type')}")
            print(f"  Source: {meta.get('source')}")
            print(f"  Original: {text}...")
            print(f"  Enriched prefix: {enriched[:200]}...")
        print(f"\n[reingest] Dry run complete. Would process {total} chunks.")
        return

    # Process in batches
    batch_size = args.batch_size
    processed = 0
    offset = 0
    start = time.time()

    while offset < total:
        # Fetch batch
        batch = collection.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"],
        )

        if not batch["ids"]:
            break

        ids = batch["ids"]
        documents = batch["documents"]
        metadatas = batch["metadatas"]

        # Enrich and re-embed
        enriched_texts = []
        for i, doc in enumerate(documents):
            meta = metadatas[i]
            enriched = enrich_chunk(
                text=doc,
                chunk_type=meta.get("chunk_type", "general"),
                source=meta.get("source", "unknown"),
                category=meta.get("category", ""),
            )
            enriched_texts.append(enriched)

        # Batch embed
        new_embeddings = embed_texts(enriched_texts)

        # Update embeddings in ChromaDB (text stays the same)
        collection.update(
            ids=ids,
            embeddings=new_embeddings,
        )

        processed += len(ids)
        elapsed = time.time() - start
        rate = processed / elapsed if elapsed > 0 else 0
        print(f"[reingest] {processed}/{total} chunks re-embedded ({rate:.1f} chunks/sec)")

        offset += batch_size

    elapsed = time.time() - start
    print(f"\n[reingest] DONE! {processed} chunks re-embedded in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
