#!/usr/bin/env python3
"""Migrate CustomerCare collection from OpenAI embeddings to BGE-M3.

Creates cc_knowledge_bge_m3 collection, re-embeds all chunks, verifies count.
"""

import json, urllib.request, sys, time

CHROMA_BASE = "http://localhost:8100/api/v2/tenants/default_tenant/databases/default_database/collections"
BGE_M3_URL = "http://localhost:8104"
OLD_COLLECTION = "cc_knowledge"
NEW_COLLECTION = "cc_knowledge_bge_m3"
BATCH_SIZE = 50  # chunks per batch for embedding
FETCH_BATCH = 100  # chunks per ChromaDB get


def chroma_get(path):
    req = urllib.request.Request(f"{CHROMA_BASE}/{path}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def chroma_post(path, data):
    payload = json.dumps(data).encode()
    req = urllib.request.Request(f"{CHROMA_BASE}/{path}", data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def embed_bge(texts):
    data = json.dumps({"texts": texts}).encode()
    req = urllib.request.Request(f"{BGE_M3_URL}/embed", data=data)
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["embeddings"]


def get_collection_id(name):
    cols = chroma_get("")
    for c in cols:
        if c["name"] == name:
            return c["id"]
    return None


def create_collection(name):
    data = json.dumps({"name": name, "metadata": {"hnsw:space": "cosine"}}).encode()
    req = urllib.request.Request(CHROMA_BASE, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
    return result["id"]


def get_count(col_id):
    req = urllib.request.Request(f"{CHROMA_BASE}/{col_id}/count")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def main():
    print("=== CustomerCare BGE-M3 Migration ===\n")

    # 1. Get old collection
    old_id = get_collection_id(OLD_COLLECTION)
    if not old_id:
        print(f"ERROR: Collection '{OLD_COLLECTION}' not found!")
        sys.exit(1)
    old_count = get_count(old_id)
    print(f"Source: {OLD_COLLECTION} ({old_count} chunks)")

    # 2. Create or get new collection
    new_id = get_collection_id(NEW_COLLECTION)
    if new_id:
        new_count = get_count(new_id)
        print(f"Target exists: {NEW_COLLECTION} ({new_count} chunks)")
        if new_count > 0 and "--force" not in sys.argv:
            print("Target not empty! Use --force to overwrite.")
            sys.exit(1)
    else:
        new_id = create_collection(NEW_COLLECTION)
        print(f"Created: {NEW_COLLECTION} (id: {new_id})")

    # 3. Fetch all chunks from old collection in batches
    print(f"\nFetching {old_count} chunks from {OLD_COLLECTION}...")
    all_ids = []
    all_docs = []
    all_metas = []

    offset = 0
    while offset < old_count:
        batch = chroma_post(f"{old_id}/get", {
            "limit": FETCH_BATCH,
            "offset": offset,
            "include": ["documents", "metadatas"],
        })
        ids = batch.get("ids", [])
        docs = batch.get("documents", [])
        metas = batch.get("metadatas", [])

        if not ids:
            break

        all_ids.extend(ids)
        all_docs.extend(docs)
        all_metas.extend(metas)
        offset += len(ids)

        if offset % 1000 == 0 or offset >= old_count:
            print(f"  Fetched {offset}/{old_count}")

    print(f"Total fetched: {len(all_ids)} chunks")

    # 4. Re-embed and upsert in batches
    print(f"\nRe-embedding with BGE-M3 and upserting to {NEW_COLLECTION}...")
    total = 0
    errors = 0

    for i in range(0, len(all_ids), BATCH_SIZE):
        batch_ids = all_ids[i:i + BATCH_SIZE]
        batch_docs = all_docs[i:i + BATCH_SIZE]
        batch_metas = all_metas[i:i + BATCH_SIZE]

        # Clean metadata (ChromaDB requires str/int/float/bool values)
        clean_metas = []
        for m in batch_metas:
            if m is None:
                clean_metas.append({})
                continue
            clean = {}
            for k, v in m.items():
                if isinstance(v, (str, int, float, bool)):
                    clean[k] = v
                elif v is None:
                    clean[k] = ""
                else:
                    clean[k] = str(v)
            clean_metas.append(clean)

        # Filter out empty docs
        valid = [(id_, doc, meta) for id_, doc, meta in zip(batch_ids, batch_docs, clean_metas) if doc]
        if not valid:
            continue

        v_ids, v_docs, v_metas = zip(*valid)

        try:
            embeddings = embed_bge(list(v_docs))
            chroma_post(f"{new_id}/upsert", {
                "ids": list(v_ids),
                "documents": list(v_docs),
                "embeddings": embeddings,
                "metadatas": list(v_metas),
            })
            total += len(v_ids)
        except Exception as e:
            print(f"  ERROR batch {i}-{i+BATCH_SIZE}: {e}")
            errors += 1

        if total % 500 == 0 or total + BATCH_SIZE >= len(all_ids):
            print(f"  Migrated: {total}/{len(all_ids)}")

    # 5. Verify
    new_count = get_count(new_id)
    print(f"\n=== Migration Complete ===")
    print(f"Source: {old_count} chunks")
    print(f"Target: {new_count} chunks")
    print(f"Errors: {errors}")

    if new_count >= old_count * 0.99:
        print("✅ Migration successful!")
    else:
        print(f"⚠️  Count mismatch! Expected ~{old_count}, got {new_count}")


if __name__ == "__main__":
    main()
