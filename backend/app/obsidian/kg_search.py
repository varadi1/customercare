"""Knowledge Graph augmented search.

Given a query, detects entities, traverses the graph for related entities,
and returns additional chunk IDs to boost in the main hybrid search.
"""

from __future__ import annotations

import json
from typing import Any

import asyncpg

PG_DSN = "postgresql://klara:klara_docs_2026@host.docker.internal:5433/obsidian_rag"

_pool: asyncpg.Pool | None = None


async def _get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(PG_DSN, min_size=2, max_size=5)
    return _pool


async def detect_entities(query: str, threshold: float = 0.25, limit: int = 5) -> list[dict]:
    """Detect KG entities mentioned in the query using trigram similarity.

    Returns list of matched entities with id, name, type, similarity.
    """
    pool = await _get_pool()

    # Split query into meaningful tokens (2+ words for better entity matching)
    words = query.split()
    candidates = []

    # Try the full query
    candidates.append(query)

    # Try consecutive word pairs and triples
    for i in range(len(words)):
        if i + 1 < len(words):
            candidates.append(f"{words[i]} {words[i+1]}")
        if i + 2 < len(words):
            candidates.append(f"{words[i]} {words[i+1]} {words[i+2]}")
        # Single words (only if 4+ chars to avoid noise)
        if len(words[i]) >= 4:
            candidates.append(words[i])

    # Deduplicate
    seen = set()
    unique_candidates = []
    for c in candidates:
        cl = c.lower()
        if cl not in seen:
            seen.add(cl)
            unique_candidates.append(c)

    matched = {}  # name → entity dict (keep highest similarity)

    for candidate in unique_candidates:
        rows = await pool.fetch(
            """SELECT id, name, type, aliases,
                      similarity(name, $1) AS sim
               FROM kg_entities
               WHERE similarity(name, $1) > $2
               ORDER BY similarity(name, $1) DESC
               LIMIT $3""",
            candidate, threshold, limit,
        )
        for r in rows:
            key = (r["name"], r["type"])
            sim = float(r["sim"])
            if key not in matched or sim > matched[key]["similarity"]:
                matched[key] = {
                    "id": r["id"],
                    "name": r["name"],
                    "type": r["type"],
                    "similarity": sim,
                }

    # Sort by similarity descending, limit
    results = sorted(matched.values(), key=lambda x: x["similarity"], reverse=True)
    return results[:limit]


async def expand_entities(
    entity_ids: list[int],
    max_hops: int = 1,
    max_related: int = 20,
) -> list[dict]:
    """Traverse the graph from given entities to find related entities.

    Returns related entities with their relation context.
    """
    pool = await _get_pool()
    if not entity_ids:
        return []

    # Get direct relations (1-hop)
    rows = await pool.fetch(
        """SELECT DISTINCT
               CASE WHEN r.source_id = ANY($1::int[]) THEN t.id ELSE s.id END AS related_id,
               CASE WHEN r.source_id = ANY($1::int[]) THEN t.name ELSE s.name END AS related_name,
               CASE WHEN r.source_id = ANY($1::int[]) THEN t.type ELSE s.type END AS related_type,
               r.relation_type,
               r.confidence,
               CASE WHEN r.source_id = ANY($1::int[]) THEN s.name ELSE t.name END AS from_entity
           FROM kg_relations r
           JOIN kg_entities s ON s.id = r.source_id
           JOIN kg_entities t ON t.id = r.target_id
           WHERE r.source_id = ANY($1::int[]) OR r.target_id = ANY($1::int[])
           ORDER BY r.confidence DESC
           LIMIT $2""",
        entity_ids, max_related,
    )

    related = []
    seen_ids = set(entity_ids)
    for r in rows:
        rid = r["related_id"]
        if rid in seen_ids:
            continue
        seen_ids.add(rid)
        related.append({
            "id": rid,
            "name": r["related_name"],
            "type": r["related_type"],
            "relation_type": r["relation_type"],
            "confidence": float(r["confidence"]),
            "from_entity": r["from_entity"],
        })

    return related


async def get_entity_file_paths(entity_ids: list[int]) -> list[str]:
    """Get file paths associated with entities (from source_file)."""
    pool = await _get_pool()
    if not entity_ids:
        return []

    rows = await pool.fetch(
        """SELECT DISTINCT source_file
           FROM kg_entities
           WHERE id = ANY($1::int[]) AND source_file IS NOT NULL""",
        entity_ids,
    )
    return [r["source_file"] for r in rows]


async def get_graph_boosted_chunks(
    file_paths: list[str],
    limit: int = 10,
) -> list[dict]:
    """Get chunks from files related to graph-expanded entities."""
    pool = await _get_pool()
    if not file_paths:
        return []

    rows = await pool.fetch(
        """SELECT chunk_id, content, context_prefix, file_path, file_name, folder, metadata
           FROM obsidian_chunks
           WHERE file_path = ANY($1::text[])
           ORDER BY chunk_index ASC
           LIMIT $2""",
        file_paths, limit,
    )

    results = []
    for r in rows:
        content = r["content"]
        ctx = r["context_prefix"]
        meta = json.loads(r["metadata"]) if r["metadata"] else {}
        meta.update({
            "file_path": r["file_path"],
            "file_name": r["file_name"],
            "folder": r["folder"],
        })
        results.append({
            "id": r["chunk_id"],
            "content": f"{ctx}\n\n{content}" if ctx else content,
            "text": f"{ctx}\n\n{content}" if ctx else content,
            "context_prefix": ctx,
            "metadata": meta,
            "graph_boosted": True,
        })

    return results


async def graph_augmented_search(
    query: str,
    entity_threshold: float = 0.35,
    max_entities: int = 5,
    max_related: int = 15,
    max_graph_chunks: int = 8,
) -> dict[str, Any]:
    """Full graph expansion pipeline for a query.

    Returns:
        {
            "matched_entities": [...],
            "related_entities": [...],
            "graph_chunks": [...],
            "graph_file_paths": [...],
        }
    """
    # Step 1: Detect entities in query
    matched = await detect_entities(query, threshold=entity_threshold, limit=max_entities)
    if not matched:
        return {
            "matched_entities": [],
            "related_entities": [],
            "graph_chunks": [],
            "graph_file_paths": [],
        }

    entity_ids = [e["id"] for e in matched]

    # Step 2: Expand graph (1-hop related entities)
    related = await expand_entities(entity_ids, max_related=max_related)

    # Step 3: Collect file paths from both matched + related entities
    all_entity_ids = entity_ids + [e["id"] for e in related]
    file_paths = await get_entity_file_paths(all_entity_ids)

    # Step 4: Get chunks from those files
    graph_chunks = await get_graph_boosted_chunks(file_paths, limit=max_graph_chunks)

    return {
        "matched_entities": matched,
        "related_entities": related,
        "graph_chunks": graph_chunks,
        "graph_file_paths": file_paths,
    }
