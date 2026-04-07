"""Knowledge Graph search for Hanna OETP RAG.

Implements entity-based expansion: query → entities → 1-hop relations → related chunks.
"""

from __future__ import annotations

import asyncpg
from typing import Optional
import json
import re

# Use the same PostgreSQL connection as search.py
import os
PG_DSN = os.environ.get("HANNA_PG_DSN", "postgresql://klara:klara_docs_2026@hanna-db:5432/hanna_oetp")

# Connection pool (shared with search.py if possible)
_kg_pool: Optional[asyncpg.Pool] = None


async def _get_kg_pool() -> asyncpg.Pool:
    """Get or create the PostgreSQL connection pool for KG operations."""
    global _kg_pool
    if _kg_pool is None:
        _kg_pool = await asyncpg.create_pool(PG_DSN, min_size=1, max_size=5)
    return _kg_pool


async def kg_search(query: str, top_k: int = 10, only_valid: bool = True) -> list[dict]:
    """Knowledge Graph-based search expansion.

    Pipeline:
    1. Extract entity names from query (ILIKE match in kg_entities)
    2. Find 1-hop neighbors via kg_relations
    3. Get chunks associated with matched + neighbor entities
    4. Return chunks in search result format

    Args:
        query: Search query string
        top_k: Maximum chunks to return
        only_valid: If True, exclude expired chunks (valid_to set)

    Returns:
        List of chunk dicts compatible with search.py format
    """
    if not query.strip():
        return []

    try:
        pool = await _get_kg_pool()
    except Exception as e:
        print(f"[kg-search] Pool connection failed: {e}")
        return []

    try:
        # Step 1: Find entities matching query terms
        matched_entities = await _find_matching_entities(pool, query)
        if not matched_entities:
            return []

        # Step 2: Expand to 1-hop neighbors
        all_entities = await _expand_entities_1hop(pool, matched_entities)

        # Step 3: Get chunks for all entities (matched + neighbors)
        chunks = await _get_entity_chunks(pool, all_entities, top_k, only_valid)
        
        print(f"[kg-search] Query '{query}' → {len(matched_entities)} entities → "
              f"{len(all_entities)} total → {len(chunks)} chunks")
        
        return chunks
        
    except Exception as e:
        print(f"[kg-search] KG search failed for '{query}': {e}")
        return []


async def _find_matching_entities(pool: asyncpg.Pool, query: str) -> list[dict]:
    """Find entities whose names match query terms (ILIKE)."""
    # Simple approach: split query into words, find entities containing any word
    query_words = [w.strip().lower() for w in re.split(r'[,\s]+', query) if len(w.strip()) >= 3]
    
    if not query_words:
        return []
    
    # Build ILIKE conditions for each word
    ilike_conditions = []
    params = []
    for i, word in enumerate(query_words[:5]):  # Max 5 words to avoid too complex query
        ilike_conditions.append(f"name ILIKE ${i+1}")
        params.append(f"%{word}%")
    
    if not ilike_conditions:
        return []
    
    sql = f"""
        SELECT id, name, type, metadata
        FROM kg_entities 
        WHERE {' OR '.join(ilike_conditions)}
        ORDER BY char_length(name) ASC
        LIMIT 20
    """
    
    try:
        rows = await pool.fetch(sql, *params)
        entities = []
        for r in rows:
            entities.append({
                "id": r["id"],
                "name": r["name"],
                "type": r["type"],
                "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
            })
        return entities
    except Exception as e:
        print(f"[kg-search] Entity search failed: {e}")
        return []


async def _expand_entities_1hop(pool: asyncpg.Pool, seed_entities: list[dict]) -> list[dict]:
    """Expand entities to include 1-hop neighbors via kg_relations."""
    if not seed_entities:
        return []
    
    seed_ids = [e["id"] for e in seed_entities]
    
    try:
        # Find relations where seed entities are source OR target
        sql = """
            SELECT DISTINCT 
                CASE 
                    WHEN source_id = ANY($1) THEN target_id
                    WHEN target_id = ANY($1) THEN source_id
                END as neighbor_id
            FROM kg_relations 
            WHERE source_id = ANY($1) OR target_id = ANY($1)
        """
        
        neighbor_rows = await pool.fetch(sql, seed_ids)
        neighbor_ids = [r["neighbor_id"] for r in neighbor_rows if r["neighbor_id"]]
        
        if not neighbor_ids:
            return seed_entities  # No neighbors found
        
        # Get neighbor entity details
        neighbor_sql = """
            SELECT id, name, type, metadata
            FROM kg_entities 
            WHERE id = ANY($1)
        """
        
        neighbor_entity_rows = await pool.fetch(neighbor_sql, neighbor_ids)
        neighbor_entities = []
        for r in neighbor_entity_rows:
            neighbor_entities.append({
                "id": r["id"],
                "name": r["name"],
                "type": r["type"],
                "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
                "is_neighbor": True,
            })
        
        # Combine seed + neighbors (deduplicate by ID)
        all_entities = list(seed_entities)  # Copy seed entities
        existing_ids = {e["id"] for e in seed_entities}
        
        for neighbor in neighbor_entities:
            if neighbor["id"] not in existing_ids:
                all_entities.append(neighbor)
                existing_ids.add(neighbor["id"])
        
        return all_entities
        
    except Exception as e:
        print(f"[kg-search] Entity expansion failed: {e}")
        return seed_entities  # Fallback to just seed entities


async def _get_entity_chunks(pool: asyncpg.Pool, entities: list[dict], limit: int, only_valid: bool = True) -> list[dict]:
    """Get chunks associated with the given entities."""
    if not entities:
        return []

    entity_ids = [e["id"] for e in entities]

    try:
        # Join kg_entity_chunks with chunks table
        valid_filter = ""
        if only_valid:
            valid_filter = "AND (c.metadata->>'valid_to' IS NULL OR c.metadata->>'valid_to' = '')"

        sql = f"""
            SELECT DISTINCT c.id, c.doc_id, c.doc_type, c.program, c.title,
                   c.content, c.content_enriched, c.metadata, c.authority_score,
                   c.source_date,
                   COUNT(ec.entity_id) as entity_count
            FROM chunks c
            INNER JOIN kg_entity_chunks ec ON c.id = ec.chunk_id
            WHERE ec.entity_id = ANY($1)
            {valid_filter}
            GROUP BY c.id, c.doc_id, c.doc_type, c.program, c.title,
                     c.content, c.content_enriched, c.metadata, c.authority_score,
                     c.source_date
            ORDER BY entity_count DESC, c.authority_score DESC
            LIMIT $2
        """

        chunk_rows = await pool.fetch(sql, entity_ids, limit)
        
        chunks = []
        for r in chunk_rows:
            # Parse metadata if it's JSON string
            metadata = {}
            if r["metadata"]:
                try:
                    metadata = json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"]
                except (json.JSONDecodeError, TypeError):
                    metadata = {}
            
            # Format chunk in same style as search.py _semantic_search
            chunk = {
                "id": r["id"],
                "text": r["content_enriched"] or r["content"],
                "source": r["doc_id"],
                "category": r["program"],
                "chunk_type": r["doc_type"],
                "score": 0.5,  # Base score for KG results (will be adjusted by RRF)
                "metadata": {
                    **metadata,
                    "title": r["title"],
                    "authority_score": float(r["authority_score"]) if r["authority_score"] else 0.5,
                    "source_date": r["source_date"].isoformat() if r["source_date"] else None,
                    "entity_count": r["entity_count"],  # How many entities match this chunk
                },
                "kg_score": 0.5,
                "entity_count": r["entity_count"],
            }
            chunks.append(chunk)
        
        return chunks
        
    except Exception as e:
        print(f"[kg-search] Chunk retrieval failed: {e}")
        return []


async def get_kg_stats() -> dict:
    """Get statistics about the knowledge graph."""
    try:
        pool = await _get_kg_pool()
        
        entities_count = await pool.fetchval("SELECT COUNT(*) FROM kg_entities")
        relations_count = await pool.fetchval("SELECT COUNT(*) FROM kg_relations")
        entity_chunks_count = await pool.fetchval("SELECT COUNT(*) FROM kg_entity_chunks")
        
        # Entity type breakdown
        entity_types = await pool.fetch(
            "SELECT type, COUNT(*) as cnt FROM kg_entities GROUP BY type ORDER BY cnt DESC"
        )
        
        return {
            "entities": entities_count,
            "relations": relations_count,
            "entity_chunk_links": entity_chunks_count,
            "entity_types": {r["type"]: r["cnt"] for r in entity_types},
            "status": "active" if entities_count > 0 else "empty"
        }
    except Exception as e:
        return {
            "entities": 0,
            "relations": 0, 
            "entity_chunk_links": 0,
            "entity_types": {},
            "status": "error",
            "error": str(e)
        }