"""Hybrid search: semantic (PostgreSQL+pgvector) + BM25 keyword → RRF → Cohere rerank.

Migrated from ChromaDB to PostgreSQL+pgvector for OETP knowledge base.
Database: postgresql://klara:klara_docs_2026@host.docker.internal:5433/hanna_oetp
"""

from __future__ import annotations

import asyncpg
from typing import Optional
from ..config import settings
from .embeddings import embed_query
from .reranker import rerank, get_status as _get_reranker_status
from .authority import apply_authority_weighting


def _get_reranker_mode() -> str:
    """Get actual reranker mode from the reranker module."""
    try:
        return _get_reranker_status().get("mode", "unknown")
    except Exception:
        return "unknown"

from .query_expansion import expand_query, expand_query_async
# references imported lazily in main.py to avoid circular import

# PostgreSQL connection pool
_pool: Optional[asyncpg.Pool] = None
PG_DSN = "postgresql://klara:klara_docs_2026@host.docker.internal:5433/hanna_oetp"


async def _get_pool() -> asyncpg.Pool:
    """Get or create the PostgreSQL connection pool."""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(PG_DSN, min_size=2, max_size=10)
    return _pool


async def _semantic_search(
    query: str,
    top_k: int,
    category: str | None = None,
    chunk_type: str | None = None,
    only_valid: bool = True,
) -> list[dict]:
    """Pure semantic (embedding) search via PostgreSQL+pgvector."""
    pool = await _get_pool()
    
    # Get query embedding
    query_embedding = embed_query(query)
    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
    
    # Build WHERE clauses
    where_clauses = ["embedding IS NOT NULL"]
    params = [embedding_str]
    param_idx = 2
    
    if category:
        where_clauses.append(f"program = ${param_idx}")
        params.append(category)
        param_idx += 1
    
    if chunk_type:
        where_clauses.append(f"doc_type = ${param_idx}")
        params.append(chunk_type)
        param_idx += 1
    
    # only_valid filter - assume there's a valid_to field or similar
    # For now, skip this filter since the schema doesn't clearly specify
    
    params.append(top_k)
    limit_param = f"${param_idx}"
    
    where_sql = " AND ".join(where_clauses)
    
    sql = f"""
        SELECT id, doc_id, doc_type, program, title, content, content_enriched,
               metadata, authority_score, source_date,
               1 - (embedding <=> $1::vector) AS semantic_score
        FROM chunks
        WHERE {where_sql}
        ORDER BY embedding <=> $1::vector
        LIMIT {limit_param}
    """
    
    try:
        rows = await pool.fetch(sql, *params)
    except Exception as e:
        print(f"[hanna-oetp] Semantic search failed: {e}")
        return []
    
    results = []
    for r in rows:
        score = float(r["semantic_score"])
        # Parse metadata if it's JSON string
        metadata = {}
        if r["metadata"]:
            try:
                import json
                metadata = json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"]
            except (json.JSONDecodeError, TypeError):
                metadata = {}
        
        # Map PostgreSQL fields to expected format
        results.append({
            "id": r["id"],
            "text": r["content_enriched"] or r["content"],
            "source": r["doc_id"],
            "category": r["program"],
            "chunk_type": r["doc_type"],
            "score": round(score, 4),
            "metadata": {
                **metadata,
                "title": r["title"],
                "authority_score": float(r["authority_score"]) if r["authority_score"] else 0.5,
                "source_date": r["source_date"].isoformat() if r["source_date"] else None,
            },
            "semantic_score": round(score, 4),
        })
    
    return results


async def _bm25_search_pg(
    query: str,
    top_k: int,
    category: str | None = None,
    chunk_type: str | None = None,
) -> list[dict]:
    """BM25-style full-text search via PostgreSQL tsvector."""
    pool = await _get_pool()
    
    # Build WHERE clauses
    where_clauses = ["content_tsvector @@ plainto_tsquery('hungarian', $1)"]
    params = [query]
    param_idx = 2
    
    if category:
        where_clauses.append(f"program = ${param_idx}")
        params.append(category)
        param_idx += 1
    
    if chunk_type:
        where_clauses.append(f"doc_type = ${param_idx}")
        params.append(chunk_type)
        param_idx += 1
    
    params.append(top_k)
    limit_param = f"${param_idx}"
    
    where_sql = " AND ".join(where_clauses)
    
    sql = f"""
        SELECT id, doc_id, doc_type, program, title, content, content_enriched,
               metadata, authority_score, source_date,
               ts_rank(content_tsvector, plainto_tsquery('hungarian', $1)) AS bm25_score
        FROM chunks
        WHERE {where_sql}
        ORDER BY ts_rank(content_tsvector, plainto_tsquery('hungarian', $1)) DESC
        LIMIT {limit_param}
    """
    
    try:
        rows = await pool.fetch(sql, *params)
    except Exception as e:
        print(f"[hanna-oetp] BM25 search failed: {e}")
        return []
    
    results = []
    for r in rows:
        score = float(r["bm25_score"])
        # Parse metadata if it's JSON string
        metadata = {}
        if r["metadata"]:
            try:
                import json
                metadata = json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"]
            except (json.JSONDecodeError, TypeError):
                metadata = {}
        
        results.append({
            "id": r["id"],
            "text": r["content_enriched"] or r["content"],
            "source": r["doc_id"],
            "category": r["program"],
            "chunk_type": r["doc_type"],
            "score": round(score, 4),
            "metadata": {
                **metadata,
                "title": r["title"],
                "authority_score": float(r["authority_score"]) if r["authority_score"] else 0.5,
                "source_date": r["source_date"].isoformat() if r["source_date"] else None,
            },
            "bm25_score": round(score, 4),
        })
    
    return results


def _reciprocal_rank_fusion(
    *result_lists: list[dict],
    k: int = 60,
) -> list[dict]:
    """Reciprocal Rank Fusion to combine multiple ranked result lists.

    RRF score = sum(1 / (k + rank_i)) for each list where the doc appears.
    Uses 'id' field as identity key (deduplication).
    """
    fused_scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            # Use ID as dedup key
            key = doc.get("id", "")
            if not key:
                continue
            
            rrf_score = 1.0 / (k + rank + 1)

            if key in fused_scores:
                fused_scores[key] += rrf_score
            else:
                fused_scores[key] = rrf_score
                doc_map[key] = doc

    # Sort by fused score
    sorted_keys = sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True)

    output = []
    for key in sorted_keys:
        doc = doc_map[key].copy()
        doc["rrf_score"] = round(fused_scores[key], 6)
        doc["score"] = round(fused_scores[key], 6)
        output.append(doc)

    return output


def search(
    query: str,
    top_k: int | None = None,
    category: str | None = None,
    chunk_type: str | None = None,
    only_valid: bool = True,
) -> list[dict]:
    """Hybrid search pipeline (SYNC wrapper for async implementation):

    1. Semantic search (PostgreSQL+pgvector) → top N candidates
    2. BM25 keyword search (PostgreSQL tsvector) → top N candidates
    3. Reciprocal Rank Fusion → merged top N
    4. (If Cohere API key set) Rerank → final top K

    Falls back gracefully at each stage if components unavailable.
    """
    import asyncio
    
    # Create new event loop if none exists
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(search_async(
        query=query,
        top_k=top_k,
        category=category,
        chunk_type=chunk_type,
        only_valid=only_valid,
    ))


async def search_async(
    query: str,
    top_k: int | None = None,
    category: str | None = None,
    chunk_type: str | None = None,
    only_valid: bool = True,
) -> list[dict]:
    """Async hybrid search pipeline:
    
    1. Query expansion (gpt-4o-mini) → 2-3 query variants
    2. Semantic + BM25 search for EACH variant
    3. RRF fusion across all results
    4. Rerank (local BGE or Cohere fallback)
    5. Authority weighting
    """
    retrieval_k = settings.search_top_k
    final_k = top_k or settings.rerank_top_k

    # Stage 0: Query expansion
    queries = await expand_query_async(query)
    print(f"[hanna-oetp] Query expansion: {queries}")

    # Stage 1+2: Search for each expanded query
    all_semantic = []
    all_bm25 = []

    for q in queries:
        semantic_results = await _semantic_search(
            query=q,
            top_k=retrieval_k,
            category=category,
            chunk_type=chunk_type,
            only_valid=only_valid,
        )
        all_semantic.extend(semantic_results)

        # Use PostgreSQL BM25 instead of external BM25Index
        try:
            bm25_results = await _bm25_search_pg(
                query=q,
                top_k=retrieval_k,
                category=category,
                chunk_type=chunk_type,
            )
            all_bm25.extend(bm25_results)
        except Exception as e:
            print(f"[hanna-oetp] BM25 search failed for '{q}': {e}")

    # Stage 2.5: Knowledge Graph search (entity-based expansion)
    kg_results = []
    try:
        from .kg_search import kg_search
        kg_results = await kg_search(query, top_k=retrieval_k)  # Use original query, not expanded
    except Exception as e:
        print(f"[hanna-oetp] KG search failed: {e}")

    # Stage 3: RRF fusion across all results (semantic + BM25 + KG)
    result_lists = [all_semantic]
    if all_bm25:
        result_lists.append(all_bm25)
    if kg_results:
        result_lists.append(kg_results)
    
    fused = _reciprocal_rank_fusion(*result_lists) if result_lists else []

    candidates = fused[:retrieval_k]

    # Stage 3.5: Priority injection — ensure official docs get to reranker
    candidates = _inject_priority_chunks(candidates, fused)

    # Stage 4: Rerank with the ORIGINAL query (not expanded)
    if candidates:
        try:
            reranked = await rerank(query, candidates, top_n=final_k)
            # Stage 5: Authority weighting (uses authority_score from metadata)
            return apply_authority_weighting(reranked)
        except Exception as e:
            print(f"[hanna-oetp] Rerank failed: {e}")
            return apply_authority_weighting(candidates[:final_k])

    return candidates[:final_k]


async def _get_collection_stats_async() -> dict:
    """Get stats about the OETP knowledge base (PostgreSQL version)."""
    try:
        pool = await _get_pool()
        
        total = await pool.fetchval("SELECT COUNT(*) FROM chunks")
        programs = await pool.fetch(
            "SELECT program, COUNT(*) as cnt FROM chunks GROUP BY program ORDER BY cnt DESC"
        )
        doc_types = await pool.fetch(
            "SELECT doc_type, COUNT(*) as cnt FROM chunks GROUP BY doc_type ORDER BY cnt DESC"
        )
        
        # Get KG stats
        kg_stats = {}
        try:
            from .kg_search import get_kg_stats
            kg_stats = await get_kg_stats()
        except Exception as e:
            kg_stats = {"error": str(e)}
        
        return {
            "total_chunks": total,
            "storage": "postgresql+pgvector",
            "database": "hanna_oetp",
            "programs": {r["program"]: r["cnt"] for r in programs},
            "doc_types": {r["doc_type"]: r["cnt"] for r in doc_types},
            "reranker": _get_reranker_mode(),
            "knowledge_graph": kg_stats,
        }
    except Exception as e:
        return {"error": str(e)}


def get_collection_stats() -> dict:
    """Sync wrapper for _get_collection_stats_async()."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(_get_collection_stats_async())


def _inject_priority_chunks(candidates: list[dict], all_fused: list[dict]) -> list[dict]:
    """Ensure official documents get a chance at the reranker stage.
    
    Problem: When many email chunks contain common keywords, they dominate 
    both semantic and BM25 retrieval, pushing out official document chunks 
    from the top-N candidates.
    
    Solution: If the top-N candidates don't include any priority chunk types,
    scan the full fused results and inject the best priority chunks into
    the candidate list (replacing the weakest email_reply candidates).
    """
    from .authority import PRIORITY_CHUNK_TYPES
    
    # Check if candidates already have priority chunks
    has_priority = any(
        c.get("chunk_type", "") in PRIORITY_CHUNK_TYPES 
        for c in candidates
    )
    if has_priority:
        return candidates
    
    # Find priority chunks in the full fused results (beyond top-N)
    priority_from_fused = [
        r for r in all_fused 
        if r.get("chunk_type", "") in PRIORITY_CHUNK_TYPES
    ]
    
    if not priority_from_fused:
        return candidates
    
    # Inject up to 3 priority chunks, replacing the weakest non-priority candidates
    inject_count = min(3, len(priority_from_fused))
    injected = 0
    
    for priority_chunk in priority_from_fused[:inject_count]:
        # Don't inject if already in candidates (by ID)
        pid = priority_chunk.get("id", "")
        if any(c.get("id") == pid for c in candidates):
            continue
        
        # Replace the weakest non-priority candidate from the end
        for j in range(len(candidates) - 1, -1, -1):
            if candidates[j].get("chunk_type", "") not in PRIORITY_CHUNK_TYPES:
                candidates[j] = priority_chunk
                injected += 1
                break
    
    if injected > 0:
        print(f"[hanna-oetp] Injected {injected} priority chunks into reranker candidates")
    
    return candidates


async def invalidate_chunks(chunk_ids: list[str], reason: str = "") -> dict:
    """Mark chunks as invalid in PostgreSQL.
    
    This sets a valid_to date or similar field.
    Returns count of updated chunks.
    """
    from datetime import date
    
    pool = await _get_pool()
    today = date.today().isoformat()
    
    updated = 0
    errors = []
    
    # Assuming there's a valid_to field - adapt as needed based on actual schema
    for chunk_id in chunk_ids:
        try:
            result = await pool.execute(
                "UPDATE chunks SET metadata = metadata || $1 WHERE id = $2",
                {"valid_to": today, "invalidation_reason": reason},
                chunk_id
            )
            if result == "UPDATE 1":
                updated += 1
            else:
                errors.append(f"{chunk_id}: not found")
        except Exception as e:
            errors.append(f"{chunk_id}: {str(e)}")
    
    return {"updated": updated, "errors": errors}


async def find_chunks_by_text(search_text: str, limit: int = 50) -> list[dict]:
    """Find chunks containing specific text (for invalidation purposes).
    
    Returns list of {id, text_preview, metadata}.
    """
    pool = await _get_pool()
    
    try:
        rows = await pool.fetch(
            """SELECT id, content, metadata, title, doc_type, program
               FROM chunks 
               WHERE content ILIKE $1 
               LIMIT $2""",
            f"%{search_text}%",
            limit
        )
        
        matches = []
        for r in rows:
            matches.append({
                "id": r["id"],
                "text_preview": r["content"][:300] + "..." if len(r["content"]) > 300 else r["content"],
                "metadata": {
                    **(r["metadata"] if r["metadata"] else {}),
                    "title": r["title"],
                    "doc_type": r["doc_type"],
                    "program": r["program"],
                },
            })
        
        return matches
    except Exception as e:
        print(f"[hanna-oetp] Text search failed: {e}")
        return []


# Sync wrappers for compatibility
def invalidate_chunks_sync(chunk_ids: list[str], reason: str = "") -> dict:
    """Sync wrapper for invalidate_chunks."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(invalidate_chunks(chunk_ids, reason))


def find_chunks_by_text_sync(search_text: str, limit: int = 50) -> list[dict]:
    """Sync wrapper for find_chunks_by_text."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(find_chunks_by_text(search_text, limit))