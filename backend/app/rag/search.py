"""Hybrid search: semantic (ChromaDB) + BM25 keyword → RRF → Cohere rerank."""

from __future__ import annotations

import chromadb
from typing import Optional
from ..config import settings
from .embeddings import embed_query
from .bm25 import BM25Index
from .reranker import rerank
from .authority import apply_authority_weighting
from .query_expansion import expand_query, expand_query_async

_chroma: Optional[chromadb.HttpClient] = None


def get_chroma() -> chromadb.HttpClient:
    global _chroma
    if _chroma is None:
        _chroma = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )
    return _chroma


def get_collection():
    """Get or create the main knowledge collection."""
    client = get_chroma()
    return client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )


def _semantic_search(
    query: str,
    top_k: int,
    category: str | None = None,
    chunk_type: str | None = None,
    only_valid: bool = True,
) -> list[dict]:
    """Pure semantic (embedding) search via ChromaDB."""
    collection = get_collection()

    if collection.count() == 0:
        return []

    query_embedding = embed_query(query)

    # Build where clause
    where_clauses = []
    if category:
        where_clauses.append({"category": category})
    if chunk_type:
        where_clauses.append({"chunk_type": chunk_type})
    if only_valid:
        where_clauses.append({"valid_to": ""})

    where = None
    if len(where_clauses) == 1:
        where = where_clauses[0]
    elif len(where_clauses) > 1:
        where = {"$and": where_clauses}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        where=where if where_clauses else None,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    if results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results["distances"] else 0
            chunk_id = results["ids"][0][i] if results.get("ids") else ""
            score = 1 - (distance / 2)

            output.append({
                "id": chunk_id,
                "text": doc,
                "source": meta.get("source", ""),
                "category": meta.get("category", ""),
                "chunk_type": meta.get("chunk_type", ""),
                "score": round(score, 4),
                "metadata": meta,
                "semantic_score": round(score, 4),
            })

    return output


def _reciprocal_rank_fusion(
    *result_lists: list[dict],
    k: int = 60,
) -> list[dict]:
    """Reciprocal Rank Fusion to combine multiple ranked result lists.

    RRF score = sum(1 / (k + rank_i)) for each list where the doc appears.
    Uses 'text' field as identity key (deduplication).
    """
    fused_scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            # Use text content as dedup key (could use chunk ID if available)
            key = doc["text"][:200]  # First 200 chars as fingerprint
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
    """Hybrid search pipeline:

    1. Semantic search (ChromaDB embeddings) → top N candidates
    2. BM25 keyword search → top N candidates
    3. Reciprocal Rank Fusion → merged top N
    4. (If Cohere API key set) Rerank → final top K

    Falls back gracefully at each stage if components unavailable.
    """
    retrieval_k = settings.search_top_k  # How many to retrieve per method
    final_k = top_k or settings.rerank_top_k  # Final results to return

    # Stage 0: Query expansion
    queries = expand_query(query)
    print(f"[hanna] Query expansion (sync): {queries}")

    # Stage 1+2: Search for each expanded query
    all_semantic = []
    all_bm25 = []

    for q in queries:
        semantic_results = _semantic_search(
            query=q,
            top_k=retrieval_k,
            category=category,
            chunk_type=chunk_type,
            only_valid=only_valid,
        )
        all_semantic.extend(semantic_results)

        try:
            bm25_index = BM25Index.get()
            bm25_results = bm25_index.search(q, top_k=retrieval_k)
            all_bm25.extend(bm25_results)
        except Exception as e:
            print(f"[hanna] BM25 search failed: {e}")

    # Stage 3: RRF fusion
    if all_bm25:
        fused = _reciprocal_rank_fusion(all_semantic, all_bm25)
    else:
        fused = _reciprocal_rank_fusion(all_semantic) if all_semantic else []

    candidates = fused[:retrieval_k]

    # Stage 4: Rerank with ORIGINAL query
    if candidates:
        try:
            from .reranker import rerank_sync
            reranked = rerank_sync(query, candidates, top_n=final_k)
            # Stage 5: Authority weighting
            return apply_authority_weighting(reranked)
        except Exception as e:
            print(f"[hanna] Rerank failed, using RRF order: {e}")
            return apply_authority_weighting(candidates[:final_k])

    return candidates[:final_k]


async def search_async(
    query: str,
    top_k: int | None = None,
    category: str | None = None,
    chunk_type: str | None = None,
    only_valid: bool = True,
) -> list[dict]:
    """Async version of search with query expansion.
    
    Pipeline:
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
    print(f"[hanna] Query expansion: {queries}")

    # Stage 1+2: Search for each expanded query
    all_semantic = []
    all_bm25 = []

    for q in queries:
        semantic_results = _semantic_search(
            query=q,
            top_k=retrieval_k,
            category=category,
            chunk_type=chunk_type,
            only_valid=only_valid,
        )
        all_semantic.extend(semantic_results)

        try:
            bm25_index = BM25Index.get()
            bm25_results = bm25_index.search(q, top_k=retrieval_k)
            all_bm25.extend(bm25_results)
        except Exception as e:
            print(f"[hanna] BM25 search failed for '{q}': {e}")

    # Stage 3: RRF fusion across all results
    if all_bm25:
        fused = _reciprocal_rank_fusion(all_semantic, all_bm25)
    else:
        fused = _reciprocal_rank_fusion(all_semantic) if all_semantic else []

    candidates = fused[:retrieval_k]

    # Stage 4: Rerank with the ORIGINAL query (not expanded)
    if candidates:
        try:
            reranked = await rerank(query, candidates, top_n=final_k)
            # Stage 5: Authority weighting
            return apply_authority_weighting(reranked)
        except Exception as e:
            print(f"[hanna] Rerank failed: {e}")
            return apply_authority_weighting(candidates[:final_k])

    return candidates[:final_k]


def get_collection_stats() -> dict:
    """Get stats about the knowledge base."""
    try:
        collection = get_collection()
        bm25_index = BM25Index.get()
        return {
            "total_chunks": collection.count(),
            "collection_name": settings.chroma_collection,
            "bm25_indexed": len(bm25_index._docs) if not bm25_index._dirty else "stale",
            "reranker": "cohere" if settings.cohere_api_key else "disabled",
        }
    except Exception as e:
        return {"error": str(e)}


def invalidate_chunks(chunk_ids: list[str], reason: str = "") -> dict:
    """Mark chunks as invalid by setting valid_to to today's date.
    
    This doesn't delete chunks - they remain searchable with only_valid=False.
    Returns count of updated chunks.
    """
    from datetime import date
    
    collection = get_collection()
    today = date.today().isoformat()
    
    updated = 0
    errors = []
    
    for chunk_id in chunk_ids:
        try:
            # Get existing chunk
            result = collection.get(ids=[chunk_id], include=["metadatas"])
            if not result["ids"]:
                errors.append(f"{chunk_id}: not found")
                continue
            
            # Update metadata
            existing_meta = result["metadatas"][0]
            existing_meta["valid_to"] = today
            existing_meta["invalidation_reason"] = reason
            
            collection.update(
                ids=[chunk_id],
                metadatas=[existing_meta],
            )
            updated += 1
        except Exception as e:
            errors.append(f"{chunk_id}: {str(e)}")
    
    return {"updated": updated, "errors": errors}


def find_chunks_by_text(search_text: str, limit: int = 50) -> list[dict]:
    """Find chunks containing specific text (for invalidation purposes).
    
    Returns list of {id, text_preview, metadata}.
    """
    collection = get_collection()
    
    # ChromaDB doesn't support full-text search, so we use semantic search
    # and then filter results that contain the search text
    query_embedding = embed_query(search_text)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=limit * 3,  # Fetch more to filter
        include=["documents", "metadatas"],
    )
    
    matches = []
    search_lower = search_text.lower()
    
    for i, doc in enumerate(results["documents"][0]):
        if search_lower in doc.lower():
            matches.append({
                "id": results["ids"][0][i],
                "text_preview": doc[:300] + "..." if len(doc) > 300 else doc,
                "metadata": results["metadatas"][0][i],
            })
            if len(matches) >= limit:
                break
    
    return matches
