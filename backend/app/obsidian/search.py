"""Obsidian search — Hybrid search (semantic + BM25) with reranking."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
import chromadb

from ..rag.embeddings import embed_query
from ..rag import reranker
from ..config import settings


# Search log file
SEARCH_LOG_PATH = Path("/app/data/obsidian_search_log.jsonl")

# BM25 Index for Obsidian notes
_bm25_index = None
_bm25_documents = []
_bm25_doc_map = {}  # id -> document dict


def _log_search(
    query: str, 
    results_count: int, 
    folder_filter: str | None = None, 
    caller: str | None = None,
    method: str = "semantic"
):
    """Log a search query."""
    SEARCH_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "results_count": results_count,
        "folder_filter": folder_filter,
        "caller": caller,
        "method": method
    }
    with open(SEARCH_LOG_PATH, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


def get_search_stats(since_hours: int = 24) -> dict[str, Any]:
    """Get search statistics from the log."""
    if not SEARCH_LOG_PATH.exists():
        return {"total_searches": 0, "queries": [], "callers": {}}
    
    cutoff = datetime.now().timestamp() - (since_hours * 3600)
    searches = []
    callers = {}
    methods = {}
    
    with open(SEARCH_LOG_PATH) as f:
        for line in f:
            try:
                entry = json.loads(line)
                entry_time = datetime.fromisoformat(entry["timestamp"]).timestamp()
                if entry_time >= cutoff:
                    searches.append(entry)
                    caller = entry.get("caller", "unknown")
                    callers[caller] = callers.get(caller, 0) + 1
                    method = entry.get("method", "semantic")
                    methods[method] = methods.get(method, 0) + 1
            except Exception:
                continue
    
    return {
        "total_searches": len(searches),
        "since_hours": since_hours,
        "queries": [s["query"] for s in searches[-20:]],
        "callers": callers,
        "methods": methods
    }


def get_obsidian_collection(collection_name: str = "obsidian_notes"):
    """Get or create the Obsidian notes collection."""
    client = chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
    )
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


# ─── BM25 Index ───────────────────────────────────────────────────────────────

def _build_bm25_index(collection_name: str = "obsidian_notes"):
    """Build BM25 index from Obsidian collection."""
    global _bm25_index, _bm25_documents, _bm25_doc_map
    
    try:
        from rank_bm25 import BM25Okapi
        
        collection = get_obsidian_collection(collection_name)
        all_docs = collection.get(include=["documents", "metadatas"])
        
        if not all_docs["ids"]:
            print("[obsidian] No documents for BM25 index")
            return
        
        _bm25_documents = []
        _bm25_doc_map = {}
        tokenized = []
        
        for i, doc_id in enumerate(all_docs["ids"]):
            doc_text = all_docs["documents"][i]
            metadata = all_docs["metadatas"][i]
            
            doc_dict = {
                "id": doc_id,
                "text": doc_text,
                "metadata": metadata,
            }
            _bm25_documents.append(doc_dict)
            _bm25_doc_map[doc_id] = doc_dict
            
            # Simple tokenization
            tokens = doc_text.lower().split()
            tokenized.append(tokens)
        
        _bm25_index = BM25Okapi(tokenized)
        print(f"[obsidian] BM25 index built: {len(_bm25_documents)} documents")
        
    except Exception as e:
        print(f"[obsidian] Failed to build BM25 index: {e}")


def invalidate_bm25_index():
    """Invalidate BM25 index (call after ingest)."""
    global _bm25_index
    _bm25_index = None


def _bm25_search(query: str, limit: int = 20) -> list[dict]:
    """Search using BM25."""
    global _bm25_index, _bm25_documents
    
    if _bm25_index is None:
        _build_bm25_index()
    
    if _bm25_index is None or not _bm25_documents:
        return []
    
    tokens = query.lower().split()
    scores = _bm25_index.get_scores(tokens)
    
    # Get top results
    scored_docs = [(score, i) for i, score in enumerate(scores) if score > 0]
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    results = []
    for score, idx in scored_docs[:limit]:
        doc = _bm25_documents[idx].copy()
        doc["bm25_score"] = round(score, 4)
        results.append(doc)
    
    return results


# ─── Semantic Search ──────────────────────────────────────────────────────────

def _semantic_search(
    query: str,
    limit: int = 20,
    folder_filter: str | None = None,
    collection_name: str = "obsidian_notes",
) -> list[dict]:
    """Search using semantic similarity (embeddings)."""
    query_embedding = embed_query(query)
    collection = get_obsidian_collection(collection_name)
    
    # Build where clause
    where_clause = {"source_type": "obsidian"}
    if folder_filter:
        where_clause["folder"] = folder_filter
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=limit,
        where=where_clause,
        include=["documents", "metadatas", "distances"]
    )
    
    formatted = []
    if results["ids"] and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            score = 1 - (distance / 2)  # Convert to 0-1 similarity
            
            formatted.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "semantic_score": round(score, 4),
            })
    
    return formatted


# ─── RRF Fusion ───────────────────────────────────────────────────────────────

def _reciprocal_rank_fusion(
    *result_lists: list[dict],
    k: int = 60,
) -> list[dict]:
    """Reciprocal Rank Fusion to combine ranked result lists."""
    fused_scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}
    
    for results in result_lists:
        for rank, doc in enumerate(results):
            doc_id = doc.get("id", doc.get("text", "")[:100])
            rrf_score = 1.0 / (k + rank + 1)
            
            if doc_id in fused_scores:
                fused_scores[doc_id] += rrf_score
            else:
                fused_scores[doc_id] = rrf_score
                doc_map[doc_id] = doc
    
    # Sort by fused score
    sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
    
    results = []
    for doc_id in sorted_ids:
        doc = doc_map[doc_id].copy()
        doc["rrf_score"] = round(fused_scores[doc_id], 6)
        results.append(doc)
    
    return results


# ─── Main Search Functions ────────────────────────────────────────────────────

def search_obsidian_notes(
    query: str,
    limit: int = 10,
    folder_filter: str | None = None,
    collection_name: str = "obsidian_notes",
    caller: str | None = None
) -> list[dict[str, Any]]:
    """Search Obsidian notes using semantic similarity only (legacy).
    
    For hybrid search with reranking, use search_obsidian_hybrid().
    """
    results = _semantic_search(query, limit, folder_filter, collection_name)
    
    # Format for backwards compatibility
    formatted = []
    for r in results:
        formatted.append({
            "id": r["id"],
            "content": r["text"],
            "metadata": r["metadata"],
            "similarity_score": r["semantic_score"]
        })
    
    try:
        _log_search(query, len(formatted), folder_filter, caller, "semantic")
    except Exception:
        pass
    
    return formatted


async def search_obsidian_hybrid(
    query: str,
    limit: int = 10,
    folder_filter: str | None = None,
    collection_name: str = "obsidian_notes",
    caller: str | None = None,
    use_reranker: bool = True,
    instruction: str = "",
    use_graph: bool = True,
) -> dict[str, Any]:
    """Hybrid search: semantic + BM25 + optional Knowledge Graph → RRF fusion → reranking.
    
    Args:
        query: Search query
        limit: Maximum results to return
        folder_filter: Optional folder filter
        collection_name: ChromaDB collection name
        caller: Who made the search (bob, max, eve)
        use_reranker: Whether to apply reranking (default True)
        instruction: Custom instruction for reranker
        use_graph: Whether to use Knowledge Graph expansion (default True)
        
    Returns:
        dict with "results" list and optional "graph_context"
    """
    # 1. Semantic search
    semantic_results = _semantic_search(
        query, 
        limit=limit * 3,  # Get more for fusion
        folder_filter=folder_filter,
        collection_name=collection_name
    )
    
    # 2. BM25 search
    bm25_results = _bm25_search(query, limit=limit * 3)
    
    # 3. Knowledge Graph expansion (if enabled)
    graph_context = None
    graph_chunk_ids = set()  # chunk IDs that should get a boost
    graph_results = []

    if use_graph:
        try:
            from . import kg_search as obsidian_kg_search

            # 3a. Detect entities in query
            matched_entities = await obsidian_kg_search.detect_entities(
                query, threshold=0.35, limit=5
            )

            if matched_entities:
                entity_ids = [e["id"] for e in matched_entities]

                # 3b. Expand graph (1-hop neighbors)
                related_entities = await obsidian_kg_search.expand_entities(
                    entity_ids, max_related=15
                )

                # 3c. Get chunks linked to matched + related entities
                all_entity_ids = entity_ids + [e["id"] for e in related_entities]
                pool = await obsidian_kg_search._get_pool()
                
                entity_chunk_rows = await pool.fetch(
                    """SELECT DISTINCT c.chunk_id AS id, c.content AS text,
                              c.file_path, c.file_name, c.folder, c.context_prefix,
                              c.metadata
                       FROM kg_entity_chunks ec
                       JOIN obsidian_chunks c ON c.chunk_id = ec.chunk_id
                       WHERE ec.entity_id = ANY($1::int[])
                       LIMIT $2""",
                    all_entity_ids, limit * 2,
                )

                for r in entity_chunk_rows:
                    chunk_id = r["id"]
                    graph_chunk_ids.add(chunk_id)
                    meta = json.loads(r["metadata"]) if r["metadata"] else {}
                    meta.update({
                        "file_path": r["file_path"],
                        "file_name": r["file_name"],
                        "folder": r["folder"],
                    })
                    content = r["text"]
                    ctx = r["context_prefix"]
                    graph_results.append({
                        "id": chunk_id,
                        "text": f"{ctx}\n\n{content}" if ctx else content,
                        "metadata": meta,
                        "graph_boosted": True,
                        # Low base score — reranker will re-score properly
                        "semantic_score": 0.005,
                    })

                graph_context = {
                    "detected_entities": [
                        {"id": e["id"], "name": e["name"], "type": e["type"],
                         "similarity": e.get("similarity", 0)}
                        for e in matched_entities
                    ],
                    "expanded_entities": [
                        {"id": e["id"], "name": e["name"], "type": e["type"],
                         "relation": e.get("relation_type", "")}
                        for e in related_entities
                    ],
                    "graph_chunks_added": len(graph_results),
                }
        except Exception as e:
            # Graph errors should not break search
            print(f"[obsidian] Graph expansion error: {e}")
    
    # 4. RRF fusion (semantic + BM25 + graph chunks)
    fusion_lists = [semantic_results, bm25_results]
    if graph_results:
        fusion_lists.append(graph_results)
    fused = _reciprocal_rank_fusion(*fusion_lists)

    # 4b. Entity boost: if a chunk's ID is in graph_chunk_ids, boost its RRF score
    if graph_chunk_ids:
        GRAPH_BOOST = 0.005  # small additive boost
        for doc in fused:
            if doc.get("id") in graph_chunk_ids:
                doc["rrf_score"] = doc.get("rrf_score", 0) + GRAPH_BOOST
                doc["graph_boosted"] = True
        # Re-sort after boosting
        fused.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)
    
    # 5. Rerank (if enabled and we have results)
    if use_reranker and fused:
        # Prepare documents for reranker
        docs_for_rerank = [
            {"text": doc.get("text", doc.get("content", "")), **doc}
            for doc in fused[:limit * 2]
        ]
        
        # Default instruction for Obsidian context
        if not instruction:
            instruction = "Preferáld a frissebb jegyzeteket és a magyar nyelvű tartalmat."
        
        reranked = await reranker.rerank(
            query=query,
            documents=docs_for_rerank,
            top_n=limit,
            instruction=instruction,
        )
        results = reranked
    else:
        results = fused[:limit]
    
    # Format results
    formatted = []
    graph_boosted_count = 0
    for r in results:
        is_boosted = r.get("graph_boosted", False)
        if is_boosted:
            graph_boosted_count += 1
        formatted.append({
            "id": r.get("id", ""),
            "content": r.get("text", r.get("content", "")),
            "metadata": r.get("metadata", {}),
            "score": r.get("score", r.get("rerank_score", r.get("rrf_score", 0))),
            "semantic_score": r.get("semantic_score"),
            "bm25_score": r.get("bm25_score"),
            "rrf_score": r.get("rrf_score"),
            "rerank_score": r.get("rerank_score"),
            "reranker": r.get("reranker"),
            "graph_boosted": is_boosted,
        })
    
    if graph_context:
        graph_context["graph_boosted_count"] = graph_boosted_count

    method = "hybrid"
    if use_graph and graph_context:
        method += "+graph"
    if use_reranker:
        method += "+rerank"

    try:
        _log_search(query, len(formatted), folder_filter, caller, method)
    except Exception:
        pass
    
    return {"results": formatted, "graph_context": graph_context}


def get_obsidian_stats(collection_name: str = "obsidian_notes") -> dict[str, Any]:
    """Get statistics about the Obsidian notes collection."""
    try:
        collection = get_obsidian_collection(collection_name)
        
        total_results = collection.get()
        total_chunks = len(total_results["ids"]) if total_results["ids"] else 0
        
        if total_chunks == 0:
            return {
                "total_chunks": 0,
                "total_files": 0,
                "folders": {},
                "collection": collection_name,
                "bm25_indexed": False
            }
        
        folder_counts = {}
        file_paths = set()
        
        for metadata in total_results["metadatas"]:
            folder = metadata.get("folder", "unknown")
            folder_counts[folder] = folder_counts.get(folder, 0) + 1
            file_paths.add(metadata.get("file_path", ""))
        
        return {
            "total_chunks": total_chunks,
            "total_files": len(file_paths),
            "folders": folder_counts,
            "collection": collection_name,
            "bm25_indexed": _bm25_index is not None,
            "bm25_doc_count": len(_bm25_documents) if _bm25_documents else 0
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "total_chunks": 0,
            "total_files": 0,
            "folders": {},
            "collection": collection_name,
            "bm25_indexed": False
        }
