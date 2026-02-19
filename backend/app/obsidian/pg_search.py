"""Obsidian search — PostgreSQL + pgvector hybrid search (semantic + BM25 + RRF + reranking).

Replaces ChromaDB-based search with native PostgreSQL pgvector + tsvector.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import asyncpg

from ..rag.embeddings import embed_query
from ..rag import reranker
from . import kg_search

# Connection pool (initialized on first use)
_pool: asyncpg.Pool | None = None

# PostgreSQL connection config
PG_DSN = "postgresql://klara:klara_docs_2026@host.docker.internal:5433/obsidian_rag"

SEARCH_LOG_PATH = Path("/app/data/obsidian_search_log.jsonl")


async def _get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(PG_DSN, min_size=2, max_size=10)
    return _pool


def _log_search(
    query: str,
    results_count: int,
    folder_filter: str | None = None,
    caller: str | None = None,
    method: str = "semantic",
):
    try:
        SEARCH_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "results_count": results_count,
            "folder_filter": folder_filter,
            "caller": caller,
            "method": method,
        }
        with open(SEARCH_LOG_PATH, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass


def get_search_stats(since_hours: int = 24) -> dict[str, Any]:
    if not SEARCH_LOG_PATH.exists():
        return {"total_searches": 0, "queries": [], "callers": {}}

    cutoff = datetime.now().timestamp() - (since_hours * 3600)
    searches, callers, methods = [], {}, {}

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
        "methods": methods,
    }


# ─── Search Functions ─────────────────────────────────────────────────────────


async def search_obsidian_notes(
    query: str,
    limit: int = 10,
    folder_filter: str | None = None,
    collection_name: str = "obsidian_notes",  # ignored, kept for compat
    caller: str | None = None,
) -> list[dict[str, Any]]:
    """Semantic-only search (legacy compat)."""
    query_embedding = embed_query(query)
    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    pool = await _get_pool()

    folder_clause = ""
    params = [embedding_str, limit]
    if folder_filter:
        folder_clause = "AND folder = $3"
        params.append(folder_filter)

    sql = f"""
        SELECT chunk_id, content, context_prefix, file_path, file_name, folder, metadata,
               1 - (embedding <=> $1::vector) AS semantic_score
        FROM obsidian_chunks
        WHERE embedding IS NOT NULL {folder_clause}
        ORDER BY embedding <=> $1::vector
        LIMIT $2
    """

    rows = await pool.fetch(sql, *params)

    results = []
    for r in rows:
        content = r["content"]
        ctx = r["context_prefix"]
        results.append({
            "id": r["chunk_id"],
            "content": f"{ctx}\n\n{content}" if ctx else content,
            "context_prefix": ctx,
            "metadata": {
                "file_path": r["file_path"],
                "file_name": r["file_name"],
                "folder": r["folder"],
                **(json.loads(r["metadata"]) if r["metadata"] else {}),
            },
            "similarity_score": float(r["semantic_score"]),
        })

    _log_search(query, len(results), folder_filter, caller, "semantic")
    return results


async def search_obsidian_hybrid(
    query: str,
    limit: int = 10,
    folder_filter: str | None = None,
    collection_name: str = "obsidian_notes",  # ignored
    caller: str | None = None,
    use_reranker: bool = True,
    instruction: str = "",
    use_graph: bool = True,
) -> list[dict[str, Any]]:
    """Hybrid search: pgvector cosine + tsvector BM25 → RRF fusion → reranking."""
    query_embedding = embed_query(query)
    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    pool = await _get_pool()
    fetch_limit = limit * 3

    # Build folder filter
    folder_clause = ""
    params: list = [embedding_str, query]
    param_idx = 3
    if folder_filter:
        folder_clause = f"AND folder = ${param_idx}"
        params.append(folder_filter)
        param_idx += 1

    sql = f"""
        WITH semantic AS (
            SELECT id, chunk_id, content, context_prefix, file_path, file_name, folder, metadata,
                   1 - (embedding <=> $1::vector) AS semantic_score,
                   ROW_NUMBER() OVER (ORDER BY embedding <=> $1::vector) AS sem_rank
            FROM obsidian_chunks
            WHERE embedding IS NOT NULL {folder_clause}
            ORDER BY embedding <=> $1::vector
            LIMIT {fetch_limit}
        ),
        bm25 AS (
            SELECT id,
                   ts_rank_cd(tsv, plainto_tsquery('simple', $2)) AS bm25_score,
                   ROW_NUMBER() OVER (ORDER BY ts_rank_cd(tsv, plainto_tsquery('simple', $2)) DESC) AS bm25_rank
            FROM obsidian_chunks
            WHERE tsv @@ plainto_tsquery('simple', $2) {folder_clause}
            LIMIT {fetch_limit}
        )
        SELECT s.chunk_id, s.content, s.context_prefix, s.file_path, s.file_name, s.folder, s.metadata,
               s.semantic_score,
               COALESCE(b.bm25_score, 0) AS bm25_score,
               (1.0 / (60 + s.sem_rank)) + (1.0 / (60 + COALESCE(b.bm25_rank, 9999))) AS rrf_score
        FROM semantic s
        LEFT JOIN bm25 b ON b.id = s.id
        ORDER BY (1.0 / (60 + s.sem_rank)) + (1.0 / (60 + COALESCE(b.bm25_rank, 9999))) DESC
        LIMIT {limit * 2}
    """

    rows = await pool.fetch(sql, *params)

    results = []
    for r in rows:
        meta = json.loads(r["metadata"]) if r["metadata"] else {}
        meta.update({
            "file_path": r["file_path"],
            "file_name": r["file_name"],
            "folder": r["folder"],
        })
        content = r["content"]
        ctx = r["context_prefix"]
        display_content = f"{ctx}\n\n{content}" if ctx else content
        results.append({
            "id": r["chunk_id"],
            "content": display_content,
            "text": display_content,  # reranker expects 'text'
            "context_prefix": ctx,
            "metadata": meta,
            "semantic_score": float(r["semantic_score"]),
            "bm25_score": float(r["bm25_score"]),
            "rrf_score": float(r["rrf_score"]),
        })

    # Graph expansion: detect entities, traverse graph, merge related chunks
    graph_info = None
    if use_graph:
        try:
            graph_info = await kg_search.graph_augmented_search(query)
            if graph_info["graph_chunks"]:
                # Merge graph chunks (avoid duplicates by chunk_id)
                existing_ids = {r["id"] for r in results}
                graph_added = 0
                for gc in graph_info["graph_chunks"]:
                    if gc["id"] not in existing_ids:
                        # Give graph chunks a baseline rrf_score so they rank mid-list
                        gc["semantic_score"] = 0.0
                        gc["bm25_score"] = 0.0
                        gc["rrf_score"] = 0.005  # low base, reranker will re-score
                        gc["graph_boosted"] = True
                        results.append(gc)
                        existing_ids.add(gc["id"])
                        graph_added += 1
                if graph_added:
                    print(f"[obsidian-search] Graph expansion added {graph_added} chunks "
                          f"({len(graph_info['matched_entities'])} entities matched, "
                          f"{len(graph_info['related_entities'])} related)")
        except Exception as e:
            print(f"[obsidian-search] Graph expansion failed (non-fatal): {e}")

    # Rerank
    if use_reranker and results:
        if not instruction:
            instruction = "Preferáld a frissebb jegyzeteket és a magyar nyelvű tartalmat."
        try:
            reranked = await reranker.rerank(
                query=query,
                documents=results,
                top_n=limit,
                instruction=instruction,
            )
            if reranked:
                results = reranked
        except Exception:
            results = results[:limit]
    else:
        results = results[:limit]

    # Format
    formatted = []
    for r in results:
        entry = {
            "id": r.get("id", ""),
            "content": r.get("content", r.get("text", "")),
            "metadata": r.get("metadata", {}),
            "score": r.get("score", r.get("rerank_score", r.get("rrf_score", 0))),
            "semantic_score": r.get("semantic_score"),
            "bm25_score": r.get("bm25_score"),
            "rrf_score": r.get("rrf_score"),
            "rerank_score": r.get("rerank_score"),
            "reranker": r.get("reranker"),
        }
        if r.get("graph_boosted"):
            entry["graph_boosted"] = True
        formatted.append(entry)

    method = "hybrid+rerank"
    if graph_info and graph_info["matched_entities"]:
        method = "hybrid+graph+rerank"

    _log_search(query, len(formatted), folder_filter, caller, method)
    return formatted


async def get_obsidian_stats(collection_name: str = "obsidian_notes") -> dict[str, Any]:
    """Get Obsidian chunk statistics from PostgreSQL."""
    try:
        pool = await _get_pool()

        total = await pool.fetchval("SELECT COUNT(*) FROM obsidian_chunks")
        files = await pool.fetchval("SELECT COUNT(DISTINCT file_path) FROM obsidian_chunks")

        folder_rows = await pool.fetch(
            "SELECT folder, COUNT(*) AS cnt FROM obsidian_chunks GROUP BY folder ORDER BY cnt DESC"
        )
        folders = {r["folder"]: r["cnt"] for r in folder_rows}

        return {
            "total_chunks": total,
            "total_files": files,
            "folders": folders,
            "backend": "postgresql+pgvector",
            "collection": collection_name,
        }
    except Exception as e:
        return {
            "error": str(e),
            "total_chunks": 0,
            "total_files": 0,
            "folders": {},
            "backend": "postgresql+pgvector",
        }
