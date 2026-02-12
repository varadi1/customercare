"""Reranker — Local Contextual AI service with Cohere fallback."""

from __future__ import annotations

import httpx
from ..config import settings

COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"
LOCAL_RERANKER_URL = "http://host.docker.internal:8102"  # Native service on Mac

# Status
_local_available = False
_mode = "none"


async def _check_local_reranker() -> bool:
    """Check if local reranker service is available."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{LOCAL_RERANKER_URL}/health")
            if resp.status_code == 200:
                data = resp.json()
                return data.get("status") == "ok"
    except Exception:
        pass
    return False


async def initialize() -> str:
    """Initialize the reranker.
    
    Tries local service first, falls back to Cohere.
    
    Returns:
        String indicating which reranker is active: "local", "cohere", or "none"
    """
    global _local_available, _mode
    
    # Try local reranker service
    if await _check_local_reranker():
        _local_available = True
        _mode = "local"
        print("[hanna] Reranker: Using LOCAL Contextual AI 6B service (:8102)")
        return "local"
    
    # Fall back to Cohere if API key available
    if settings.cohere_api_key:
        _local_available = False
        _mode = "cohere"
        print("[hanna] Reranker: Using COHERE API (local service not available)")
        return "cohere"
    
    _mode = "none"
    print("[hanna] Reranker: NONE available (local down, no Cohere API key)")
    return "none"


async def rerank(
    query: str,
    documents: list[dict],
    top_n: int | None = None,
    instruction: str = "",
) -> list[dict]:
    """Rerank search results.
    
    Uses local Contextual AI service if available, falls back to Cohere API.
    
    Args:
        query: The search query
        documents: List of dicts with at least 'text' key
        top_n: Number of top results to return (default: settings.rerank_top_k)
        instruction: Optional instruction for local reranker (ignored by Cohere)

    Returns:
        Reranked list of document dicts with updated scores
    """
    if not documents:
        return []
    
    top_n = top_n or settings.rerank_top_k
    
    # Try local reranker service first
    if _local_available:
        try:
            result = await _rerank_local(query, documents, top_n, instruction)
            if result is not None:
                return result
        except Exception as e:
            print(f"[hanna] Local rerank failed, trying Cohere: {e}")
    
    # Cohere fallback
    return await _rerank_cohere(query, documents, top_n)


async def _rerank_local(
    query: str,
    documents: list[dict],
    top_n: int,
    instruction: str = "",
) -> list[dict] | None:
    """Rerank using local Contextual AI service."""
    doc_texts = [d["text"] for d in documents]
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{LOCAL_RERANKER_URL}/rerank",
                json={
                    "query": query,
                    "documents": doc_texts,
                    "instruction": instruction,
                    "top_n": min(top_n, len(documents)),
                },
            )
            resp.raise_for_status()
            data = resp.json()
        
        # Map reranked results back to original documents
        reranked = []
        for item in data.get("results", []):
            idx = item["index"]
            doc = documents[idx].copy()
            doc["score"] = round(item["score"], 4)
            doc["rerank_score"] = round(item["score"], 4)
            doc["reranker"] = "contextual-ai-6b"
            reranked.append(doc)
        
        return reranked
        
    except Exception as e:
        print(f"[hanna] Local reranker error: {e}")
        return None


async def _rerank_cohere(
    query: str,
    documents: list[dict],
    top_n: int,
) -> list[dict]:
    """Rerank using Cohere API (fallback)."""
    if not settings.cohere_api_key:
        print("[hanna] No Cohere API key, returning unranked results")
        return documents[:top_n]

    doc_texts = [d["text"] for d in documents]

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                COHERE_RERANK_URL,
                headers={
                    "Authorization": f"Bearer {settings.cohere_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.rerank_model,
                    "query": query,
                    "documents": doc_texts,
                    "top_n": min(top_n, len(documents)),
                },
            )
            resp.raise_for_status()
            data = resp.json()

        reranked = []
        for item in data.get("results", []):
            idx = item["index"]
            doc = documents[idx].copy()
            doc["score"] = round(item["relevance_score"], 4)
            doc["rerank_score"] = round(item["relevance_score"], 4)
            doc["reranker"] = "cohere"
            reranked.append(doc)

        return reranked

    except Exception as e:
        print(f"[hanna] Cohere rerank failed, returning unranked: {e}")
        return documents[:top_n]


def rerank_sync(
    query: str,
    documents: list[dict],
    top_n: int | None = None,
    instruction: str = "",
) -> list[dict]:
    """Synchronous version of rerank (for non-async contexts)."""
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, rerank(query, documents, top_n, instruction))
            return future.result()
    else:
        return asyncio.run(rerank(query, documents, top_n, instruction))


def get_status() -> dict:
    """Get reranker status."""
    return {
        "mode": _mode,
        "local_available": _local_available,
        "local_url": LOCAL_RERANKER_URL,
        "cohere_configured": bool(settings.cohere_api_key),
    }
