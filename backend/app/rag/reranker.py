"""Reranker — Local-only reranker service (no Cohere fallback)."""

from __future__ import annotations

import httpx
from ..config import settings

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
    """Initialize the reranker (local only, no Cohere fallback).

    Returns:
        String indicating which reranker is active: "local" or "none"
    """
    global _local_available, _mode

    if await _check_local_reranker():
        _local_available = True
        _mode = "local"
        print("[hanna] Reranker: Using LOCAL BGE reranker v2-m3 service (:8102)")
        return "local"

    _mode = "none"
    print("[hanna] Reranker: Local service not available — returning unranked results (NO Cohere fallback)")
    return "none"


async def rerank(
    query: str,
    documents: list[dict],
    top_n: int | None = None,
    instruction: str = "",
) -> list[dict]:
    """Rerank search results using local service only.

    If local reranker is unavailable, returns documents unranked (no Cohere fallback).
    """
    if not documents:
        return []

    top_n = top_n or settings.rerank_top_k

    if not _local_available:
        # Re-check in case the service came back up
        if await _check_local_reranker():
            global _mode
            _local_available_update = True
            globals()['_local_available'] = True
            _mode = "local"
            print("[hanna] Reranker: Local service recovered (:8102)")
        else:
            return documents[:top_n]

    result = await _rerank_local(query, documents, top_n, instruction)
    if result is not None:
        return result

    print("[hanna] Local rerank failed, returning unranked results")
    return documents[:top_n]


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

        reranked = []
        for item in data.get("results", []):
            idx = item["index"]
            doc = documents[idx].copy()
            doc["score"] = round(item["score"], 4)
            doc["rerank_score"] = round(item["score"], 4)
            doc["reranker"] = "bge-reranker-v2-m3"
            reranked.append(doc)

        return reranked

    except Exception as e:
        print(f"[hanna] Local reranker error: {e}")
        return None


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
        "cohere_configured": False,
    }
