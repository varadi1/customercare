#!/usr/bin/env python3
"""BGE-M3 embedding service — INGEST instance (port 8114).

Dedicated for batch/ingest workloads. Rate-limited to max 4 concurrent requests
so it never starves the search instance (port 8104).

Memory management:
- torch.mps.empty_cache() after every request (frees MPS tensor cache)
- gc.collect() every N requests (frees Python-side fragmentation)
- RSS watchdog: graceful shutdown if memory exceeds limit
"""

import asyncio
import gc
import os
import signal
import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import torch

# ─── Configuration ────────────────────────────────────────────────────────────

MAX_CONCURRENT = 4
GC_EVERY_N = 25            # more frequent GC for ingest (larger batches)
RSS_LIMIT_MB = 5000        # graceful restart above this RSS
RSS_CHECK_EVERY_N = 10     # check more often (ingest = heavier requests)

# ─── App ──────────────────────────────────────────────────────────────────────

_semaphore: asyncio.Semaphore | None = None

app = FastAPI(title="BGE-M3 Ingest Embedding Service")

model = None
_request_count = 0


def _get_rss_mb() -> float:
    """Get current RSS in MB (macOS/Linux)."""
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def _graceful_shutdown():
    """Signal self to restart (launchd KeepAlive will respawn)."""
    print(f"[bge-m3-ingest] RSS {_get_rss_mb():.0f} MB > {RSS_LIMIT_MB} MB limit, restarting...")
    os.kill(os.getpid(), signal.SIGTERM)


@app.on_event("startup")
def load_model():
    global model, _semaphore
    _semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[bge-m3-ingest] Loading BGE-M3 on {device} (max_concurrent={MAX_CONCURRENT})...")

    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("BAAI/bge-m3", device=device)
    model = m
    print(f"[bge-m3-ingest] Loaded, dim={m.get_sentence_embedding_dimension()}, RSS={_get_rss_mb():.0f} MB")


class EmbedRequest(BaseModel):
    texts: list[str]


@app.post("/embed")
async def embed(req: EmbedRequest):
    global _request_count

    if _semaphore is None:
        raise HTTPException(503, "Model not loaded yet")

    if _semaphore._value == 0:
        raise HTTPException(429, f"Too many concurrent requests (max {MAX_CONCURRENT})")

    async with _semaphore:
        _request_count += 1

        # Truncate texts to 6000 chars to prevent OOM/stuck
        texts = [t[:6000] for t in req.texts]
        # Run in thread to not block event loop
        result = await asyncio.to_thread(
            model.encode, texts,
            batch_size=16,  # smaller batch for stability
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if isinstance(result, np.ndarray):
            result = result.tolist()

        # Level 1: free MPS tensor cache every request
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # Level 2: periodic full GC
        if _request_count % GC_EVERY_N == 0:
            gc.collect()

        # Level 3: RSS watchdog — graceful restart if over limit
        if _request_count % RSS_CHECK_EVERY_N == 0:
            rss = _get_rss_mb()
            if rss > RSS_LIMIT_MB:
                threading.Thread(target=_graceful_shutdown, daemon=True).start()

        return {"embeddings": result}


@app.get("/health")
async def health():
    queue = MAX_CONCURRENT - (_semaphore._value if _semaphore else MAX_CONCURRENT)
    return {
        "status": "ok",
        "model": "BAAI/bge-m3",
        "dim": 1024,
        "role": "ingest",
        "port": 8114,
        "max_concurrent": MAX_CONCURRENT,
        "active_requests": queue,
        "requests_served": _request_count,
        "rss_mb": round(_get_rss_mb()),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8114)
