#!/usr/bin/env python3
"""BGE-M3 embedding service - optimized for Apple Silicon.

Memory management:
- torch.mps.empty_cache() after every request (frees MPS tensor cache)
- gc.collect() every N requests (frees Python-side fragmentation)
- RSS watchdog: graceful shutdown if memory exceeds limit
"""

import gc
import os
import signal
import threading

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import torch

# ─── Configuration ────────────────────────────────────────────────────────────

GC_EVERY_N = 50            # full gc.collect() every N requests
RSS_LIMIT_MB = 5000        # graceful restart above this RSS
RSS_CHECK_EVERY_N = 20     # check RSS every N requests

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="BGE-M3 Embedding Service")

model = None
_request_count = 0


def _get_rss_mb() -> float:
    """Get current RSS in MB (macOS/Linux)."""
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def _graceful_shutdown():
    """Signal self to restart (launchd KeepAlive will respawn)."""
    print(f"[bge-m3-search] RSS {_get_rss_mb():.0f} MB > {RSS_LIMIT_MB} MB limit, restarting...")
    os.kill(os.getpid(), signal.SIGTERM)


@app.on_event("startup")
def load_model():
    global model
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[bge-m3-search] Loading BGE-M3 on {device}...")

    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("BAAI/bge-m3", device=device)
    model = m
    print(f"[bge-m3-search] Loaded, dim={m.get_sentence_embedding_dimension()}, RSS={_get_rss_mb():.0f} MB")


class EmbedRequest(BaseModel):
    texts: list[str]


@app.post("/embed")
async def embed(req: EmbedRequest):
    global _request_count
    _request_count += 1

    result = model.encode(req.texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
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
            # Finish this request, then die (launchd restarts us)
            threading.Thread(target=_graceful_shutdown, daemon=True).start()

    return {"embeddings": result}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "BAAI/bge-m3",
        "dim": 1024,
        "requests_served": _request_count,
        "rss_mb": round(_get_rss_mb()),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8104)
