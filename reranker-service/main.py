#!/usr/bin/env python3
"""BGE Reranker v2 M3 Service — Standalone FastAPI service for Mac Studio MPS."""

import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# ─── Global Model State ───────────────────────────────────────────────────────

_model = None
_tokenizer = None
_device = None
_load_time = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup."""
    global _model, _tokenizer, _device, _load_time

    _device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"[reranker] Loading model: {MODEL_NAME}")
    print(f"[reranker] Device: {_device}")
    start = time.time()

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    _model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,  # float32 is stable on MPS for this size
    ).to(_device)
    _model.eval()

    _load_time = time.time() - start
    print(f"[reranker] Model loaded in {_load_time:.1f}s")

    yield

    print("[reranker] Shutting down...")


app = FastAPI(
    title="BGE Reranker v2 M3 Service",
    description="Local reranking using BAAI/bge-reranker-v2-m3 on Mac Studio MPS",
    version="2.0.0",
    lifespan=lifespan,
)


# ─── Request/Response Models ──────────────────────────────────────────────────

class RerankRequest(BaseModel):
    query: str
    documents: list[str]
    instruction: str = ""  # kept for API compat, not used by bge
    top_n: int = 10


class RerankResult(BaseModel):
    index: int
    score: float
    text: str


class RerankResponse(BaseModel):
    results: list[RerankResult]
    model: str
    device: str
    inference_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    load_time_seconds: float


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check."""
    return HealthResponse(
        status="ok" if _model is not None else "loading",
        model=MODEL_NAME,
        device=_device or "unknown",
        load_time_seconds=_load_time or 0,
    )


@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    """Rerank documents based on query relevance."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not request.documents:
        return RerankResponse(
            results=[],
            model=MODEL_NAME,
            device=_device,
            inference_time_ms=0,
        )

    # BGE reranker uses query-document pairs as input
    pairs = [[request.query, doc] for doc in request.documents]

    start = time.time()

    # Process in batches — bge-m3 is small enough for larger batches
    BATCH_SIZE = 16
    scores = []
    for batch_start in range(0, len(pairs), BATCH_SIZE):
        batch_pairs = pairs[batch_start:batch_start + BATCH_SIZE]

        # Tokenize as sentence pairs
        enc = _tokenizer(
            batch_pairs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        enc = {k: v.to(_device) for k, v in enc.items()}

        with torch.no_grad():
            out = _model(**enc)

        # CrossEncoder logits → relevance scores (sigmoid for 0-1 range)
        batch_scores = torch.sigmoid(out.logits.squeeze(-1)).cpu().tolist()
        if isinstance(batch_scores, float):
            batch_scores = [batch_scores]
        scores.extend(batch_scores)

        del enc, out

    inference_time = (time.time() - start) * 1000

    # Build results
    results = [
        RerankResult(index=i, score=round(score, 4), text=doc)
        for i, (doc, score) in enumerate(zip(request.documents, scores))
    ]

    # Sort by score descending
    results.sort(key=lambda x: x.score, reverse=True)

    # Apply top_n
    results = results[:request.top_n]

    return RerankResponse(
        results=results,
        model=MODEL_NAME,
        device=_device,
        inference_time_ms=round(inference_time, 1),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8102)
