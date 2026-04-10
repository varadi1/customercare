"""Embedding generation — BGE-M3 (local) with OpenAI fallback."""

from __future__ import annotations

import httpx
from ..config import settings

_openai_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=settings.openai_api_key)
    return _openai_client


def _embed_bge_m3(texts: list[str]) -> list[list[float]]:
    """Embed via local BGE-M3 service."""
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(
            f"{settings.bge_m3_url}/embed",
            json={"texts": texts},
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]


def _embed_openai(texts: list[str]) -> list[list[float]]:
    """Embed via OpenAI API (fallback)."""
    client = _get_openai_client()
    all_embeddings = []
    batch_size = 512
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            model=settings.embedding_model,
            input=batch,
            dimensions=settings.embedding_dimensions,
        )
        all_embeddings.extend([d.embedding for d in response.data])
    return all_embeddings


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using configured backend."""
    if settings.embedding_backend == "bge-m3":
        try:
            return _embed_bge_m3(texts)
        except Exception as e:
            print(f"[cc] BGE-M3 embedding failed: {e}")
            if settings.openai_api_key:
                print("[cc] Falling back to OpenAI embeddings")
                return _embed_openai(texts)
            raise
    return _embed_openai(texts)


def embed_query(query: str) -> list[float]:
    """Embed a single query."""
    return embed_texts([query])[0]


def _embed_bge_m3_ingest(texts: list[str]) -> list[list[float]]:
    """Embed via local BGE-M3 INGEST instance (port 8114)."""
    with httpx.Client(timeout=300.0) as client:
        resp = client.post(
            f"{settings.bge_m3_ingest_url}/embed",
            json={"texts": texts},
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]


def embed_texts_ingest(texts: list[str]) -> list[list[float]]:
    """Embed texts using the dedicated ingest instance (port 8114).
    Falls back to search instance if ingest is unavailable."""
    try:
        return _embed_bge_m3_ingest(texts)
    except Exception as e:
        print(f"[cc] BGE-M3 ingest instance failed: {e}, falling back to search instance")
        return embed_texts(texts)
