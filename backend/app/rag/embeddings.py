"""OpenAI embedding generation."""

from __future__ import annotations

from openai import OpenAI
from ..config import settings

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using OpenAI."""
    client = get_client()
    # OpenAI supports up to 2048 texts per batch
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


def embed_query(query: str) -> list[float]:
    """Embed a single query."""
    return embed_texts([query])[0]
