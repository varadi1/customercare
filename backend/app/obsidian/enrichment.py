"""Contextual enrichment for Obsidian RAG chunks.

Adds LLM-generated context prefix to each chunk before embedding,
improving retrieval quality by providing document-level context.

Ported from jogszabály RAG enrichment pipeline.
"""

from __future__ import annotations

import os
import time
from typing import Optional

from openai import OpenAI

ENRICHMENT_MODEL = os.environ.get("ENRICHMENT_MODEL", "gpt-4o-mini")
ENRICHMENT_ENABLED = os.environ.get("ENRICHMENT_ENABLED", "true").lower() == "true"
ENRICHMENT_BATCH_SIZE = int(os.environ.get("ENRICHMENT_BATCH_SIZE", "10"))

# Rate limiting: max requests per minute (gpt-4o-mini tier 1 = 500 RPM)
ENRICHMENT_DELAY = float(os.environ.get("ENRICHMENT_DELAY", "0.15"))

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()  # uses OPENAI_API_KEY env
    return _client


CONTEXT_PROMPT = """Te egy Obsidian tudásbázis-elemző rendszer vagy. Az alábbi szövegrészlet egy személyes/munkahelyi tudásbázisból származik.

<fájl>
{file_name}
</fájl>

<mappa>
{folder_type}
</mappa>

<tartalom>
{chunk_text}
</tartalom>

Add meg a szövegrészlet rövid kontextusát (2-3 mondat, magyarul), ami segít megérteni:
1. Milyen típusú dokumentumból származik (jegyzet, projekt leírás, meeting, feladatlista, stb.)
2. Mi a fő témája/kontextusa
3. Milyen személyek, projektek, vagy szervezetek kapcsolódnak hozzá

CSAK a kontextust írd, semmi mást. Ne ismételd a szövegrészletet. Tömör, informatív legyen."""


def enrich_chunk(
    chunk_text: str,
    file_name: str,
    folder_type: str,
    model: Optional[str] = None,
) -> Optional[str]:
    """Generate context prefix for a single chunk.

    Returns the context string, or None on failure.
    """
    client = _get_client()
    truncated = chunk_text[:2000]

    try:
        resp = client.chat.completions.create(
            model=model or ENRICHMENT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": CONTEXT_PROMPT.format(
                        file_name=file_name,
                        folder_type=folder_type,
                        chunk_text=truncated,
                    ),
                }
            ],
            max_tokens=200,
            temperature=0,
        )
        context = resp.choices[0].message.content.strip()
        return context if context else None
    except Exception as e:
        print(f"[obsidian-enrichment] Failed for {file_name}: {e}")
        return None


def enrich_chunks_batch(
    chunks: list[dict],
    file_name: str,
    folder_type: str,
    model: Optional[str] = None,
) -> list[dict]:
    """Enrich a list of chunks with context prefixes.

    Each chunk dict should have 'content' key.
    Returns new list with 'context_prefix' and 'enriched_content' added.
    Chunks that already have 'context_prefix' are skipped.
    """
    if not ENRICHMENT_ENABLED:
        return chunks

    result = []
    enriched_count = 0

    for chunk in chunks:
        if chunk.get("context_prefix"):
            result.append(chunk)
            continue

        context = enrich_chunk(
            chunk_text=chunk["content"],
            file_name=file_name,
            folder_type=folder_type,
            model=model,
        )

        enriched = dict(chunk)
        if context:
            enriched["context_prefix"] = context
            enriched["original_content"] = chunk["content"]
            enriched["enriched_content"] = f"{context}\n\n{chunk['content']}"
            enriched_count += 1
        result.append(enriched)

        time.sleep(ENRICHMENT_DELAY)

    print(f"[obsidian-enrichment] Enriched {enriched_count}/{len(chunks)} chunks for {file_name}")
    return result
