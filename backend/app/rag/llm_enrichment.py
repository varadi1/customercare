"""LLM-based contextual enrichment for Hanna RAG chunks.

Uses GPT-4o-mini to generate context prefixes per chunk, then re-embeds
with BGE-M3 for better retrieval quality.

Two modes:
1. Batch mode (OpenAI Batch API) — 50% cheaper, for bulk enrichment
2. Inline mode — for new chunks at ingest time
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE = "https://api.openai.com/v1"
MODEL = "gpt-4o-mini"

# System prompt for context generation
SYSTEM_PROMPT = """Rövid (1-2 mondatos) kontextus leírást generálsz egy szövegrészlethez (chunk-hoz).
A kontextus leírásnak tartalmaznia kell:
- Milyen típusú dokumentumból származik (pályázati felhívás, email válasz, GYIK, útmutató, stb.)
- Mi a chunk fő témája
- Ha email: milyen kérdésre válaszol

Formátum: egyetlen rövid bekezdés, magyarul. NE idézd a szöveget, csak a kontextust írd le.
Példa: "Ez egy ügyfélszolgálati email válasz, amely a napelemes rendszer méretezési követelményeire vonatkozó kérdést válaszolja meg az OETP pályázat kapcsán."
"""


def _build_user_prompt(text: str, source: str, chunk_type: str, category: str) -> str:
    """Build the user prompt for context generation."""
    # Truncate to ~2000 chars to save tokens
    truncated = text[:2000]
    return (
        f"Forrás: {source}\n"
        f"Típus: {chunk_type}\n"
        f"Kategória: {category}\n\n"
        f"Szöveg:\n{truncated}"
    )


def generate_context_sync(
    text: str,
    source: str,
    chunk_type: str = "document",
    category: str = "general",
) -> str:
    """Generate context prefix for a single chunk (synchronous)."""
    if not OPENAI_API_KEY:
        return ""

    user_prompt = _build_user_prompt(text, source, chunk_type, category)

    resp = httpx.post(
        f"{OPENAI_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 150,
            "temperature": 0.3,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def generate_contexts_batch_inline(
    chunks: list[dict[str, Any]],
    batch_size: int = 20,
) -> list[str]:
    """Generate context prefixes for multiple chunks using regular API (not Batch API).

    Good for small batches (<50 chunks). For larger batches, use Batch API.
    Each chunk dict needs: text, source, chunk_type, category
    """
    contexts = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        for chunk in batch:
            try:
                ctx = generate_context_sync(
                    text=chunk["text"],
                    source=chunk.get("source", ""),
                    chunk_type=chunk.get("chunk_type", "document"),
                    category=chunk.get("category", "general"),
                )
                contexts.append(ctx)
            except Exception as e:
                print(f"[llm-enrich] Failed for chunk: {e}")
                contexts.append("")
        if i + batch_size < len(chunks):
            time.sleep(0.5)  # Rate limit
    return contexts


# ── OpenAI Batch API ──────────────────────────────────────────────────────

def create_batch_file(
    chunks: list[dict[str, Any]],
    output_path: str,
) -> str:
    """Create a JSONL file for OpenAI Batch API.

    Returns the output file path.
    """
    with open(output_path, "w") as f:
        for i, chunk in enumerate(chunks):
            user_prompt = _build_user_prompt(
                text=chunk["text"],
                source=chunk.get("source", ""),
                chunk_type=chunk.get("chunk_type", "document"),
                category=chunk.get("category", "general"),
            )
            request = {
                "custom_id": chunk.get("id", f"chunk-{i}"),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": 150,
                    "temperature": 0.3,
                },
            }
            f.write(json.dumps(request) + "\n")
    return output_path


def submit_batch(jsonl_path: str) -> str:
    """Upload JSONL and submit batch job. Returns batch_id."""
    # Upload file
    with open(jsonl_path, "rb") as f:
        resp = httpx.post(
            f"{OPENAI_BASE}/files",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            files={"file": ("batch_input.jsonl", f, "application/jsonl")},
            data={"purpose": "batch"},
            timeout=120,
        )
    resp.raise_for_status()
    file_id = resp.json()["id"]
    print(f"[batch] Uploaded file: {file_id}")

    # Create batch
    resp = httpx.post(
        f"{OPENAI_BASE}/batches",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "input_file_id": file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
        timeout=30,
    )
    resp.raise_for_status()
    batch_id = resp.json()["id"]
    print(f"[batch] Created batch: {batch_id}")
    return batch_id


def check_batch_status(batch_id: str) -> dict:
    """Check batch job status."""
    resp = httpx.get(
        f"{OPENAI_BASE}/batches/{batch_id}",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def download_batch_results(batch_id: str) -> dict[str, str]:
    """Download batch results. Returns {custom_id: context_text}."""
    status = check_batch_status(batch_id)
    if status["status"] != "completed":
        raise ValueError(f"Batch not completed: {status['status']}")

    output_file_id = status["output_file_id"]
    resp = httpx.get(
        f"{OPENAI_BASE}/files/{output_file_id}/content",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        timeout=120,
    )
    resp.raise_for_status()

    results = {}
    for line in resp.text.strip().split("\n"):
        obj = json.loads(line)
        custom_id = obj["custom_id"]
        try:
            content = obj["response"]["body"]["choices"][0]["message"]["content"].strip()
            results[custom_id] = content
        except (KeyError, IndexError):
            results[custom_id] = ""

    return results
