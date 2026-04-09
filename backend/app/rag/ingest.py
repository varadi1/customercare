"""Document ingestion — chunk, embed, store in PostgreSQL+pgvector.

Migrated from ChromaDB to PostgreSQL for OETP knowledge base.
Database: postgresql://klara:klara_docs_2026@host.docker.internal:5433/hanna_oetp
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import date

import asyncpg
import fitz  # pymupdf
from bs4 import BeautifulSoup

from ..config import settings
from .chunker import chunk_text, chunk_markdown
from .embeddings import embed_texts_ingest as embed_texts
from .contextual import enrich_chunk
from .kg_extract import extract_and_store as kg_extract_chunk

import os
PG_DSN = os.environ.get("HANNA_PG_DSN", "postgresql://klara:klara_docs_2026@hanna-db:5432/hanna_oetp")

# chunk_type → doc_type mapping (legacy ChromaDB names → PostgreSQL)
DOC_TYPE_MAP = {
    "palyazat_felhivas": "felhívás",
    "palyazat_melleklet": "melléklet",
    "kozlemeny": "közlemény",
    "gyik": "gyik",
    "segedlet": "segédlet",
    "email_reply": "email_reply",
    "email_question": "email_question",
    "email_qa": "email_reply",
    "document": "dokumentum",
    "general": "dokumentum",
}

AUTHORITY_MAP = {
    "felhívás": 0.95,
    "melléklet": 0.90,
    "közlemény": 0.85,
    "gyik": 0.80,
    "segédlet": 0.75,
    "dokumentum": 0.60,
    "email_reply": 0.40,
    "email_question": 0.35,
}

_pool: asyncpg.Pool | None = None


async def _get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(PG_DSN, min_size=2, max_size=10)
    return _pool


def _generate_chunk_id(source: str, index: int) -> str:
    raw = f"{source}::{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _resolve_doc_type(chunk_type: str) -> str:
    return DOC_TYPE_MAP.get(chunk_type, "dokumentum")


def _resolve_authority(doc_type: str) -> float:
    return AUTHORITY_MAP.get(doc_type, 0.5)


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


async def ingest_text_async(
    text: str,
    source: str,
    category: str = "general",
    chunk_type: str = "document",
    valid_from: str | None = None,
    valid_to: str | None = None,
    version: int = 1,
    supersedes: str | None = None,
    use_markdown_chunker: bool = False,
    program: str | None = None,
) -> int:
    """Chunk, embed, and store text in PostgreSQL.

    Returns number of chunks created.
    """
    if use_markdown_chunker:
        chunks = chunk_markdown(text)
    else:
        chunks = chunk_text(text)

    if not chunks:
        return 0

    doc_type = _resolve_doc_type(chunk_type)
    authority = _resolve_authority(doc_type)
    resolved_program = program or "OETP"

    # Contextual enrichment
    enriched_chunks = [
        enrich_chunk(chunk, chunk_type=chunk_type, source=source, category=category)
        for chunk in chunks
    ]

    # Embed the ENRICHED chunks
    embeddings = embed_texts(enriched_chunks)

    pool = await _get_pool()
    today = date.today().isoformat()

    # Auto-supersede old chunks
    if supersedes:
        try:
            async with pool.acquire() as conn:
                result = await conn.execute(
                    """UPDATE chunks 
                       SET metadata = metadata || $1::jsonb
                       WHERE doc_id = $2 AND (metadata->>'valid_to' IS NULL OR metadata->>'valid_to' = '')""",
                    json.dumps({"valid_to": valid_from or today, "superseded_by": source}),
                    supersedes,
                )
                print(f"[ingest] Auto-superseded chunks from '{supersedes}': {result}")
        except Exception as e:
            print(f"[ingest] Supersede warning (non-fatal): {e}")

    # Parse source_date
    source_date = None
    if valid_from:
        try:
            from datetime import datetime
            source_date = datetime.fromisoformat(valid_from)
        except (ValueError, TypeError):
            pass

    # Insert chunks
    inserted = 0
    async with pool.acquire() as conn:
        for i, (chunk, enriched, embedding) in enumerate(zip(chunks, enriched_chunks, embeddings)):
            chunk_id = _generate_chunk_id(source, i)
            embedding_str = "[" + ",".join(str(float(x)) for x in embedding) + "]"

            try:
                await conn.execute(
                    """INSERT INTO chunks (
                        id, doc_id, doc_type, program, chunk_index, title,
                        content, content_enriched, embedding, metadata,
                        authority_score, source_date, content_hash
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::vector, $10, $11, $12, $13)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        content_enriched = EXCLUDED.content_enriched,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        authority_score = EXCLUDED.authority_score,
                        updated_at = now()
                    """,
                    chunk_id,
                    source,
                    doc_type,
                    resolved_program,
                    i,
                    source,  # title = source filename
                    chunk,
                    enriched if enriched != chunk else None,
                    embedding_str,
                    json.dumps({
                        "category": category,
                        "chunk_type": chunk_type,
                        "version": version,
                        "supersedes": supersedes or "",
                        "valid_from": valid_from or today,
                        "valid_to": valid_to or "",
                        "indexed_at": today,
                    }),
                    authority,
                    source_date,
                    _content_hash(chunk),
                )
                inserted += 1
            except Exception as e:
                print(f"[ingest] Error inserting chunk {chunk_id}: {e}")

    # KG extraction for high-value doc types (non-blocking on failure)
    if doc_type in {"felhívás", "melléklet", "közlemény", "gyik", "segédlet", "dokumentum"}:
        kg_ent_total = 0
        kg_rel_total = 0
        try:
            async with pool.acquire() as conn:
                for i, chunk in enumerate(chunks):
                    chunk_id = _generate_chunk_id(source, i)
                    ent, rel = await kg_extract_chunk(conn, chunk_id, chunk, doc_type, source)
                    kg_ent_total += ent
                    kg_rel_total += rel
            if kg_ent_total or kg_rel_total:
                print(f"[ingest] KG: {kg_ent_total} entities, {kg_rel_total} relations from {source}")
        except Exception as e:
            print(f"[ingest] KG extraction warning (non-fatal): {e}")

    return inserted


def ingest_text(
    text: str,
    source: str,
    category: str = "general",
    chunk_type: str = "document",
    valid_from: str | None = None,
    valid_to: str | None = None,
    version: int = 1,
    supersedes: str | None = None,
    use_markdown_chunker: bool = False,
    program: str | None = None,
) -> int:
    """Sync wrapper for ingest_text_async."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If already in async context, create a new loop in thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    ingest_text_async(
                        text=text, source=source, category=category,
                        chunk_type=chunk_type, valid_from=valid_from,
                        valid_to=valid_to, version=version,
                        supersedes=supersedes,
                        use_markdown_chunker=use_markdown_chunker,
                        program=program,
                    )
                )
                return future.result()
        else:
            return loop.run_until_complete(
                ingest_text_async(
                    text=text, source=source, category=category,
                    chunk_type=chunk_type, valid_from=valid_from,
                    valid_to=valid_to, version=version,
                    supersedes=supersedes,
                    use_markdown_chunker=use_markdown_chunker,
                    program=program,
                )
            )
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(
            ingest_text_async(
                text=text, source=source, category=category,
                chunk_type=chunk_type, valid_from=valid_from,
                valid_to=valid_to, version=version,
                supersedes=supersedes,
                use_markdown_chunker=use_markdown_chunker,
                program=program,
            )
        )


def ingest_pdf(
    pdf_path: str,
    source: str | None = None,
    category: str = "general",
    chunk_type: str = "document",
    **kwargs,
) -> int:
    """Extract text from PDF and ingest."""
    doc = fitz.open(pdf_path)
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()

    full_text = "\n\n".join(text_parts)
    source = source or pdf_path.split("/")[-1]

    return ingest_text(
        text=full_text, source=source, category=category,
        chunk_type=chunk_type, **kwargs,
    )


def ingest_html(
    html: str,
    source: str,
    category: str = "general",
    chunk_type: str = "email_reply",
    **kwargs,
) -> int:
    """Extract text from HTML and ingest."""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n", strip=True)

    return ingest_text(
        text=text, source=source, category=category,
        chunk_type=chunk_type, **kwargs,
    )


def ingest_email_pair(
    question_text: str,
    answer_text: str,
    source: str,
    category: str = "general",
    **kwargs,
) -> int:
    """Ingest a question-answer email pair as a single unit."""
    combined = f"KÉRDÉS:\n{question_text}\n\nVÁLASZ:\n{answer_text}"

    return ingest_text(
        text=combined, source=source, category=category,
        chunk_type="email_reply", **kwargs,
    )


async def expire_chunks_async(source: str, version_below: int | None = None) -> int:
    """Mark chunks as expired (set valid_to metadata to today)."""
    pool = await _get_pool()
    today = date.today().isoformat()

    async with pool.acquire() as conn:
        if version_below:
            result = await conn.execute(
                """UPDATE chunks 
                   SET metadata = metadata || $1::jsonb
                   WHERE doc_id = $2 AND (metadata->>'version')::int < $3""",
                json.dumps({"valid_to": today}),
                source,
                version_below,
            )
        else:
            result = await conn.execute(
                """UPDATE chunks 
                   SET metadata = metadata || $1::jsonb
                   WHERE doc_id = $2""",
                json.dumps({"valid_to": today}),
                source,
            )

    # Parse "UPDATE N" result
    try:
        count = int(result.split()[-1])
    except (IndexError, ValueError):
        count = 0
    return count


def expire_chunks(source: str, version_below: int | None = None) -> int:
    """Sync wrapper for expire_chunks_async."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(expire_chunks_async(source, version_below))
