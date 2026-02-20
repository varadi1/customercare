"""Document ingestion — chunk, embed, store in ChromaDB."""

from __future__ import annotations

import hashlib
import uuid
from datetime import date

import fitz  # pymupdf
from bs4 import BeautifulSoup

from ..config import settings
from .chunker import chunk_text, chunk_markdown
from .embeddings import embed_texts_ingest as embed_texts
from .search import get_collection
from .contextual import enrich_chunk


def _generate_chunk_id(source: str, index: int) -> str:
    """Deterministic chunk ID from source + index."""
    raw = f"{source}::{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


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
) -> int:
    """Chunk, embed, and store text in ChromaDB.

    Returns number of chunks created.
    """
    # Chunk
    if use_markdown_chunker:
        chunks = chunk_markdown(text)
    else:
        chunks = chunk_text(text)

    if not chunks:
        return 0

    # Contextual enrichment: embed with context prefix for better retrieval
    enriched_chunks = [
        enrich_chunk(chunk, chunk_type=chunk_type, source=source, category=category)
        for chunk in chunks
    ]

    # Auto-supersede: if supersedes is set, invalidate the old source
    if supersedes:
        try:
            collection = get_collection()
            old = collection.get(where={"source": supersedes}, include=["metadatas"])
            if old["ids"]:
                for meta in old["metadatas"]:
                    meta["valid_to"] = valid_from or date.today().isoformat()
                    meta["superseded_by"] = source
                collection.update(ids=old["ids"], metadatas=old["metadatas"])
                print(f"[ingest] Auto-superseded {len(old['ids'])} chunks from '{supersedes}'")
        except Exception as e:
            print(f"[ingest] Supersede warning (non-fatal): {e}")

    # Embed the ENRICHED chunks (but store original text in ChromaDB)
    embeddings = embed_texts(enriched_chunks)

    # Prepare metadata
    today = date.today().isoformat()
    ids = []
    metadatas = []
    for i, chunk in enumerate(chunks):
        chunk_id = _generate_chunk_id(source, i)
        ids.append(chunk_id)
        metadatas.append({
            "source": source,
            "category": category,
            "chunk_type": chunk_type,
            "valid_from": valid_from or today,
            "valid_to": valid_to or "",
            "version": version,
            "supersedes": supersedes or "",
            "indexed_at": today,
            "chunk_index": i,
        })

    # Store in ChromaDB
    collection = get_collection()
    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(chunks)


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
        text=full_text,
        source=source,
        category=category,
        chunk_type=chunk_type,
        **kwargs,
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
        text=text,
        source=source,
        category=category,
        chunk_type=chunk_type,
        **kwargs,
    )


def ingest_email_pair(
    question_text: str,
    answer_text: str,
    source: str,
    category: str = "general",
    **kwargs,
) -> int:
    """Ingest a question-answer email pair as a single unit.

    Keeps Q&A together so retrieval returns the full context.
    """
    combined = f"KÉRDÉS:\n{question_text}\n\nVÁLASZ:\n{answer_text}"

    return ingest_text(
        text=combined,
        source=source,
        category=category,
        chunk_type="email_reply",
        **kwargs,
    )


def expire_chunks(source: str, version_below: int | None = None) -> int:
    """Mark chunks as expired (set valid_to to today).

    Used when a document is updated and old chunks should be retired.
    """
    collection = get_collection()
    today = date.today().isoformat()

    # Get existing chunks for this source
    where = {"source": source}
    if version_below:
        where = {
            "$and": [
                {"source": source},
                {"version": {"$lt": version_below}},
            ]
        }

    existing = collection.get(where=where, include=["metadatas"])

    if not existing["ids"]:
        return 0

    # Update valid_to
    updated_metadatas = []
    for meta in existing["metadatas"]:
        meta["valid_to"] = today
        updated_metadatas.append(meta)

    collection.update(
        ids=existing["ids"],
        metadatas=updated_metadatas,
    )

    return len(existing["ids"])
