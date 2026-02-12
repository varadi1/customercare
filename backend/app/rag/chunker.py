"""Document chunking with tiktoken."""

from __future__ import annotations

import tiktoken
from ..config import settings


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """Split text into overlapping token-based chunks.

    Uses tiktoken (cl100k_base) for accurate token counting.
    Returns list of text chunks.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    if len(tokens) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)

        if end >= len(tokens):
            break

        start += chunk_size - chunk_overlap

    return chunks


def chunk_markdown(text: str) -> list[str]:
    """Chunk markdown with awareness of headings and paragraphs.

    Tries to split on paragraph boundaries, falling back to
    token-based chunking if paragraphs are too large.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    chunk_size = settings.chunk_size
    chunk_overlap = settings.chunk_overlap

    # Split by double newline (paragraphs)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current_chunk: list[str] = []
    current_tokens = 0
    current_heading = ""

    for para in paragraphs:
        # Track headings for context
        if para.startswith("#"):
            current_heading = para

        para_tokens = len(enc.encode(para))

        # Single paragraph exceeds chunk size — force split
        if para_tokens > chunk_size:
            # Flush current chunk
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_tokens = 0

            # Token-split the big paragraph
            sub_chunks = chunk_text(para, chunk_size, chunk_overlap)
            for sc in sub_chunks:
                # Prepend heading for context if available
                if current_heading and not sc.startswith("#"):
                    chunks.append(f"{current_heading}\n\n{sc}")
                else:
                    chunks.append(sc)
            continue

        # Would adding this paragraph exceed chunk size?
        if current_tokens + para_tokens > chunk_size and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            # Keep last paragraph for overlap context
            overlap_text = current_chunk[-1] if current_chunk else ""
            overlap_tokens = len(enc.encode(overlap_text))
            if overlap_tokens <= chunk_overlap:
                current_chunk = [overlap_text]
                current_tokens = overlap_tokens
            else:
                current_chunk = []
                current_tokens = 0

        current_chunk.append(para)
        current_tokens += para_tokens

    # Flush remaining
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks
