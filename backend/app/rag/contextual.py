"""Contextual embeddings — enrich chunks with context prefix before embedding.

Based on the technique from the Jogszabály RAG: prepend a short context
description to each chunk before embedding, so the embedding captures
not just the text content but also WHERE it comes from and WHAT it's about.
"""

from __future__ import annotations


# Context templates by chunk_type
CONTEXT_TEMPLATES: dict[str, str] = {
    "palyazat_felhivas": (
        "Ez az OETP (Otthonfelújítási Program) hivatalos pályázati felhívásának részlete. "
        "Forrás: {source}. Ez a pályázat legfontosabb, legautoritatívabb dokumentuma."
    ),
    "palyazat_melleklet": (
        "Ez az OETP pályázati felhívás hivatalos mellékletének részlete. "
        "Forrás: {source}. Kiegészíti a felhívás feltételeit."
    ),
    "kozlemeny": (
        "Ez egy hivatalos közlemény az OETP programmal kapcsolatban. "
        "Forrás: {source}."
    ),
    "gyik": (
        "Ez egy gyakran ismételt kérdés (GYIK) és válasz az OETP programról. "
        "Forrás: {source}."
    ),
    "segedlet": (
        "Ez egy segédlet/útmutató az OETP pályázathoz. "
        "Forrás: {source}."
    ),
    "email_reply": (
        "Ez egy korábbi ügyfélszolgálati email válasz az OETP programmal kapcsolatban. "
        "Forrás: {source} postafiók. Korábbi válasz, nem feltétlenül aktuális."
    ),
    "email_qa": (
        "Ez egy korábbi ügyfélszolgálati kérdés-válasz pár az OETP programmal kapcsolatban. "
        "Forrás: {source} postafiók."
    ),
    "document": (
        "Ez egy dokumentum az OETP programmal kapcsolatban. "
        "Forrás: {source}."
    ),
    "general": (
        "Ez egy dokumentum az OETP ügyfélszolgálati tudásbázisból. "
        "Forrás: {source}."
    ),
}

DEFAULT_TEMPLATE = "Forrás: {source}. Típus: {chunk_type}."


def build_context_prefix(
    chunk_type: str,
    source: str,
    category: str = "",
    **kwargs,
) -> str:
    """Build a context prefix for a chunk.
    
    Args:
        chunk_type: Type of the chunk (e.g., 'palyazat_felhivas', 'email_reply')
        source: Source identifier (filename, mailbox, etc.)
        category: Category tag
        
    Returns:
        Context prefix string to prepend to the chunk text before embedding
    """
    template = CONTEXT_TEMPLATES.get(chunk_type, DEFAULT_TEMPLATE)
    return template.format(
        source=source,
        chunk_type=chunk_type,
        category=category,
        **kwargs,
    )


def enrich_chunk(
    text: str,
    chunk_type: str,
    source: str,
    category: str = "",
    **kwargs,
) -> str:
    """Enrich a chunk with context prefix.
    
    Returns the full text (prefix + original) for embedding.
    The original text is still stored separately in ChromaDB.
    """
    prefix = build_context_prefix(chunk_type, source, category, **kwargs)
    return f"{prefix}\n\n{text}"
