"""Cross-reference resolution for OETP knowledge base.

Identifies and resolves references in search results like:
- "Felhívás 4.2. pont" → fetch that section from the felhívás
- "GYIK 12. pont" → fetch that Q&A from the GYIK
- "1. számú melléklet" → reference to specific annex
- "lásd a segédletet" → link to the kitöltési segédlet
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .embeddings import embed_query


@dataclass
class Reference:
    """A detected cross-reference in a chunk."""
    ref_type: str       # "felhivas", "gyik", "melleklet", "segedlet", "kozlemeny"
    ref_text: str       # Original reference text found in the chunk
    section: str        # Section/point identifier (e.g. "4.2", "12")
    resolved: bool = False
    resolved_chunk: Optional[dict] = None


# ── Reference detection patterns ──────────────────────────────────

# Pattern: "Felhívás X.Y. pont/fejezet"
FELHIVAS_PATTERN = re.compile(
    r'[Ff]elhívás(?:ának?|ban|ból)?\s+'
    r'(\d+\.?\d*\.?)\s*'
    r'(?:pont|fejezet|szakasz|rész)',
    re.UNICODE
)

# Pattern: "X. pont/fejezet" in context of felhívás (broader)
SECTION_PATTERN = re.compile(
    r'(\d+\.?\d*\.?)\s*(?:pont|fejezet)\s*(?:sze?rint|alap[jJ]án|értelmében|ban|ben|ja|jában)?',
    re.UNICODE
)

# Pattern: "GYIK X. pont/kérdés"
GYIK_PATTERN = re.compile(
    r'GYIK\s*(?:-?\s*(?:ben|ből|t))?\s*(\d+)\s*\.?\s*(?:pont|kérdés)?',
    re.UNICODE
)

# Pattern: "X. számú melléklet"
MELLEKLET_PATTERN = re.compile(
    r'(\d+)\s*\.?\s*(?:számú\s+)?melléklet',
    re.UNICODE
)

# Pattern: references to segédlet
SEGEDLET_PATTERN = re.compile(
    r'(?:kitöltési\s+)?segédlet(?:ben|ből|et|nek|hez)?',
    re.UNICODE | re.IGNORECASE
)

# Pattern: references to közlemény
KOZLEMENY_PATTERN = re.compile(
    r'közlemény(?:ben|ből|t|nek|hez|ek)?',
    re.UNICODE | re.IGNORECASE
)


# ── Source identification ──────────────────────────────────────────

# Known source name fragments for matching
SOURCE_MATCHERS = {
    "felhivas": ["felhiv", "Felhiv", "felhívás"],
    "gyik": ["GYIK", "gyik", "gyakran"],
    "melleklet": ["melléklet", "melleklet", "útmutató", "utmutato"],
    "segedlet": ["segédlet", "segedlet", "kitöltés"],
    "kozlemeny": ["közlemény", "kozlemeny", "nffku"],
}


def detect_references(text: str) -> list[Reference]:
    """Detect cross-references in a chunk of text.
    
    Args:
        text: The chunk text to scan for references
        
    Returns:
        List of detected Reference objects
    """
    refs: list[Reference] = []
    seen = set()
    
    # Felhívás section references
    for m in FELHIVAS_PATTERN.finditer(text):
        section = m.group(1).rstrip('.')
        key = f"felhivas:{section}"
        if key not in seen:
            seen.add(key)
            refs.append(Reference(
                ref_type="felhivas",
                ref_text=m.group(0),
                section=section,
            ))
    
    # GYIK point references
    for m in GYIK_PATTERN.finditer(text):
        section = m.group(1)
        key = f"gyik:{section}"
        if key not in seen:
            seen.add(key)
            refs.append(Reference(
                ref_type="gyik",
                ref_text=m.group(0),
                section=section,
            ))
    
    # Melléklet references
    for m in MELLEKLET_PATTERN.finditer(text):
        section = m.group(1)
        key = f"melleklet:{section}"
        if key not in seen:
            seen.add(key)
            refs.append(Reference(
                ref_type="melleklet",
                ref_text=m.group(0),
                section=section,
            ))
    
    # Segédlet references (no section, just presence)
    if SEGEDLET_PATTERN.search(text):
        key = "segedlet:*"
        if key not in seen:
            seen.add(key)
            refs.append(Reference(
                ref_type="segedlet",
                ref_text="segédlet",
                section="*",
            ))
    
    return refs


def _get_collection():
    """Lazy import to avoid circular dependency."""
    from .search import get_collection
    return get_collection()


def resolve_reference(ref: Reference, max_results: int = 2) -> list[dict]:
    """Resolve a single reference by searching ChromaDB.
    
    Uses semantic search with a constructed query to find the referenced section.
    
    Args:
        ref: The Reference to resolve
        max_results: Maximum chunks to return per reference
        
    Returns:
        List of matching chunk dicts
    """
    collection = _get_collection()
    
    if collection.count() == 0:
        return []
    
    # Build a targeted search query based on reference type
    if ref.ref_type == "felhivas":
        search_query = f"Felhívás {ref.section}. pont fejezet"
        source_filter = "elhiv"  # matches "Felhivas" and "felhívás"
    elif ref.ref_type == "gyik":
        search_query = f"GYIK {ref.section}. kérdés pont"
        source_filter = "GYIK"   # matches "OEPT_GYIK"
    elif ref.ref_type == "melleklet":
        search_query = f"{ref.section}. számú melléklet"
        source_filter = None  # mellékletek can be in multiple sources
    elif ref.ref_type == "segedlet":
        search_query = "kitöltési segédlet útmutató"
        source_filter = "egedlet"  # matches "segédlet" and "segedlet"
    else:
        return []
    
    # Build ChromaDB where clause with source filter
    query_embedding = embed_query(search_query)
    
    where_clauses = [{"valid_to": ""}]
    
    # Source-level filtering — use chunk_type where known
    chunk_type_filter = None
    if ref.ref_type == "felhivas":
        chunk_type_filter = "document"  # Felhívás is "document"
    elif ref.ref_type == "gyik":
        chunk_type_filter = "faq"  # GYIK is "faq"
    elif ref.ref_type == "segedlet":
        chunk_type_filter = "document"
    
    if chunk_type_filter:
        where_clauses.append({"chunk_type": chunk_type_filter})
    
    where = where_clauses[0] if len(where_clauses) == 1 else {"$and": where_clauses}
    
    print(f"[hanna] Resolving ref: {ref.ref_type}:{ref.section} query='{search_query}'")
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=max_results * 10,  # Fetch more, then filter by source
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    
    if not results["documents"] or not results["documents"][0]:
        return []
    
    # Post-filter by source name
    matched = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        source = meta.get("source", "")
        distance = results["distances"][0][i]
        score = 1 - (distance / 2)
        
        # Source matching — must contain the filter string
        if source_filter:
            source_lower = source.lower()
            if source_filter.lower() not in source_lower:
                continue
        
        # Section matching: prefer chunks containing the section number
        # But don't hard-filter — the semantic search already targets the right area
        section_bonus = 0.0
        if ref.ref_type in ("felhivas", "gyik") and ref.section != "*":
            section_pattern = re.compile(rf'\b{re.escape(ref.section)}[\.\s]')
            if section_pattern.search(doc[:1500]):
                section_bonus = 0.2  # Boost chunks that mention the section
        
        matched.append({
            "id": results["ids"][0][i],
            "text": doc,
            "source": source,
            "category": meta.get("category", ""),
            "chunk_type": meta.get("chunk_type", ""),
            "score": round(score + section_bonus, 4),
            "metadata": meta,
            "ref_type": ref.ref_type,
            "ref_section": ref.section,
            "ref_text": ref.ref_text,
        })
        
        if len(matched) >= max_results:
            break
    
    if not matched:
        print(f"[hanna] Ref not resolved: {ref.ref_type}:{ref.section} (no matching source)")
    
    return matched


def resolve_references_in_results(
    results: list[dict],
    max_refs_per_result: int = 3,
    max_total_refs: int = 5,
) -> list[dict]:
    """Scan search results for cross-references and resolve them.
    
    Args:
        results: The main search results
        max_refs_per_result: Max references to resolve per result chunk
        max_total_refs: Maximum total referenced chunks to return
        
    Returns:
        List of referenced chunk dicts (deduplicated)
    """
    all_refs: list[Reference] = []
    
    # Detect references in all results
    for result in results:
        text = result.get("text", "")
        refs = detect_references(text)
        all_refs.extend(refs[:max_refs_per_result])
    
    if not all_refs:
        return []
    
    # Deduplicate by (type, section)
    seen = set()
    unique_refs = []
    for ref in all_refs:
        key = f"{ref.ref_type}:{ref.section}"
        if key not in seen:
            seen.add(key)
            unique_refs.append(ref)
    
    # Resolve each unique reference
    resolved_chunks: list[dict] = []
    resolved_ids: set[str] = set()
    
    # Also exclude IDs already in main results
    main_ids = {r.get("id", "") for r in results}
    
    for ref in unique_refs:
        if len(resolved_chunks) >= max_total_refs:
            break
        
        chunks = resolve_reference(ref)
        for chunk in chunks:
            chunk_id = chunk.get("id", "")
            if chunk_id not in resolved_ids and chunk_id not in main_ids:
                resolved_ids.add(chunk_id)
                resolved_chunks.append(chunk)
                
                if len(resolved_chunks) >= max_total_refs:
                    break
    
    return resolved_chunks
