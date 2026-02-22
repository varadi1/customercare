"""Authority weighting — boost results from authoritative sources."""

from __future__ import annotations

# Authority weights by chunk_type
# Higher = more authoritative (0.0 - 1.0)
AUTHORITY_WEIGHTS: dict[str, float] = {
    # Official program documents — always preferred
    # PostgreSQL doc_type names (Hungarian) + ChromaDB legacy names
    "felhívás": 1.00,               # Pályázati felhívás — THE source of truth
    "palyazat_felhivas": 1.00,      # Legacy ChromaDB name
    "melléklet": 0.95,              # Felhívás mellékletei
    "palyazat_melleklet": 0.95,     # Legacy
    "közlemény": 0.90,              # Hivatalos közlemények
    "kozlemeny": 0.90,              # Legacy
    "gyik": 0.85,                   # GYIK — curated Q&A
    "segédlet": 0.80,               # Segédletek, útmutatók
    "segedlet": 0.80,               # Legacy
    
    # Internal knowledge — lower priority
    "dokumentum": 0.55,             # Általános dokumentumok
    "document": 0.55,               # Legacy
    "general": 0.50,                # Egyéb
    
    # Email-based knowledge — useful patterns but NOT authoritative
    "email_reply": 0.40,            # Korábbi email válaszok
    "email_qa": 0.40,               # Email Q&A párok
    "email_question": 0.30,         # Beérkezett kérdések (legalacsonyabb)
}

# Default weight for unknown chunk types
DEFAULT_WEIGHT = 0.45

# How much authority affects the final score (0 = no effect, 1 = full effect)
AUTHORITY_INFLUENCE = 0.40

# Chunk types that should be guaranteed in top results when relevant
PRIORITY_CHUNK_TYPES = {
    "felhívás", "melléklet", "gyik", "közlemény",  # PostgreSQL doc_type
    "palyazat_felhivas", "palyazat_melleklet", "kozlemeny",  # Legacy ChromaDB
}


def get_authority_weight(chunk_type: str) -> float:
    """Get the authority weight for a chunk type."""
    return AUTHORITY_WEIGHTS.get(chunk_type, DEFAULT_WEIGHT)


def apply_authority_weighting(results: list[dict], influence: float | None = None) -> list[dict]:
    """Apply authority weighting to search results.
    
    Final score = base_score * (1 - influence) + base_score * authority * influence
    
    This means:
    - A high-authority source gets a boost
    - A low-authority source gets penalized
    - The influence parameter controls how much authority matters
    
    Args:
        results: List of search result dicts (must have 'score' and 'chunk_type')
        influence: How much authority affects score (0-1, default: AUTHORITY_INFLUENCE)
    
    Returns:
        Results with adjusted scores, re-sorted by new score
    """
    if not results:
        return results
    
    influence = influence if influence is not None else AUTHORITY_INFLUENCE
    
    weighted = []
    for doc in results:
        doc = doc.copy()
        chunk_type = doc.get("chunk_type", "general")
        authority = get_authority_weight(chunk_type)
        base_score = doc.get("score", 0)
        
        # Weighted combination: relevance stays primary, authority is a modifier
        adjusted_score = base_score * (1 - influence) + base_score * authority * influence
        
        doc["score"] = round(adjusted_score, 4)
        doc["authority_weight"] = authority
        doc["pre_authority_score"] = base_score
        weighted.append(doc)
    
    # Re-sort by adjusted score
    weighted.sort(key=lambda x: x["score"], reverse=True)
    
    # Authority floor: ensure priority chunk types with score > 0.5 are in top 3
    _apply_authority_floor(weighted)
    
    return weighted


def _apply_authority_floor(results: list[dict]) -> None:
    """Ensure high-authority chunks with decent scores are in the top results.
    
    If a priority chunk type (felhivas, melleklet, gyik, kozlemeny) has
    pre_authority_score > 0.4 but is not in the top 3, swap it in.
    
    Also ensures at least one non-email_reply source appears in top 5
    when available (diversity guarantee).
    Modifies the list in-place.
    """
    if len(results) <= 3:
        return
    
    # 1. Priority floor: promote priority chunks into top 3
    for i in range(3, len(results)):
        chunk_type = results[i].get("chunk_type", "")
        pre_score = results[i].get("pre_authority_score", 0)
        
        if chunk_type in PRIORITY_CHUNK_TYPES and pre_score > 0.4:
            # Find the lowest-ranked non-priority item in top 3 to swap with
            for j in range(2, -1, -1):
                if results[j].get("chunk_type", "") not in PRIORITY_CHUNK_TYPES:
                    results[j], results[i] = results[i], results[j]
                    break
    
    # 2. Diversity guarantee: if top 5 are ALL email_reply, promote the best non-reply
    top5_types = [r.get("chunk_type", "") for r in results[:5]]
    if all(t in ("email_reply", "email_question", "email_qa") for t in top5_types):
        for i in range(5, len(results)):
            ct = results[i].get("chunk_type", "")
            if ct not in ("email_reply", "email_question", "email_qa"):
                # Promote to position 2 (after best result, before the rest)
                item = results.pop(i)
                results.insert(2, item)
                break
