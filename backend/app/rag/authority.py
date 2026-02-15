"""Authority weighting — boost results from authoritative sources."""

from __future__ import annotations

# Authority weights by chunk_type
# Higher = more authoritative (0.0 - 1.0)
AUTHORITY_WEIGHTS: dict[str, float] = {
    # Official program documents
    "palyazat_felhivas": 1.00,      # Pályázati felhívás — THE source of truth
    "palyazat_melleklet": 0.95,     # Felhívás mellékletei
    "kozlemeny": 0.85,              # Hivatalos közlemények
    "gyik": 0.80,                   # GYIK — curated Q&A
    "segedlet": 0.75,               # Segédletek, útmutatók
    
    # Internal knowledge
    "document": 0.65,               # Általános dokumentumok
    "general": 0.60,                # Egyéb
    
    # Email-based knowledge (useful but not authoritative)
    "email_reply": 0.50,            # Korábbi email válaszok
    "email_qa": 0.50,               # Email Q&A párok
}

# Default weight for unknown chunk types
DEFAULT_WEIGHT = 0.55

# How much authority affects the final score (0 = no effect, 1 = full effect)
AUTHORITY_INFLUENCE = 0.40

# Chunk types that should be guaranteed in top results when relevant
PRIORITY_CHUNK_TYPES = {"palyazat_felhivas", "palyazat_melleklet", "gyik", "kozlemeny"}


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
    """Ensure high-authority chunks with decent scores are in the top 3.
    
    If a priority chunk type (felhivas, melleklet, gyik, kozlemeny) has
    pre_authority_score > 0.5 but is not in the top 3, swap it in.
    Modifies the list in-place.
    """
    if len(results) <= 3:
        return
    
    # Find priority chunks outside top 3 that have good scores
    for i in range(3, len(results)):
        chunk_type = results[i].get("chunk_type", "")
        pre_score = results[i].get("pre_authority_score", 0)
        
        if chunk_type in PRIORITY_CHUNK_TYPES and pre_score > 0.5:
            # Find the lowest-ranked non-priority item in top 3 to swap with
            for j in range(2, -1, -1):
                if results[j].get("chunk_type", "") not in PRIORITY_CHUNK_TYPES:
                    results[j], results[i] = results[i], results[j]
                    break
