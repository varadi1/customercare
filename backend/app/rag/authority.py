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
# Increased from 0.40 to 0.55 to ensure official docs outrank emails
AUTHORITY_INFLUENCE = 0.55

# Chunk types that should be guaranteed in top results when relevant
PRIORITY_CHUNK_TYPES = {
    "felhívás", "melléklet", "gyik", "közlemény", "segédlet",  # PostgreSQL doc_type
    "palyazat_felhivas", "palyazat_melleklet", "kozlemeny", "segedlet",  # Legacy ChromaDB
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

    Three guarantees:
    1. Priority chunks with decent relevance get into top 3
    2. If top 5 are all emails, promote best non-email to position 2
    3. Source-type diversity: ensure at least 2 distinct priority types in top 5
       (e.g. felhívás + gyik, not just felhívás x3)

    Modifies the list in-place.
    """
    if len(results) <= 3:
        return

    EMAIL_TYPES = ("email_reply", "email_question", "email_qa", "lesson")

    # 1. Priority floor: promote priority chunks into top 3
    for i in range(3, len(results)):
        chunk_type = results[i].get("chunk_type", "")
        pre_score = results[i].get("pre_authority_score", 0)

        if chunk_type in PRIORITY_CHUNK_TYPES and pre_score > 0.3:
            # Find the lowest-ranked non-priority item in top 3 to swap with
            for j in range(2, -1, -1):
                if results[j].get("chunk_type", "") not in PRIORITY_CHUNK_TYPES:
                    results[j], results[i] = results[i], results[j]
                    break

    # 2. Diversity guarantee: if top 5 are ALL email types, promote best non-email
    top5_types = [r.get("chunk_type", "") for r in results[:5]]
    if all(t in EMAIL_TYPES for t in top5_types):
        for i in range(5, len(results)):
            ct = results[i].get("chunk_type", "")
            if ct not in EMAIL_TYPES:
                item = results.pop(i)
                results.insert(2, item)
                break

    # 3. Source-type diversity: ensure multiple priority types in top 5
    # Only swap OUT email/lesson types — never displace felhívás or melléklet
    PROTECTED_TYPES = {"felhívás", "palyazat_felhivas", "melléklet", "palyazat_melleklet"}
    top5_priority_types = {
        r.get("chunk_type", "") for r in results[:5]
    } & PRIORITY_CHUNK_TYPES

    if len(top5_priority_types) <= 1 and len(results) > 5:
        for i in range(5, len(results)):
            ct = results[i].get("chunk_type", "")
            pre_score = results[i].get("pre_authority_score", 0)
            if ct in PRIORITY_CHUNK_TYPES and ct not in top5_priority_types and pre_score > 0.15:
                # Replace the weakest email in top 5 (never displace protected types)
                for j in range(4, -1, -1):
                    rj_type = results[j].get("chunk_type", "")
                    if rj_type in EMAIL_TYPES:
                        results[j], results[i] = results[i], results[j]
                        break
                break
