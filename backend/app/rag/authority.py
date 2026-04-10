"""Authority weighting — boost results from authoritative sources.

Weights loaded from program.yaml doc_types section. Falls back to hardcoded defaults.
"""

from __future__ import annotations

from ..config import get_program_config


def _build_authority_weights() -> dict[str, float]:
    """Build authority weights from program.yaml doc_types. Falls back to defaults."""
    pcfg = get_program_config()
    doc_types = pcfg.get("doc_types", {})

    weights = {}
    for dt_name, dt_cfg in doc_types.items():
        if isinstance(dt_cfg, dict) and "authority" in dt_cfg:
            weights[dt_name] = dt_cfg["authority"]

    if not weights:
        # Fallback defaults
        weights = {
            "felhívás": 0.95, "melléklet": 0.90, "közlemény": 0.85,
            "gyik": 0.80, "segédlet": 0.75, "dokumentum": 0.60,
            "email_reply": 0.40, "email_question": 0.35,
        }

    return weights


AUTHORITY_WEIGHTS = _build_authority_weights()

# Default weight for unknown chunk types
DEFAULT_WEIGHT = 0.45

# How much authority affects the final score (0 = no effect, 1 = full effect)
# Increased from 0.40 to 0.55 to ensure official docs outrank emails
AUTHORITY_INFLUENCE = 0.55

# Chunk types that should be guaranteed in top results when relevant
# Built from doc_types with authority >= 0.75 (official documents)
PRIORITY_CHUNK_TYPES = {
    name for name, cfg in get_program_config().get("doc_types", {}).items()
    if isinstance(cfg, dict) and cfg.get("authority", 0) >= 0.75
} or {"felhívás", "melléklet", "gyik", "közlemény", "segédlet"}


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

    # 1. Email cap: max 2 email-type chunks in top 5
    # Problem: with 9K+ email chunks, they dominate top positions even when
    # official docs are more authoritative. Cap emails to leave room for
    # multiple priority doc types (felhívás + gyik + segédlet).
    email_count_top5 = sum(1 for r in results[:5] if r.get("chunk_type", "") in EMAIL_TYPES)
    if email_count_top5 > 2 and len(results) > 5:
        # Find non-email chunks below position 5 to promote
        for i in range(5, len(results)):
            ct = results[i].get("chunk_type", "")
            if ct not in EMAIL_TYPES:
                # Replace weakest email in top 5
                for j in range(4, -1, -1):
                    if results[j].get("chunk_type", "") in EMAIL_TYPES:
                        results[j], results[i] = results[i], results[j]
                        email_count_top5 -= 1
                        break
                if email_count_top5 <= 2:
                    break

    # 2. Priority floor: promote priority chunks into top 3
    # Threshold lowered: if a priority chunk made it through retrieval + reranking,
    # it's relevant enough to deserve a top position regardless of score.
    for i in range(3, len(results)):
        chunk_type = results[i].get("chunk_type", "")
        pre_score = results[i].get("pre_authority_score", 0)

        if chunk_type in PRIORITY_CHUNK_TYPES and pre_score > 0.0:
            # Find the lowest-ranked non-priority item in top 3 to swap with
            for j in range(2, -1, -1):
                if results[j].get("chunk_type", "") not in PRIORITY_CHUNK_TYPES:
                    results[j], results[i] = results[i], results[j]
                    break

    # 3. Diversity guarantee: if top 5 are ALL email types, promote best non-email
    top5_types = [r.get("chunk_type", "") for r in results[:5]]
    if all(t in EMAIL_TYPES for t in top5_types):
        for i in range(5, len(results)):
            ct = results[i].get("chunk_type", "")
            if ct not in EMAIL_TYPES:
                item = results.pop(i)
                results.insert(2, item)
                break

    # 4. Source-type diversity: maximize distinct priority types in top 5
    # Keep promoting until we either run out of missing types or swappable email slots.
    PROTECTED_TYPES = {"felhívás", "palyazat_felhivas", "melléklet", "palyazat_melleklet"}

    for _round in range(3):  # max 3 promotion rounds
        top5_priority_types = {
            r.get("chunk_type", "") for r in results[:5]
        } & PRIORITY_CHUNK_TYPES

        promoted = False
        for i in range(5, len(results)):
            ct = results[i].get("chunk_type", "")
            if ct in PRIORITY_CHUNK_TYPES and ct not in top5_priority_types:
                # Replace the weakest email in top 5 (never displace protected types)
                for j in range(4, -1, -1):
                    rj_type = results[j].get("chunk_type", "")
                    if rj_type in EMAIL_TYPES:
                        results[j], results[i] = results[i], results[j]
                        promoted = True
                        break
                break
        if not promoted:
            break
