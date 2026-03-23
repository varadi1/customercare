"""Adaptive k — dynamically adjust retrieval depth based on query complexity.

Simple queries ("mi az OETP?") need k=5, while complex multi-step queries
("milyen lépésekből áll a pályázat benyújtása?") need k=8-10 to capture
multiple relevant document types (felhívás + GYIK + segédlet + melléklet).

Uses lightweight heuristics — no LLM call needed.
"""

from __future__ import annotations

import re

from ..config import settings

# OETP-specific multi-concept indicators
_MULTI_STEP_MARKERS = re.compile(
    r"(?:hogyan|milyen lépés|mi kell|mit kell|mik a|milyen dokumentum|"
    r"milyen feltétel|mi szükséges|mikor kell|mennyit kell|hány|összehasonlít|"
    r"különbség|eltérés|miben más|miben tér el|egyszerre|is.*is|valamint|illetve|"
    r"milyen melléklet|benyújtás.*lépés|lépésről lépésre)",
    re.IGNORECASE,
)

_SIMPLE_MARKERS = re.compile(
    r"^(?:mi az?|ki az?|hol (?:van|található)|mikor (?:volt|van|lesz)|"
    r"mennyi az?|melyik|hány darab|hol kell)\b",
    re.IGNORECASE,
)

# Known OETP entity terms (presence of multiple → complex query)
_ENTITY_TERMS = {
    "inverter", "napelem", "akkumulátor", "energiatároló", "hőszivattyú",
    "villanyóra", "mérőóra", "fogyasztásmérő", "szaldó", "betáplálás",
    "hálózati csatlakozás", "engedély", "tervrajz", "nyilatkozat",
    "e-közmű", "hmke", "kiserőmű", "naperőmű", "pályázat", "felhívás",
    "hiánypótlás", "elszámolás", "szerződés", "meghatalmazás",
    "társasház", "tulajdoni lap", "albetét",
}


def classify_query(query: str) -> str:
    """Classify query complexity as 'simple', 'medium', or 'complex'.

    Returns:
        One of: 'simple', 'medium', 'complex'
    """
    words = query.split()
    word_count = len(words)
    query_lower = query.lower()

    # Check for explicit multi-step / comparison markers
    has_multi_marker = bool(_MULTI_STEP_MARKERS.search(query_lower))
    has_simple_marker = bool(_SIMPLE_MARKERS.search(query_lower))

    # Count distinct entity terms mentioned
    entity_count = sum(1 for term in _ENTITY_TERMS if term in query_lower)

    # Classification logic
    if has_simple_marker and word_count <= 8 and entity_count <= 1:
        return "simple"

    if has_multi_marker or entity_count >= 3 or word_count > 15:
        return "complex"

    if entity_count >= 2 or word_count > 10:
        return "medium"

    return "simple"


def get_adaptive_k(query: str) -> dict:
    """Get adaptive k values based on query complexity.

    Returns:
        Dict with 'final_k', 'retrieval_k', 'rerank_k', 'complexity'.
    """
    if not settings.adaptive_k_enabled:
        return {
            "final_k": settings.rerank_top_k,
            "retrieval_k": settings.search_top_k,
            "rerank_k": settings.rerank_top_k + 5,
            "complexity": "default",
        }

    complexity = classify_query(query)

    k_map = {
        "simple": {
            "final_k": settings.rerank_top_k,       # 5
            "retrieval_k": settings.search_top_k,    # 20
        },
        "medium": {
            "final_k": settings.rerank_top_k + 2,    # 7
            "retrieval_k": settings.search_top_k + 5, # 25
        },
        "complex": {
            "final_k": settings.rerank_top_k + 5,    # 10
            "retrieval_k": settings.search_top_k + 10, # 30
        },
    }

    values = k_map[complexity]
    rerank_k = min(values["final_k"] + 5, values["retrieval_k"])

    print(f"[hanna-oetp] Adaptive k: '{complexity}' → final_k={values['final_k']}, retrieval_k={values['retrieval_k']}")

    return {
        "final_k": values["final_k"],
        "retrieval_k": values["retrieval_k"],
        "rerank_k": rerank_k,
        "complexity": complexity,
    }
