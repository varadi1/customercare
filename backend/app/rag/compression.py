"""Contextual Compression — post-rerank filtering to reduce noise.

Two complementary strategies:
1. Score floor: remove chunks with rerank_score below a hard threshold
2. Elbow detection: find the steepest score drop and cut there

This is especially important for the OETP corpus where ~9K email chunks
compete with ~550 official doc chunks. Even after reranking, low-relevance
emails can occupy slots that would be better used by official docs ranked
lower but genuinely relevant.

IMPORTANT: Priority chunk types (felhívás, gyik, segédlet, etc.) are NEVER
removed by compression — they are protected. Only email-type chunks with
low rerank scores get compressed out.
"""

from __future__ import annotations

from ..config import settings
from .authority import PRIORITY_CHUNK_TYPES


def compress_results(
    results: list[dict],
    min_results: int | None = None,
    score_floor: float | None = None,
    gap_ratio: float | None = None,
) -> list[dict]:
    """Filter post-rerank results by relevance quality.

    Args:
        results: Reranked results (must have 'rerank_score').
        min_results: Always keep at least this many results (default from config).
        score_floor: Remove results below this rerank_score (default from config).
        gap_ratio: If score drops by this ratio vs the TOP result, cut there.
                   E.g., 0.15 means "if score < 15% of the best result".

    Returns:
        Filtered results list (preserves order).
    """
    if not results or len(results) <= 2:
        return results

    min_results = min_results if min_results is not None else settings.compression_min_results
    score_floor = score_floor if score_floor is not None else settings.compression_score_floor
    gap_ratio = gap_ratio if gap_ratio is not None else settings.compression_gap_ratio

    # Ensure we have rerank_scores to work with
    if not any(r.get("rerank_score") for r in results):
        return results

    top_score = results[0].get("rerank_score", 0)
    if top_score <= 0:
        return results

    compressed = []
    removed_scores = []

    for i, r in enumerate(results):
        rs = r.get("rerank_score", 0)
        ct = r.get("chunk_type", "")

        # Priority chunks are NEVER compressed out
        if ct in PRIORITY_CHUNK_TYPES:
            compressed.append(r)
            continue

        # Always keep at least min_results
        if len(compressed) < min_results:
            compressed.append(r)
            continue

        # Strategy 1: Hard score floor
        if rs < score_floor:
            removed_scores.append(round(rs, 4))
            continue

        # Strategy 2: Ratio vs top score — only remove if dramatically worse
        if rs / top_score < gap_ratio:
            removed_scores.append(round(rs, 4))
            continue

        compressed.append(r)

    if removed_scores:
        print(
            f"[cc] Compression: {len(removed_scores)} chunks removed "
            f"(scores: {removed_scores[:5]}), {len(compressed)} kept"
        )

    return compressed
