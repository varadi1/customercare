"""Contextual Compression — post-rerank filtering to reduce noise.

Two complementary strategies:
1. Score floor: remove chunks with rerank_score below a hard threshold
2. Elbow detection: find the steepest score drop and cut there

This is especially important for the OETP corpus where ~9K email chunks
compete with ~550 official doc chunks. Even after reranking, low-relevance
emails can occupy slots that would be better used by official docs ranked
lower but genuinely relevant.
"""

from __future__ import annotations

from ..config import settings


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
        gap_ratio: If score drops by this ratio from the previous result, cut there.
                   E.g., 0.5 means "if score drops to less than 50% of the previous".

    Returns:
        Filtered results list (preserves order).
    """
    if not results or len(results) <= 1:
        return results

    min_results = min_results if min_results is not None else settings.compression_min_results
    score_floor = score_floor if score_floor is not None else settings.compression_score_floor
    gap_ratio = gap_ratio if gap_ratio is not None else settings.compression_gap_ratio

    # Ensure we have rerank_scores to work with
    if not any(r.get("rerank_score") for r in results):
        return results

    # Strategy 1: Score floor — remove clearly irrelevant chunks
    above_floor = []
    below_floor = []
    for r in results:
        rs = r.get("rerank_score", 0)
        if rs >= score_floor:
            above_floor.append(r)
        else:
            below_floor.append(r)

    # Strategy 2: Elbow detection — find the steepest relative score drop
    elbow_cut = len(above_floor)
    if len(above_floor) > min_results:
        for i in range(1, len(above_floor)):
            prev_score = above_floor[i - 1].get("rerank_score", 0)
            curr_score = above_floor[i].get("rerank_score", 0)

            if prev_score > 0 and curr_score / prev_score < gap_ratio:
                elbow_cut = max(i, min_results)
                break

    compressed = above_floor[:elbow_cut]

    # Guarantee: always return at least min_results
    if len(compressed) < min_results:
        # Backfill from below_floor (still ordered by rerank score)
        needed = min_results - len(compressed)
        compressed.extend(below_floor[:needed])

    if len(compressed) < len(results):
        removed = len(results) - len(compressed)
        scores_removed = [
            round(r.get("rerank_score", 0), 4)
            for r in results[len(compressed):]
        ]
        print(
            f"[hanna-oetp] Compression: {removed} chunks removed "
            f"(scores: {scores_removed[:5]}), {len(compressed)} kept"
        )

    return compressed
