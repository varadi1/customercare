"""
Dynamic authority weight learning from reasoning traces.

Analyzes which chunk_types lead to successful outcomes (SENT_AS_IS)
vs failures (REJECTED/SENT_MODIFIED) and computes per-category
authority adjustments.

The adjustments are additive corrections to the base authority weights
in rag/authority.py, bounded to [-0.15, +0.10] to prevent runaway drift.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any

import asyncpg

logger = logging.getLogger(__name__)

import os
PG_DSN = os.environ.get("HANNA_PG_DSN", "postgresql://klara:klara_docs_2026@hanna-db:5432/hanna_oetp")

# Bounds for authority adjustments
MAX_BOOST = 0.10
MAX_PENALTY = -0.15

# Minimum traces needed to compute adjustments
MIN_TRACES_FOR_LEARNING = 10

# In-memory cache (refreshed periodically)
_cached_adjustments: dict[str, dict[str, float]] = {}
_cache_timestamp: float = 0
_CACHE_TTL_SECONDS = 3600  # 1 hour


def get_cached_adjustments() -> dict[str, dict[str, float]]:
    """Return cached adjustments (may be empty if not yet computed)."""
    return _cached_adjustments


async def refresh_adjustments_cache(days: int = 30) -> dict[str, dict[str, float]]:
    """Recompute and cache authority adjustments."""
    global _cached_adjustments, _cache_timestamp
    import time
    _cached_adjustments = await compute_authority_adjustments(days=days)
    _cache_timestamp = time.time()
    logger.info("Authority adjustments cache refreshed: %d categories", len(_cached_adjustments))
    return _cached_adjustments


async def compute_authority_adjustments(days: int = 30) -> dict[str, dict[str, float]]:
    """Compute per-category authority weight adjustments from traces.

    Returns:
        {
            "inverter": {"email_reply": +0.05, "felhivas": -0.02},
            "szaldo": {"gyik": +0.08},
            ...
        }

    Positive = this chunk_type performs better than expected in this category.
    Negative = this chunk_type leads to more rejections in this category.
    """
    conn = await asyncpg.connect(PG_DSN)
    try:
        from datetime import datetime, timedelta
        since = datetime.utcnow() - timedelta(days=days)

        traces = await conn.fetch(
            """
            SELECT category, outcome, top_chunks
            FROM reasoning_traces
            WHERE outcome IN ('SENT_AS_IS', 'SENT_MODIFIED', 'REJECTED')
              AND top_chunks IS NOT NULL
              AND created_at >= $1
            """,
            since,
        )

        if len(traces) < MIN_TRACES_FOR_LEARNING:
            logger.info("Not enough traces for authority learning (%d < %d)",
                       len(traces), MIN_TRACES_FOR_LEARNING)
            return {}

        # Count chunk_type occurrences per category × outcome
        # {category: {chunk_type: {SENT_AS_IS: N, REJECTED: N, ...}}}
        stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        for trace in traces:
            category = trace["category"] or "altalanos"
            outcome = trace["outcome"]
            chunks = trace["top_chunks"]

            if isinstance(chunks, str):
                try:
                    chunks = json.loads(chunks)
                except (json.JSONDecodeError, TypeError):
                    continue

            if not isinstance(chunks, list):
                continue

            for chunk in chunks:
                chunk_type = chunk.get("chunk_type", "unknown")
                if chunk_type:
                    stats[category][chunk_type][outcome] += 1

        # Compute adjustments
        adjustments = {}
        for category, type_stats in stats.items():
            cat_adjustments = {}
            for chunk_type, outcomes in type_stats.items():
                total = sum(outcomes.values())
                if total < 3:  # too few observations
                    continue

                success_rate = outcomes.get("SENT_AS_IS", 0) / total
                reject_rate = outcomes.get("REJECTED", 0) / total

                # Adjustment: boost if high success, penalty if high rejection
                # Neutral at ~60% success rate
                adjustment = (success_rate - 0.6) * 0.25

                # Stronger penalty for rejections
                if reject_rate > 0.3:
                    adjustment -= reject_rate * 0.1

                # Clamp
                adjustment = max(MAX_PENALTY, min(MAX_BOOST, round(adjustment, 3)))

                if abs(adjustment) > 0.01:  # only store meaningful adjustments
                    cat_adjustments[chunk_type] = adjustment

            if cat_adjustments:
                adjustments[category] = cat_adjustments

        return adjustments

    finally:
        await conn.close()


async def update_chunk_survival_rates(days: int = 30) -> int:
    """Aggregate chunk survival data from feedback_analytics into chunks table.

    Returns the number of chunks updated.
    """
    conn = await asyncpg.connect(PG_DSN)
    try:
        from datetime import datetime, timedelta
        since = datetime.utcnow() - timedelta(days=days)

        # Fetch all chunk survival records
        rows = await conn.fetch(
            """
            SELECT chunk_survival
            FROM feedback_analytics
            WHERE chunk_survival IS NOT NULL
              AND chunk_survival != '[]'::jsonb
              AND created_at >= $1
            """,
            since,
        )

        if not rows:
            logger.info("No chunk survival data found for aggregation")
            return 0

        # Aggregate per chunk_id
        from collections import defaultdict
        stats: dict[str, dict] = defaultdict(lambda: {"survived": 0, "total": 0})

        for row in rows:
            survival = row["chunk_survival"]
            if isinstance(survival, str):
                try:
                    survival = json.loads(survival)
                except (json.JSONDecodeError, TypeError):
                    continue

            for entry in survival:
                cid = entry.get("chunk_id", "")
                if not cid:
                    continue
                stats[cid]["total"] += 1
                if entry.get("survived"):
                    stats[cid]["survived"] += 1

        # Update chunks table
        updated = 0
        for chunk_id, s in stats.items():
            if s["total"] < 1:
                continue
            rate = round(s["survived"] / s["total"], 3)
            result = await conn.execute(
                """
                UPDATE chunks
                SET survival_rate = $1, survival_count = $2, appearance_count = $3
                WHERE id = $4
                """,
                rate, s["survived"], s["total"], chunk_id,
            )
            if "UPDATE 1" in result:
                updated += 1

        logger.info("Updated survival rates for %d chunks (from %d analytics records)", updated, len(rows))
        return updated

    finally:
        await conn.close()


def apply_learned_adjustments(
    results: list[dict],
    category: str,
    adjustments: dict[str, dict[str, float]],
) -> list[dict]:
    """Apply learned authority adjustments to search results.

    Called after base authority weighting. Modifies scores in-place.

    Args:
        results: Search results (already authority-weighted)
        category: Detected email category
        adjustments: Output of compute_authority_adjustments()

    Returns:
        Results with adjusted scores.
    """
    cat_adj = adjustments.get(category)
    if not cat_adj:
        return results

    for doc in results:
        chunk_type = doc.get("chunk_type", "")
        adj = cat_adj.get(chunk_type, 0)
        if adj != 0:
            old_score = doc.get("score", 0)
            doc["score"] = round(old_score + adj, 4)
            doc["authority_learned_adj"] = adj

    # Re-sort
    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return results


def format_adjustments_report(adjustments: dict[str, dict[str, float]]) -> str:
    """Format adjustments as readable report."""
    if not adjustments:
        return "Nincs elég adat az authority weight tanuláshoz.\n"

    lines = ["# Authority Weight Adjustments (learned)", ""]
    for category, adj in sorted(adjustments.items()):
        lines.append(f"## {category}")
        for chunk_type, value in sorted(adj.items(), key=lambda x: x[1], reverse=True):
            sign = "+" if value > 0 else ""
            emoji = "↑" if value > 0 else "↓"
            lines.append(f"  {emoji} {chunk_type}: {sign}{value:.3f}")
        lines.append("")

    return "\n".join(lines)
