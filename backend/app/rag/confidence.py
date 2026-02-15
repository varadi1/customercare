"""Multi-factor confidence calculation for RAG results."""

from __future__ import annotations

from datetime import datetime, timedelta


def calculate_confidence(
    results: list[dict],
    template_match: tuple | None = None,
) -> str:
    """Calculate confidence level based on multiple factors.

    Args:
        results: RAG search results with score, chunk_type, metadata
        template_match: (template_key, score) or None

    Returns:
        "high", "medium", or "low"
    """
    # Factor 1: Template match
    if template_match and template_match[0] is not None:
        _, tmpl_score = template_match
        if tmpl_score > 0.8:
            return "high"

    if not results:
        return "low"

    top_score = results[0].get("score", 0) if results else 0

    # Factor 2: Authoritative source with good score
    for r in results[:3]:
        chunk_type = r.get("chunk_type", "")
        score = r.get("score", 0)
        if chunk_type in ("palyazat_felhivas", "gyik", "faq", "palyazat_melleklet") and score > 0.45:
            return "high"

    # Factor 3: Only email_reply support, no document backing
    has_document = any(
        r.get("chunk_type", "") not in ("email_reply", "")
        for r in results[:5]
        if r.get("score", 0) > 0.40
    )
    if not has_document:
        # Check if email_reply is decent
        for r in results[:3]:
            if r.get("chunk_type") == "email_reply" and r.get("score", 0) > 0.55:
                # Freshness penalty: check indexed_at
                meta = r.get("metadata", {})
                indexed_at = meta.get("indexed_at", "")
                if indexed_at:
                    try:
                        idx_date = datetime.fromisoformat(indexed_at)
                        if datetime.now() - idx_date > timedelta(days=30):
                            return "low"  # stale email reply
                    except (ValueError, TypeError):
                        pass
                return "medium"

    # Factor 4: Low top score
    if top_score < 0.40:
        return "low"

    # Default: medium if we have some results with decent scores
    if top_score >= 0.45:
        return "high" if has_document else "medium"

    return "medium"
