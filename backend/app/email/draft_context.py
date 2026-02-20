"""Draft context builder: combines RAG results + style patterns + templates.

Single endpoint that gives Hanna everything she needs to write a draft
that matches colleague style and uses the right knowledge base context.
"""

from __future__ import annotations

import re
from typing import Any

from .style_learner import load_patterns, _categorize_email, _strip_quoted
from ..rag import search as rag_search
from ..rag.references import resolve_references_in_results


async def build_draft_context(
    email_text: str,
    email_subject: str = "",
    oetp_ids: list[str] | None = None,
    pod_numbers: list[str] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Build full context for draft generation.

    Returns:
        - rag_results: top RAG search results
        - referenced_chunks: cross-referenced sections
        - category: detected email category
        - style_guide: style patterns for this category
        - suggested_greeting: most common greeting
        - suggested_closing: most common closing
        - word_count_target: recommended word count range
        - tone_tips: tone recommendations based on data
        - category_examples: real colleague response examples for this category
        - identifiers: extracted OETP/POD identifiers
    """
    # 1. RAG search
    rag_results = await rag_search.search_async(
        query=email_text[:2000],  # truncate for search
        top_k=top_k,
    )

    # Cross-references
    ref_chunks = []
    try:
        raw_refs = resolve_references_in_results(rag_results, max_total_refs=5)
        ref_chunks = raw_refs
    except Exception:
        pass

    # 2. Detect category
    category = _categorize_email(email_subject, email_text)

    # 3. Load style patterns
    patterns = load_patterns() or {}
    
    # 4. Extract category-specific style guide
    style_guide = _build_style_guide(patterns, category)

    # 5. Build response
    # Determine if legal/business context is needed (→ suggest Réka consultation)
    needs_legal = _needs_legal_context(category, email_text, rag_results)

    return {
        "rag_results": [
            {
                "text": r.get("text", "")[:500],
                "source": r.get("source", ""),
                "score": r.get("score", 0),
                "chunk_type": r.get("chunk_type", ""),
                "authority_weight": r.get("authority_weight"),
            }
            for r in rag_results
        ],
        "referenced_chunks": [
            {
                "text": r.get("text", "")[:500],
                "source": r.get("source", ""),
                "ref_type": r.get("ref_type", ""),
                "ref_section": r.get("ref_section", ""),
            }
            for r in ref_chunks
        ],
        "category": category,
        "style_guide": style_guide,
        "identifiers": {
            "oetp_ids": oetp_ids or [],
            "pod_numbers": pod_numbers or [],
            "has_identifiers": bool(oetp_ids or pod_numbers),
        },
        "needs_legal_context": needs_legal,
    }


def _needs_legal_context(category: str, email_text: str, rag_results: list) -> dict:
    """Determine if Réka (legal RAG) should be consulted."""
    text_lower = email_text.lower()

    legal_keywords = [
        "vállalkozás", "vállalkozó", "egyéni vállalkozás", "kft", "bt", "zrt",
        "gazdasági tevékenység", "székhelye", "telephelye", "fióktelepe",
        "de minimis", "gber", "állami támogatás", "közbeszerzés",
        "jogszabály", "rendelet", "törvény", "közlemény",
        "adózás", "áfa", "szja", "társasági adó",
        "örökl", "haszonélvez", "tulajdonjog", "végrehajtás",
    ]

    matched_keywords = [kw for kw in legal_keywords if kw in text_lower]
    legal_categories = {"gazdasági_tevékenység", "jogosultsag"}
    
    # Check if RAG results reference EU documents
    has_eu_source = any(
        "EU_Bizottsag" in r.get("source", "") or "eu_rendelet" in r.get("chunk_type", "")
        for r in rag_results
    )

    should_consult = (
        category in legal_categories
        or len(matched_keywords) >= 2
        or has_eu_source
    )

    return {
        "should_consult_reka": should_consult,
        "reason": f"Keywords: {matched_keywords[:5]}" if matched_keywords else ("EU source" if has_eu_source else "category match" if category in legal_categories else ""),
        "suggested_query": email_text[:300] if should_consult else "",
    }


def _build_style_guide(patterns: dict, category: str) -> dict[str, Any]:
    """Extract actionable style guide from patterns for a specific category."""
    if not patterns:
        return {
            "available": False,
            "message": "No style patterns yet. Run /style/analyze first.",
        }

    word_stats = patterns.get("word_count", {})
    tone = patterns.get("tone", {})
    greetings = patterns.get("top_greetings", [])
    closings = patterns.get("top_closings", [])
    category_examples = patterns.get("category_examples", {})

    # Get examples for this specific category
    examples = category_examples.get(category, [])
    
    # Fallback to "altalanos" if no category-specific examples
    if not examples:
        examples = category_examples.get("altalanos", [])

    # Build tone tips
    tone_tips = []
    if tone.get("uses_conditional_pct", 0) > 30:
        tone_tips.append("Használj feltételes módot: 'amennyiben', 'abban az esetben'")
    if tone.get("uses_polite_request_pct", 0) > 30:
        tone_tips.append("Udvarias kérés: 'kérjük', 'szíveskedjen'")
    if tone.get("uses_direct_pct", 0) > 20:
        tone_tips.append("Direkt megfogalmazás is OK: 'szükséges', 'kell', 'feltölteni'")
    if tone.get("has_list_pct", 0) < 10:
        tone_tips.append("Ritkán használnak listát — folyó szöveget írj")

    return {
        "available": True,
        "suggested_greeting": greetings[0][0] if greetings else "Tisztelt Pályázó!",
        "suggested_closing": "Üdvözlettel:",
        "word_count_target": {
            "min": word_stats.get("p25", 30),
            "ideal": word_stats.get("median", 63),
            "max": word_stats.get("p75", 80),
        },
        "tone_tips": tone_tips,
        "category_examples": [
            {
                "subject": ex.get("subject", ""),
                "text": ex.get("text", "")[:400],
                "word_count": ex.get("word_count", 0),
            }
            for ex in examples[:3]
        ],
        "tone_stats": {
            "conditional_pct": tone.get("uses_conditional_pct", 0),
            "polite_request_pct": tone.get("uses_polite_request_pct", 0),
            "direct_pct": tone.get("uses_direct_pct", 0),
        },
    }
