"""Draft context builder: combines RAG results + style patterns + templates.

Single endpoint that gives Hanna everything she needs to write a draft
that matches colleague style and uses the right knowledge base context.

v2 — 2026-03-01: Template matching + category confidence + feedback diffs
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .style_learner import load_patterns, _categorize_email, _strip_quoted
from .skip_filter import check_skip
from ..rag import search as rag_search
from ..rag.references import resolve_references_in_results

# ─── Template System ──────────────────────────────────────────────────────────

TEMPLATE_PATH = "/app/data/response_templates.json"
FEEDBACK_DIFF_PATH = "/app/data/feedback_diffs.json"

_template_cache: dict | None = None
_template_cache_mtime: float = 0


def _load_templates() -> dict:
    """Load response templates with file-change caching."""
    global _template_cache, _template_cache_mtime

    p = Path(TEMPLATE_PATH)
    if not p.exists():
        return {}

    mtime = p.stat().st_mtime
    if _template_cache is not None and mtime == _template_cache_mtime:
        return _template_cache

    with open(p) as f:
        data = json.load(f)

    _template_cache = data.get("templates", {})
    _template_cache_mtime = mtime
    return _template_cache


def _match_template(category: str, email_text: str, email_subject: str) -> dict | None:
    """Find the best matching template for this email.

    Returns the template dict with match_score, or None if no match.
    """
    templates = _load_templates()
    if not templates:
        return None

    text_lower = (email_text + " " + email_subject).lower()
    best_match = None
    best_score = 0

    for tid, tmpl in templates.items():
        score = 0

        # Category match (strong signal)
        if category in tmpl.get("category_match", []):
            score += 3

        # Keyword match (additive)
        keywords = tmpl.get("keyword_match", [])
        matched_kw = [kw for kw in keywords if kw.lower() in text_lower]
        score += len(matched_kw) * 1.5

        if score > best_score and score >= 3:  # minimum threshold
            best_score = score
            best_match = {
                "template_id": tid,
                "template_name": tmpl.get("name", ""),
                "template_text": tmpl.get("template_text", ""),
                "variables": tmpl.get("variables", []),
                "variable_hints": tmpl.get("variable_hints", {}),
                "notes": tmpl.get("notes", ""),
                "word_count_target": tmpl.get("word_count", 0),
                "requires_system_action": tmpl.get("requires_system_action", False),
                "confidence_boost": tmpl.get("confidence_boost", 0),
                "match_score": round(best_score, 2),
                "matched_keywords": matched_kw,
            }

    return best_match


# ─── Category Confidence Thresholds ──────────────────────────────────────────

CATEGORY_CONFIDENCE_THRESHOLDS = {
    # RAG very reliable → lower threshold for "high"
    "inverter": {"high": 0.45, "medium": 0.30},
    "napelem": {"high": 0.45, "medium": 0.30},
    "szaldo": {"high": 0.45, "medium": 0.30},
    "jogosultsag": {"high": 0.50, "medium": 0.35},
    # RAG less reliable → higher threshold
    "ertesites_kau": {"high": 0.60, "medium": 0.45},
    "meghatalmazott": {"high": 0.65, "medium": 0.50},
    "altalanos": {"high": 0.55, "medium": 0.40},
    "hatarido": {"high": 0.55, "medium": 0.40},
    # Default
    "_default": {"high": 0.55, "medium": 0.40},
}


def _get_confidence_thresholds(category: str) -> dict:
    """Get confidence thresholds for a specific category."""
    return CATEGORY_CONFIDENCE_THRESHOLDS.get(
        category,
        CATEGORY_CONFIDENCE_THRESHOLDS["_default"],
    )


# ─── Feedback Diff Hints ─────────────────────────────────────────────────────

def _get_feedback_hints(category: str, max_hints: int = 3) -> list[dict]:
    """Load feedback diff hints for this category."""
    p = Path(FEEDBACK_DIFF_PATH)
    if not p.exists():
        return []

    try:
        with open(p) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    diffs = data.get("diffs", [])
    # Filter by category, newest first
    matching = [
        d for d in diffs
        if d.get("category", "") == category
    ]
    matching.sort(key=lambda d: d.get("created_at", ""), reverse=True)

    return [
        {
            "lesson": d.get("lesson", ""),
            "example_sent": d.get("sent_text", "")[:300],
            "category": d.get("category", ""),
            "similarity": d.get("similarity", 0),
        }
        for d in matching[:max_hints]
        if d.get("lesson") or d.get("sent_text")
    ]


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
    # 0. Skip filter — detect emails that don't need a Hanna draft
    skip_info = check_skip(email_text, email_subject)
    if skip_info["skip"]:
        return {
            "skip": True,
            "skip_reason": skip_info["reason"],
            "skip_category": skip_info["skip_category"],
            "rag_results": [],
            "referenced_chunks": [],
            "category": skip_info["skip_category"],
            "style_guide": {"available": False},
            "identifiers": {
                "oetp_ids": oetp_ids or [],
                "pod_numbers": pod_numbers or [],
                "has_identifiers": bool(oetp_ids or pod_numbers),
            },
            "needs_legal_context": {"should_consult_reka": False, "reason": "", "suggested_query": ""},
            "use_template": None,
            "confidence_thresholds": {},
            "suggested_confidence": "skip",
            "feedback_hints": [],
        }

    # 1. Detect category early (needed for dynamic authority in search)
    category = _categorize_email(email_subject, email_text)

    # 2. RAG search (with email_category for dynamic authority)
    rag_results = await rag_search.search_async(
        query=email_text[:2000],
        top_k=top_k,
        email_category=category,
    )

    # Cross-references
    ref_chunks = []
    try:
        raw_refs = resolve_references_in_results(rag_results, max_total_refs=5)
        ref_chunks = raw_refs
    except Exception:
        pass

    # 3. Load style patterns (category already detected in step 1)
    patterns = load_patterns() or {}
    
    # 4. Extract category-specific style guide
    style_guide = _build_style_guide(patterns, category)

    # 5. Determine if legal/business context is needed (→ suggest Réka consultation)
    needs_legal = _needs_legal_context(category, email_text, rag_results)

    # 6. Template matching
    matched_template = _match_template(category, email_text, email_subject)

    # 7. Category-specific confidence thresholds
    confidence_thresholds = _get_confidence_thresholds(category)

    # 8. Suggested confidence based on RAG score + template boost
    top_rag_score = rag_results[0].get("score", 0) if rag_results else 0

    suggested_confidence = "low"
    if top_rag_score >= confidence_thresholds["high"]:
        suggested_confidence = "high"
    elif top_rag_score >= confidence_thresholds["medium"]:
        suggested_confidence = "medium"

    # Template match boosts confidence
    if matched_template and suggested_confidence != "high":
        boost = matched_template.get("confidence_boost", 0)
        if top_rag_score + boost >= confidence_thresholds["high"]:
            suggested_confidence = "high"
        elif top_rag_score + boost >= confidence_thresholds["medium"]:
            suggested_confidence = "medium"

    # 9. Feedback diff hints
    feedback_hints = _get_feedback_hints(category)

    # 10. Similar past traces (reasoning memory)
    similar_traces = await _get_similar_traces(email_text, category)

    return {
        "skip": False,
        "skip_reason": None,
        "skip_category": None,
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
        # v2 fields:
        "use_template": matched_template,
        "confidence_thresholds": confidence_thresholds,
        "suggested_confidence": suggested_confidence,
        "feedback_hints": feedback_hints,
        "similar_traces": similar_traces,
    }


async def _get_similar_traces(email_text: str, category: str) -> list[dict]:
    """Find similar past reasoning traces for learning.

    Returns up to 3 traces with successful outcomes,
    formatted as context hints for draft generation.
    """
    import logging
    _log = logging.getLogger(__name__)

    try:
        import asyncpg
        from ..rag.embeddings import embed_query
        from ..reasoning.traces import find_similar_traces

        query_embedding = await embed_query(email_text[:500])
        if not query_embedding:
            return []

        conn = await asyncpg.connect(
            os.environ.get("HANNA_PG_DSN", "postgresql://klara:klara_docs_2026@hanna-db:5432/hanna_oetp")
        )
        try:
            traces = await find_similar_traces(
                conn=conn,
                query_embedding=query_embedding,
                limit=3,
                program="OETP",
                min_similarity=0.6,
            )
        finally:
            await conn.close()

        # Format for draft context
        hints = []
        for t in traces:
            hint = {
                "query": t.get("query_text", "")[:200],
                "outcome": t.get("outcome"),
                "confidence": t.get("confidence"),
                "cosine_sim": round(t.get("cosine_sim", 0), 3),
            }
            if t["outcome"] == "SENT_AS_IS":
                hint["successful_draft"] = (t.get("draft_text") or "")[:300]
            elif t["outcome"] == "SENT_MODIFIED":
                hint["original_draft"] = (t.get("draft_text") or "")[:200]
                hint["actual_sent"] = (t.get("sent_text") or "")[:200]
            hints.append(hint)

        return hints

    except Exception as e:
        _log.debug("Similar trace lookup failed (non-blocking): %s", e)
        return []


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
