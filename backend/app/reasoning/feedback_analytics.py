"""
Feedback analytics — extract lessons from draft-vs-sent differences.

Level 1 of Hanna's learning system:
  1. categorize_changes() — LLM classifies WHAT the colleague changed
  2. compute_chunk_survival() — which RAG chunks survived into the sent email
  3. export_pair_to_langfuse() — dataset building for DSPy optimization
  4. store_analytics() — persist to feedback_analytics table
"""
from __future__ import annotations

import json
import logging
import os
from difflib import SequenceMatcher
from typing import Any

import asyncpg

from ..llm_client import chat_completion

logger = logging.getLogger(__name__)

PG_DSN = os.environ.get(
    "HANNA_PG_DSN",
    "postgresql://klara:klara_docs_2026@hanna-db:5432/hanna_oetp",
)

# ─── Change Categorization (LLM) ────────────────────────────────────────────

CATEGORIZE_SYSTEM = """\
Te egy ügyfélszolgálati minőségellenőr vagy. Két verziót kapsz: egy AI-draft-ot és a kolléga által ténylegesen elküldött verziót.

Elemezd a különbségeket és kategorizáld a változtatásokat.

VÁLASZ (JSON):
{
  "change_types": ["tone_change", "fact_correction", "structure_change", "missing_info_added", "info_removed", "style_adjustment"],
  "lesson": "Rövid tanulság, max 2 mondat, amit a rendszer megtanulhat.",
  "added_content": "Az a szöveg amit a kolléga HOZZÁADOTT (amit a draft NEM tartalmazott). Üres ha nincs.",
  "removed_content": "Az a szöveg amit a kolléga TÖRÖLT a draftból. Üres ha nincs."
}

change_types LEHETSÉGES ÉRTÉKEK (csak a relevánsakat add meg):
- tone_change: hangnem változott (pl. formálisabb/barátságosabb)
- fact_correction: ténybeli javítás (szám, dátum, feltétel)
- structure_change: struktúra/sorrend átrendezés
- missing_info_added: kolléga új információt adott hozzá (amit a RAG nem talált)
- info_removed: kolléga eltávolított információt a draftból
- style_adjustment: fogalmazásbeli finomítás (szóhasználat, mondatszerkezet)
"""


async def categorize_changes(
    draft_text: str,
    sent_text: str,
    category: str = "",
) -> dict[str, Any]:
    """Use LLM to classify what the colleague changed between draft and sent.

    Returns dict with change_types, lesson, added_content, removed_content.
    Cost: ~$0.005 per call (gpt-4o-mini, 300 tokens).
    """
    if not draft_text or not sent_text:
        return {"change_types": [], "lesson": "", "added_content": "", "removed_content": ""}

    user_msg = (
        f"KATEGÓRIA: {category or 'általános'}\n\n"
        f"--- AI DRAFT ---\n{draft_text[:1500]}\n\n"
        f"--- ELKÜLDÖTT VERZIÓ ---\n{sent_text[:1500]}"
    )

    try:
        result = await chat_completion(
            messages=[
                {"role": "system", "content": CATEGORIZE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            max_tokens=400,
            json_mode=True,
        )
        parsed = json.loads(result["content"])
        return {
            "change_types": parsed.get("change_types", []),
            "lesson": parsed.get("lesson", ""),
            "added_content": parsed.get("added_content", ""),
            "removed_content": parsed.get("removed_content", ""),
        }
    except Exception as e:
        logger.warning("categorize_changes failed: %s", e)
        return {"change_types": [], "lesson": f"Error: {e}", "added_content": "", "removed_content": ""}


# ─── Chunk Survival (no LLM) ────────────────────────────────────────────────

def _text_overlap(chunk_text: str, target_text: str) -> float:
    """Compute what fraction of chunk_text appears in target_text (0-1)."""
    if not chunk_text or not target_text:
        return 0.0
    # Use SequenceMatcher to find matching blocks
    sm = SequenceMatcher(None, chunk_text.lower(), target_text.lower())
    matching_chars = sum(block.size for block in sm.get_matching_blocks())
    return matching_chars / len(chunk_text) if chunk_text else 0.0


async def compute_chunk_survival(
    draft_text: str,
    sent_text: str,
    top_chunks: list[dict],
) -> list[dict]:
    """Compute which RAG chunks 'survived' into the sent email.

    For each chunk, measures text overlap with draft vs sent.
    A chunk 'survived' if its overlap with sent_text is >= 50% of its overlap with draft_text.

    Args:
        draft_text: Hanna's generated draft (plain text)
        sent_text: Actually sent email (plain text)
        top_chunks: List of chunk dicts with at least 'id' and 'chunk_type'.
                    If 'text' key exists, uses it. Otherwise queries DB.

    Returns:
        List of {chunk_id, chunk_type, survived, draft_overlap, sent_overlap}
    """
    if not top_chunks:
        return []

    # Fetch chunk texts from DB if not provided
    chunk_texts = {}
    chunks_needing_text = [c for c in top_chunks if not c.get("text")]
    if chunks_needing_text:
        chunk_ids = [c["id"] for c in chunks_needing_text if c.get("id")]
        if chunk_ids:
            try:
                conn = await asyncpg.connect(PG_DSN)
                try:
                    rows = await conn.fetch(
                        "SELECT id, content FROM chunks WHERE id = ANY($1)",
                        chunk_ids,
                    )
                    chunk_texts = {r["id"]: r["content"] or "" for r in rows}
                finally:
                    await conn.close()
            except Exception as e:
                logger.warning("Failed to fetch chunk texts: %s", e)

    results = []
    for chunk in top_chunks:
        cid = chunk.get("id", "")
        ctype = chunk.get("chunk_type", "")
        text = chunk.get("text", "") or chunk_texts.get(cid, "")

        if not text:
            results.append({
                "chunk_id": cid, "chunk_type": ctype,
                "survived": False, "draft_overlap": 0.0, "sent_overlap": 0.0,
            })
            continue

        draft_overlap = _text_overlap(text, draft_text)
        sent_overlap = _text_overlap(text, sent_text)

        # Survived if sent overlap is at least 50% of draft overlap,
        # or if sent overlap is significant on its own (>= 0.15)
        survived = (
            (draft_overlap > 0.05 and sent_overlap >= draft_overlap * 0.5)
            or sent_overlap >= 0.15
        )

        results.append({
            "chunk_id": cid,
            "chunk_type": ctype,
            "survived": survived,
            "draft_overlap": round(draft_overlap, 3),
            "sent_overlap": round(sent_overlap, 3),
        })

    return results


# ─── Langfuse Dataset Export ─────────────────────────────────────────────────

async def export_pair_to_langfuse(
    draft_text: str,
    sent_text: str,
    metadata: dict,
    dataset_name: str = "hanna-draft-pairs",
) -> bool:
    """Export a draft-sent pair to Langfuse dataset for DSPy training.

    Returns True if successful.
    """
    try:
        from ..observability import _get_langfuse
        lf = _get_langfuse()
        if not lf:
            return False

        # Ensure dataset exists
        try:
            lf.get_dataset(dataset_name)
        except Exception:
            lf.create_dataset(name=dataset_name)

        lf.create_dataset_item(
            dataset_name=dataset_name,
            input={
                "email_text": metadata.get("query_text", "")[:2000],
                "subject": metadata.get("subject", ""),
                "category": metadata.get("category", ""),
                "confidence": metadata.get("confidence", ""),
                "top_chunks": metadata.get("top_chunks", []),
            },
            expected_output={
                "body": sent_text[:2000],
            },
            metadata={
                "draft_text": draft_text[:1000],
                "similarity": metadata.get("similarity", 0),
                "outcome": metadata.get("outcome", ""),
                "match_method": metadata.get("match_method", ""),
            },
        )
        lf.flush()
        logger.info("Exported pair to Langfuse dataset '%s': %s", dataset_name, metadata.get("subject", "")[:40])
        return True
    except Exception as e:
        logger.warning("Langfuse export failed: %s", e)
        return False


# ─── Persistence ─────────────────────────────────────────────────────────────

async def store_analytics(
    trace_id: int,
    change_types: list[str],
    lesson: str,
    added_content: str,
    removed_content: str,
    chunk_survival: list[dict],
    gap_topics: list[str] | None = None,
) -> int | None:
    """Store feedback analytics to the database.

    Returns the analytics record ID, or None on failure.
    """
    try:
        conn = await asyncpg.connect(PG_DSN)
        try:
            row = await conn.fetchrow(
                """
                INSERT INTO feedback_analytics
                    (trace_id, change_types, lesson, added_content, removed_content,
                     chunk_survival, gap_topics)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)
                ON CONFLICT DO NOTHING
                RETURNING id
                """,
                trace_id,
                change_types,
                lesson,
                added_content[:2000] if added_content else "",
                removed_content[:2000] if removed_content else "",
                json.dumps(chunk_survival, ensure_ascii=False) if chunk_survival else "[]",
                gap_topics or [],
            )
            if row:
                logger.info("Stored feedback analytics for trace %d", trace_id)
                return row["id"]
            return None
        finally:
            await conn.close()
    except Exception as e:
        logger.warning("Failed to store feedback analytics: %s", e)
        return None


# ─── Run Full Analytics Pipeline ─────────────────────────────────────────────

async def run_analytics_for_feedback(
    trace_id: int,
    draft_text: str,
    sent_text: str,
    category: str,
    top_chunks: list[dict],
    metadata: dict,
) -> dict[str, Any] | None:
    """Run the full analytics pipeline for a single feedback entry.

    Calls categorize_changes, compute_chunk_survival, export_pair_to_langfuse,
    and store_analytics. Returns the combined result or None on failure.
    """
    # 1. Categorize changes (LLM)
    changes = await categorize_changes(draft_text, sent_text, category)

    # 2. Compute chunk survival (no LLM)
    survival = await compute_chunk_survival(draft_text, sent_text, top_chunks)

    # 3. Export to Langfuse
    metadata["similarity"] = metadata.get("similarity", 0)
    await export_pair_to_langfuse(draft_text, sent_text, metadata)

    # 4. Store to DB
    await store_analytics(
        trace_id=trace_id,
        change_types=changes["change_types"],
        lesson=changes["lesson"],
        added_content=changes["added_content"],
        removed_content=changes["removed_content"],
        chunk_survival=survival,
    )

    return {**changes, "chunk_survival": survival}
