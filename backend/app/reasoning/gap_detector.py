"""
Knowledge base gap detection from feedback analytics.

Level 4 of the learning system:
  1. extract_human_additions() — gather content humans added to drafts
  2. cluster_additions() — group similar additions by embedding similarity
  3. suggest_new_chunks() — check if clusters are covered by existing chunks
  4. format_gap_detection_report() — Markdown report
"""
from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from typing import Any

import asyncpg

logger = logging.getLogger(__name__)

PG_DSN = os.environ.get(
    "HANNA_PG_DSN",
    "postgresql://klara:klara_docs_2026@hanna-db:5432/hanna_oetp",
)


# ─── Extract Human Additions ─────────────────────────────────────────────────

async def extract_human_additions(days: int = 30) -> list[dict]:
    """Extract content that humans added to drafts (not present in original).

    Groups by category and returns list of {category, additions: list[str], count}.
    """
    conn = await asyncpg.connect(PG_DSN)
    try:
        from datetime import datetime, timedelta
        since = datetime.utcnow() - timedelta(days=days)

        rows = await conn.fetch(
            """
            SELECT fa.added_content, fa.change_types, fa.gap_topics,
                   rt.category
            FROM feedback_analytics fa
            JOIN reasoning_traces rt ON fa.trace_id = rt.id
            WHERE fa.added_content IS NOT NULL
              AND LENGTH(fa.added_content) > 20
              AND fa.created_at >= $1
            ORDER BY fa.created_at DESC
            """,
            since,
        )

        if not rows:
            return []

        # Group by category
        by_category: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            cat = row["category"] or "altalanos"
            by_category[cat].append(row["added_content"])

        return [
            {"category": cat, "additions": additions, "count": len(additions)}
            for cat, additions in sorted(by_category.items(), key=lambda x: len(x[1]), reverse=True)
        ]

    finally:
        await conn.close()


# ─── Cluster Similar Additions ───────────────────────────────────────────────

async def cluster_additions(
    additions: list[dict],
    min_cluster_size: int = 3,
    similarity_threshold: float = 0.7,
) -> list[dict]:
    """Cluster similar human additions by embedding similarity.

    Uses simple pairwise comparison with local BGE-M3 embeddings.
    Returns clusters with topic summaries.

    Args:
        additions: Output of extract_human_additions()
        min_cluster_size: Minimum additions to form a cluster
        similarity_threshold: Cosine similarity threshold for clustering

    Returns:
        [{topic_summary, examples: list[str], count, category}]
    """
    # Flatten all additions
    all_items: list[tuple[str, str]] = []  # (text, category)
    for entry in additions:
        cat = entry["category"]
        for text in entry["additions"]:
            if len(text.strip()) > 20:
                all_items.append((text.strip()[:500], cat))

    if len(all_items) < min_cluster_size:
        return []

    # Embed all additions (sync call, runs via httpx to local BGE-M3)
    try:
        from ..rag.embeddings import embed_texts
        texts = [item[0] for item in all_items]
        embeddings = embed_texts(texts)
        if not embeddings or len(embeddings) != len(texts):
            logger.warning("Embedding batch failed, skipping clustering")
            return []
    except Exception as e:
        logger.warning("Failed to embed additions for clustering: %s", e)
        return []

    # Simple greedy clustering by cosine similarity
    import numpy as np
    emb_array = np.array(embeddings)
    norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
    norms[norms == 0] = 1
    emb_normed = emb_array / norms

    assigned = [False] * len(all_items)
    clusters: list[dict] = []

    for i in range(len(all_items)):
        if assigned[i]:
            continue

        # Find all similar items
        sims = emb_normed @ emb_normed[i]
        cluster_indices = [i]
        assigned[i] = True

        for j in range(i + 1, len(all_items)):
            if not assigned[j] and sims[j] >= similarity_threshold:
                cluster_indices.append(j)
                assigned[j] = True

        if len(cluster_indices) >= min_cluster_size:
            examples = [all_items[idx][0] for idx in cluster_indices[:5]]
            categories = list(set(all_items[idx][1] for idx in cluster_indices))

            # Generate topic summary from the shortest example (most specific)
            topic = min(examples, key=len)[:100]

            clusters.append({
                "topic_summary": topic,
                "examples": examples,
                "count": len(cluster_indices),
                "categories": categories,
            })

    # Sort by cluster size
    clusters.sort(key=lambda x: x["count"], reverse=True)

    # Generate better topic summaries using LLM (for top clusters)
    if clusters:
        try:
            from ..llm_client import chat_completion
            for cluster in clusters[:5]:
                examples_text = "\n".join(f"- {ex[:200]}" for ex in cluster["examples"][:3])
                result = await chat_completion(
                    messages=[
                        {"role": "system", "content": "Adj egy 5-10 szavas osszefoglalast az alabbi szovegek kozos temajara. Csak a tema nevet add meg, semmi mast."},
                        {"role": "user", "content": examples_text},
                    ],
                    temperature=0,
                    max_tokens=30,
                )
                cluster["topic_summary"] = result["content"].strip().strip('"')
        except Exception as e:
            logger.warning("Topic summary generation failed: %s", e)

    return clusters


# ─── Suggest New Chunks ──────────────────────────────────────────────────────

async def suggest_new_chunks(
    clusters: list[dict],
    coverage_threshold: float = 0.6,
) -> list[dict]:
    """Check if clusters represent truly missing knowledge.

    For each cluster, search existing chunks. If no chunk is similar enough,
    suggest it as missing knowledge.

    Args:
        clusters: Output of cluster_additions()
        coverage_threshold: If best chunk similarity < this, it's a gap

    Returns:
        [{suggested_topic, evidence_count, example_texts, categories, best_existing_sim}]
    """
    if not clusters:
        return []

    suggestions = []

    try:
        from ..rag.embeddings import embed_query

        conn = await asyncpg.connect(PG_DSN)
        try:
            for cluster in clusters[:10]:
                # Use first example as representative
                representative = cluster["examples"][0]
                emb = embed_query(representative[:500])
                if not emb:
                    continue

                from ..reasoning.feedback_analytics import _text_overlap

                # Search for similar existing chunks
                emb_str = "[" + ",".join(str(x) for x in emb) + "]"
                rows = await conn.fetch(
                    """
                    SELECT id, content, doc_type,
                           1 - (embedding <=> $1::vector) AS cosine_sim
                    FROM chunks
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> $1::vector
                    LIMIT 3
                    """,
                    emb_str,
                )

                best_sim = max((r["cosine_sim"] for r in rows), default=0)

                if best_sim < coverage_threshold:
                    suggestions.append({
                        "suggested_topic": cluster["topic_summary"],
                        "evidence_count": cluster["count"],
                        "example_texts": cluster["examples"][:3],
                        "categories": cluster.get("categories", []),
                        "best_existing_sim": round(best_sim, 3),
                    })

        finally:
            await conn.close()

    except Exception as e:
        logger.warning("suggest_new_chunks failed: %s", e)

    return suggestions


# ─── Report ──────────────────────────────────────────────────────────────────

def format_gap_detection_report(
    clusters: list[dict],
    suggestions: list[dict],
) -> str:
    """Format gap detection results as Markdown."""
    lines = ["## Hiányzó tudás detekció", ""]

    if not clusters:
        lines.append("Nincs elegendő feedback adat a klaszterezéshez.")
        return "\n".join(lines)

    lines.append(f"**{len(clusters)} klaszter** azonosítva az emberi bővítésekből:")
    lines.append("")

    for i, cluster in enumerate(clusters[:10], 1):
        lines.append(f"### {i}. {cluster['topic_summary']} ({cluster['count']} eset)")
        lines.append(f"Kategóriák: {', '.join(cluster.get('categories', []))}")
        for ex in cluster["examples"][:2]:
            lines.append(f"> {ex[:150]}...")
        lines.append("")

    if suggestions:
        lines.extend(["### Javasolt új chunk-ok", ""])
        for s in suggestions:
            lines.append(
                f"- **{s['suggested_topic']}** — {s['evidence_count']}x emberi hozzáadás, "
                f"legjobb meglévő sim: {s['best_existing_sim']:.2f}"
            )
            for cat in s.get("categories", []):
                lines.append(f"  Kategória: {cat}")
    else:
        lines.append("Minden klaszter lefedett a meglévő tudásbázissal.")

    return "\n".join(lines)
