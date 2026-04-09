"""
Knowledge gap detection — identifies topics where Hanna struggles.

Analyzes reasoning_traces to find:
1. REJECTED traces (colleague completely rewrote the answer)
2. Low confidence traces (Hanna wasn't sure)
3. Recurring unanswered topics (same category, multiple failures)

Output: structured report for Obsidian !inbox/!reports/
"""
from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timedelta
from typing import Any

import asyncpg

logger = logging.getLogger(__name__)

import os
PG_DSN = os.environ.get("HANNA_PG_DSN", "postgresql://klara:klara_docs_2026@cc-db:5432/customercare")


async def generate_gap_report(days: int = 7) -> dict[str, Any]:
    """Generate knowledge gap report from recent reasoning traces.

    Args:
        days: Look back period (default 7 days)

    Returns:
        Structured report with gap categories, examples, and recommendations.
    """
    conn = await asyncpg.connect(PG_DSN)
    try:
        since = datetime.utcnow() - timedelta(days=days)

        # All traces in period
        all_traces = await conn.fetch(
            """
            SELECT id, query_text, category, confidence, outcome,
                   similarity_score, draft_text, sent_text, created_at
            FROM reasoning_traces
            WHERE created_at >= $1
            ORDER BY created_at DESC
            """,
            since,
        )

        total = len(all_traces)
        if total == 0:
            return {"status": "no_data", "period_days": days, "total_traces": 0}

        # Categorize outcomes
        rejected = [t for t in all_traces if t["outcome"] == "REJECTED"]
        modified = [t for t in all_traces if t["outcome"] == "SENT_MODIFIED"]
        sent_as_is = [t for t in all_traces if t["outcome"] == "SENT_AS_IS"]
        low_conf = [t for t in all_traces if t["confidence"] == "low"]

        # Category breakdown for problems
        problem_traces = rejected + [t for t in modified if (t["similarity_score"] or 0) < 0.5]
        category_counts = Counter(t["category"] for t in problem_traces if t["category"])
        top_problem_categories = category_counts.most_common(10)

        # Extract example queries for each problem category
        category_examples = {}
        for cat, count in top_problem_categories:
            examples = [
                {
                    "query": t["query_text"][:150] if t["query_text"] else "",
                    "outcome": t["outcome"],
                    "similarity": round(t["similarity_score"] or 0, 2),
                    "confidence": t["confidence"],
                }
                for t in problem_traces
                if t["category"] == cat
            ][:5]
            category_examples[cat] = examples

        # Success rate
        resolved = [t for t in all_traces if t["outcome"] not in ("PENDING", None)]
        success_rate = len(sent_as_is) / len(resolved) if resolved else 0

        # Confidence distribution
        conf_dist = Counter(t["confidence"] for t in all_traces if t["confidence"])

        report = {
            "status": "ok",
            "period_days": days,
            "period_start": since.isoformat(),
            "period_end": datetime.utcnow().isoformat(),
            "total_traces": total,
            "outcomes": {
                "SENT_AS_IS": len(sent_as_is),
                "SENT_MODIFIED": len(modified),
                "REJECTED": len(rejected),
                "PENDING": total - len(resolved),
            },
            "success_rate": round(success_rate, 3),
            "confidence_distribution": dict(conf_dist),
            "problem_categories": [
                {"category": cat, "count": count, "examples": category_examples.get(cat, [])}
                for cat, count in top_problem_categories
            ],
            "low_confidence_count": len(low_conf),
            "recommendations": _generate_recommendations(
                top_problem_categories, success_rate, len(rejected), len(low_conf), total,
            ),
        }

        # ── Gap detection from feedback analytics (Level 4) ──
        try:
            from .gap_detector import (
                extract_human_additions, cluster_additions,
                suggest_new_chunks, format_gap_detection_report,
            )
            additions = await extract_human_additions(days=days)
            if additions:
                from ..config import settings
                clusters = await cluster_additions(
                    additions,
                    min_cluster_size=settings.gap_detection_min_cluster,
                    similarity_threshold=settings.gap_detection_similarity_threshold,
                )
                suggestions = await suggest_new_chunks(clusters)
                report["gap_clusters"] = clusters
                report["gap_suggestions"] = suggestions
                report["gap_detection_report"] = format_gap_detection_report(clusters, suggestions)
        except Exception as e:
            logger.warning("Gap detection failed (non-fatal): %s", e)
            report["gap_detection_error"] = str(e)

        return report

    finally:
        await conn.close()


def _generate_recommendations(
    top_problems: list[tuple[str, int]],
    success_rate: float,
    rejected_count: int,
    low_conf_count: int,
    total: int,
) -> list[str]:
    """Generate actionable recommendations based on gap analysis."""
    recs = []

    if success_rate < 0.5:
        recs.append("KRITIKUS: A sikeres válaszok aránya 50% alatt van. "
                     "A tudásbázis jelentős bővítésre szorul.")
    elif success_rate < 0.7:
        recs.append("A sikeres válaszok aránya 70% alatt van. "
                     "Fókuszált bővítés szükséges a problémás kategóriákban.")

    for cat, count in top_problems[:3]:
        if count >= 3:
            recs.append(f"'{cat}' kategóriában {count} problémás válasz — "
                        f"új dokumentumok feltöltése javasolt ehhez a témához.")

    if rejected_count > total * 0.2:
        recs.append(f"A válaszok {rejected_count}/{total} ({rejected_count/total:.0%}) "
                    f"esetben teljesen átíródott. A draft minőség javítása szükséges.")

    if low_conf_count > total * 0.3:
        recs.append(f"Az alacsony konfidenciájú válaszok aránya magas ({low_conf_count}/{total}). "
                    f"Több autoritatív forrás (felhívás, melléklet) szükséges.")

    if not recs:
        recs.append("A rendszer jól teljesít. Nincs azonnali beavatkozás szükséges.")

    return recs


def format_obsidian_report(report: dict[str, Any]) -> str:
    """Format gap report as Obsidian markdown."""
    if report.get("status") == "no_data":
        return "# Hanna Knowledge Gap Report\n\nNincs adat a megadott időszakban.\n"

    lines = [
        f"# Hanna Knowledge Gap Report",
        f"",
        f"**Időszak:** {report['period_days']} nap ({report['period_start'][:10]} — {report['period_end'][:10]})",
        f"**Összes interakció:** {report['total_traces']}",
        f"**Sikeres válasz arány:** {report['success_rate']:.1%}",
        f"",
        f"## Outcome eloszlás",
        f"",
    ]

    outcomes = report.get("outcomes", {})
    for outcome, count in outcomes.items():
        pct = count / report["total_traces"] * 100 if report["total_traces"] else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        lines.append(f"- **{outcome}**: {count} ({pct:.0f}%) {bar}")

    lines.extend(["", "## Problémás kategóriák", ""])

    for item in report.get("problem_categories", []):
        lines.append(f"### {item['category']} ({item['count']} eset)")
        for ex in item.get("examples", []):
            lines.append(f"- _{ex['query'][:100]}_ → {ex['outcome']} (sim: {ex['similarity']})")
        lines.append("")

    lines.extend(["## Javaslatok", ""])
    for rec in report.get("recommendations", []):
        lines.append(f"- {rec}")

    # Gap detection section (Level 4)
    gap_report = report.get("gap_detection_report", "")
    if gap_report:
        lines.extend(["", gap_report])

    lines.extend([
        "",
        "---",
        f"_Generálva: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC_",
    ])

    return "\n".join(lines)
