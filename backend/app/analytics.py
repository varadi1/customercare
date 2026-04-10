"""Weekly thematic analysis of email topics."""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .email.draft_store import _load as load_drafts

# Topic keywords for clustering
TOPIC_KEYWORDS = {
    "POD azonosító": ["pod", "hu-xxxx", "pod azonosító"],
    "Értesítési központ": ["értesítési központ", "bejelentkezés", "kaü", "ügyfélkapu"],
    "Tulajdonviszony": ["tulajdoni lap", "tulajdonos", "hrsz", "helyrajzi"],
    "Inverter méretezés": ["inverter", "kw", "teljesítmény", "méretezés"],
    "Hiánypótlás": ["hiánypótlás", "hiánypótl", "pótlás", "dokumentum"],
    "Gazdasági tevékenység": ["vállalkozó", "székhely", "telephely", "gazdasági"],
    "Építési telek": ["építési telek", "üres telek", "építés alatt"],
    "Pályázat állapot": ["állapot", "státusz", "mikor", "folyamat", "döntés"],
    "Kifizetés": ["kifizetés", "folyósítás", "utalás", "mikor kapom"],
    "Műszaki kérdés": ["napelem", "akkumulátor", "tároló", "műszaki"],
}

from .config import settings

REPORT_DIR = Path(settings.report_dir)


def _classify_topic(text: str) -> list[str]:
    """Classify text into topics based on keywords."""
    text_lower = text.lower()
    topics = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            topics.append(topic)
    return topics or ["Egyéb"]


def _normalize_subject(subject: str) -> str:
    """Remove RE:/FW: prefixes and normalize."""
    subject = re.sub(r"^(RE|FW|Fwd|Vá|VS|AW|SV)\s*:\s*", "", subject, flags=re.IGNORECASE).strip()
    return subject.lower()


def analyze_weekly(weeks: int = 1) -> dict:
    """Analyze emails from the past N weeks.

    Uses the draft store as a proxy for processed emails.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(weeks=weeks)
    drafts = load_drafts()

    # Filter by date
    recent = []
    for d in drafts:
        created = d.get("created_at", "")
        if created:
            try:
                dt = datetime.fromisoformat(created)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt >= cutoff:
                    recent.append(d)
            except (ValueError, TypeError):
                pass

    if not recent:
        return {
            "period_weeks": weeks,
            "total_emails": 0,
            "topics": [],
            "repeated_subjects": [],
            "low_confidence_topics": [],
        }

    # Topic analysis
    topic_counter = Counter()
    topic_confidence: dict[str, list[str]] = {}
    subject_counter = Counter()

    for d in recent:
        subject = d.get("subject", "")
        confidence = d.get("confidence", "medium")

        # Topic classification from subject + body snippet
        text = subject + " " + d.get("body_preview", "")
        topics = _classify_topic(text)
        for t in topics:
            topic_counter[t] += 1
            topic_confidence.setdefault(t, []).append(confidence)

        # Subject normalization for repetition detection
        norm = _normalize_subject(subject)
        if norm and len(norm) > 5:
            subject_counter[norm] += 1

    # Top 5 topics
    top_topics = [
        {"topic": t, "count": c}
        for t, c in topic_counter.most_common(5)
    ]

    # Repeated subjects (>= 2 occurrences)
    repeated = [
        {"subject": s, "count": c}
        for s, c in subject_counter.most_common(10)
        if c >= 2
    ]

    # Low confidence topics
    low_conf_topics = []
    for topic, confs in topic_confidence.items():
        low_count = sum(1 for c in confs if c == "low")
        if low_count > 0:
            low_conf_topics.append({
                "topic": topic,
                "total": len(confs),
                "low_confidence": low_count,
                "ratio": round(low_count / len(confs), 2),
            })
    low_conf_topics.sort(key=lambda x: x["ratio"], reverse=True)

    return {
        "period_weeks": weeks,
        "total_emails": len(recent),
        "topics": top_topics,
        "repeated_subjects": repeated,
        "low_confidence_topics": low_conf_topics,
    }


def generate_weekly_report(weeks: int = 1) -> dict:
    """Generate and save weekly report to Obsidian."""
    analysis = analyze_weekly(weeks)
    now = datetime.now()

    # Format markdown report
    lines = [
        f"# CC Heti Riport — {now.strftime('%Y-%m-%d')}",
        "",
        f"**Időszak:** elmúlt {weeks} hét",
        f"**Feldolgozott emailek:** {analysis['total_emails']}",
        "",
        "## Top témakörök",
    ]
    for t in analysis["topics"]:
        lines.append(f"- **{t['topic']}**: {t['count']} email")

    lines.append("")
    lines.append("## Ismétlődő kérdések")
    if analysis["repeated_subjects"]:
        for s in analysis["repeated_subjects"][:5]:
            lines.append(f"- \"{s['subject']}\" ({s['count']}x)")
    else:
        lines.append("- Nincs ismétlődő tárgysor")

    lines.append("")
    lines.append("## Alacsony confidence témák")
    if analysis["low_confidence_topics"]:
        for t in analysis["low_confidence_topics"]:
            lines.append(f"- **{t['topic']}**: {t['low_confidence']}/{t['total']} alacsony ({t['ratio']*100:.0f}%)")
    else:
        lines.append("- Minden téma megfelelő confidence szinttel")

    md_content = "\n".join(lines)

    # Save report
    saved_path = None
    try:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"heti-riport-{now.strftime('%Y-%m-%d')}.md"
        report_path = REPORT_DIR / filename
        report_path.write_text(md_content, encoding="utf-8")
        saved_path = str(report_path)
    except Exception as e:
        print(f"[analytics] Failed to save report: {e}")

    return {
        "analysis": analysis,
        "report_markdown": md_content,
        "saved_to": saved_path,
    }
