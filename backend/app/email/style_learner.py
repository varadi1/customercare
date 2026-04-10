"""Style learner: analyze colleague sent emails to extract response patterns.

Builds a local pattern database from real sent emails that CC can use
to match her draft style to the actual colleague style.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup

from .auth import get_auth_headers

GRAPH_BASE = "https://graph.microsoft.com/v1.0"
PATTERNS_PATH = Path("/app/data/style_patterns.json")


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n", strip=True)


def _strip_quoted(text: str) -> str:
    """Remove quoted/forwarded parts from email text."""
    lines = text.split("\n")
    clean = []
    for line in lines:
        # Common quote markers
        if line.strip().startswith(">"):
            break
        if re.match(r"^-{3,}\s*(Eredeti|Original|Forwarded)", line, re.IGNORECASE):
            break
        if re.match(r"^(From|Feladó|Tárgy|Subject|Sent|Dátum|Date):", line):
            break
        if "írta:" in line and "@" in line:
            break
        clean.append(line)
    return "\n".join(clean).strip()


def _analyze_single_email(text: str) -> dict:
    """Extract style features from a single email text."""
    text = _strip_quoted(text)
    if not text or len(text) < 20:
        return {}

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    word_count = len(text.split())
    sentence_count = len(re.findall(r"[.!?]+", text)) or 1

    # Greeting detection
    greeting = ""
    if lines:
        first = lines[0]
        if any(g in first.lower() for g in ["tisztelt", "kedves", "üdvözl", "hello", "szia"]):
            greeting = first

    # Closing detection
    closing = ""
    for line in reversed(lines[-5:]):
        if any(c in line.lower() for c in ["üdvözlet", "tisztelet", "köszön", "u.i."]):
            closing = line
            break

    # Tone markers
    uses_conditional = bool(re.search(r"\bamennyiben\b|\babban az esetben\b|\bfeltéve\b", text, re.IGNORECASE))
    uses_polite_request = bool(re.search(r"\bkérjük\b|\bszíves\b|\bszíveskedj\b", text, re.IGNORECASE))
    uses_direct = bool(re.search(r"\bkell\b|\bszükséges\b|\bfeltölteni\b|\bcsatolni\b", text, re.IGNORECASE))
    has_list = bool(re.search(r"^\s*[-•*–]\s", text, re.MULTILINE))
    has_numbered_list = bool(re.search(r"^\s*\d+[.)]\s", text, re.MULTILINE))

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": round(word_count / sentence_count, 1),
        "line_count": len(lines),
        "greeting": greeting,
        "closing": closing,
        "uses_conditional": uses_conditional,
        "uses_polite_request": uses_polite_request,
        "uses_direct": uses_direct,
        "has_list": has_list,
        "has_numbered_list": has_numbered_list,
        "text_preview": text[:300],
    }


def _categorize_email(subject: str, body_text: str) -> str:
    """Categorize an email by topic for pattern grouping."""
    combined = f"{subject} {body_text}".lower()

    categories = {
        "ertesites_kau": ["értesítés", "értesítési központ", "kaü", "ügyfélkapu"],
        "hianypotlas": ["hiánypótl", "hiányosság", "pótlás", "dokumentum"],
        "inverter": ["inverter", "teljesítmény", "kw", "méretez"],
        "szaldo": ["szaldó", "elszámolás", "betáplálás"],
        "napelem": ["napelem", "napelemrendszer", "solar"],
        "tamogatas": ["támogatás", "összeg", "kifizetés", "folyósítás"],
        "hatarido": ["határidő", "mikor", "meddig", "időpont"],
        "jogosultsag": ["jogosult", "feltétel", "igényel"],
        "kivitelezo": ["kivitelező", "regisztráció", "vállalkozó"],
        "dokumentum": ["lakcímkártya", "tulajdoni lap", "igazolás", "csatolmány"],
        "altalanos": [],  # fallback
    }

    for cat, keywords in categories.items():
        if any(kw in combined for kw in keywords):
            return cat
    return "altalanos"


async def analyze_sent_items(
    mailbox: str = "lakossagitarolo@neuzrt.hu",
    hours: int = 168,  # 1 week default
    limit: int = 200,
) -> dict[str, Any]:
    """Analyze recent sent items to extract colleague response patterns.

    Returns aggregated style statistics and per-category patterns.
    """
    headers = get_auth_headers()
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    since_str = since.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Fetch sent emails
    sent_emails: list[dict] = []
    async with httpx.AsyncClient(timeout=60) as client:
        url = f"{GRAPH_BASE}/users/{mailbox}/mailFolders/SentItems/messages"
        params = {
            "$top": "100",
            "$filter": f"sentDateTime ge {since_str}",
            "$select": "id,subject,body,sentDateTime,from,toRecipients",
            "$orderby": "sentDateTime desc",
        }

        resp = await client.get(url, headers=headers, params=params)
        if resp.status_code != 200:
            return {"error": f"Failed to fetch sent items: {resp.status_code}"}

        data = resp.json()
        sent_emails.extend(data.get("value", []))

        # Paginate
        next_link = data.get("@odata.nextLink")
        pages = 0
        while next_link and pages < 4 and len(sent_emails) < limit:
            resp = await client.get(next_link, headers=headers)
            if resp.status_code != 200:
                break
            data = resp.json()
            sent_emails.extend(data.get("value", []))
            next_link = data.get("@odata.nextLink")
            pages += 1

    if not sent_emails:
        return {"status": "no_sent_emails", "count": 0}

    # Analyze each email
    analyses: list[dict] = []
    category_examples: dict[str, list[dict]] = defaultdict(list)

    for email in sent_emails:
        body_html = email.get("body", {}).get("content", "")
        body_text = _html_to_text(body_html)
        subject = email.get("subject", "")

        # Skip very short auto-replies
        stripped = _strip_quoted(body_text)
        if len(stripped) < 30:
            continue

        # Skip CC/Hanna AI drafts — they pollute style patterns with CC's own style
        if ("CC AI Draft" in body_html or "Hanna AI Draft" in body_html
                or "CC - draft" in body_html or "Hanna - draft" in body_html
                or "background:#f0f0f0" in body_html):
            continue

        features = _analyze_single_email(body_text)
        if not features:
            continue

        category = _categorize_email(subject, body_text)
        features["category"] = category
        features["subject"] = subject[:80]
        features["sent_at"] = email.get("sentDateTime", "")
        analyses.append(features)

        # Keep top examples per category (up to 5)
        if len(category_examples[category]) < 5:
            category_examples[category].append({
                "subject": subject[:80],
                "text": stripped[:500],
                "word_count": features["word_count"],
                "greeting": features["greeting"],
                "closing": features["closing"],
            })

    if not analyses:
        return {"status": "no_analyzable_emails", "count": 0}

    # Aggregate statistics
    word_counts = [a["word_count"] for a in analyses]
    sentence_counts = [a["sentence_count"] for a in analyses]

    greetings = Counter(a["greeting"] for a in analyses if a["greeting"])
    closings = Counter(a["closing"] for a in analyses if a["closing"])
    categories = Counter(a["category"] for a in analyses)

    style_stats = {
        "total_analyzed": len(analyses),
        "period_hours": hours,
        "mailbox": mailbox,
        "word_count": {
            "avg": round(sum(word_counts) / len(word_counts), 1),
            "median": sorted(word_counts)[len(word_counts) // 2],
            "min": min(word_counts),
            "max": max(word_counts),
            "p25": sorted(word_counts)[len(word_counts) // 4],
            "p75": sorted(word_counts)[3 * len(word_counts) // 4],
        },
        "sentence_count": {
            "avg": round(sum(sentence_counts) / len(sentence_counts), 1),
            "median": sorted(sentence_counts)[len(sentence_counts) // 2],
        },
        "tone": {
            "uses_conditional_pct": round(100 * sum(1 for a in analyses if a["uses_conditional"]) / len(analyses)),
            "uses_polite_request_pct": round(100 * sum(1 for a in analyses if a["uses_polite_request"]) / len(analyses)),
            "uses_direct_pct": round(100 * sum(1 for a in analyses if a["uses_direct"]) / len(analyses)),
            "has_list_pct": round(100 * sum(1 for a in analyses if a["has_list"]) / len(analyses)),
        },
        "top_greetings": greetings.most_common(5),
        "top_closings": closings.most_common(5),
        "categories": categories.most_common(15),
        "category_examples": {k: v for k, v in category_examples.items()},
    }

    # Save patterns to disk for reuse
    _save_patterns(style_stats)

    return style_stats


def _save_patterns(stats: dict):
    """Persist patterns to disk."""
    PATTERNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    stats_copy = {**stats}
    stats_copy["last_updated"] = datetime.now(timezone.utc).isoformat()
    PATTERNS_PATH.write_text(json.dumps(stats_copy, ensure_ascii=False, indent=2))
    print(f"[style_learner] Patterns saved to {PATTERNS_PATH}")


def load_patterns() -> dict | None:
    """Load saved patterns."""
    if PATTERNS_PATH.exists():
        return json.loads(PATTERNS_PATH.read_text())
    return None


async def get_category_templates(
    mailbox: str = "lakossagitarolo@neuzrt.hu",
    hours: int = 720,  # 30 days
    min_examples: int = 3,
) -> dict[str, Any]:
    """Build response templates per category from sent items.

    Groups sent emails by topic category and extracts common patterns
    that can be used as draft templates.
    """
    headers = get_auth_headers()
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    since_str = since.strftime("%Y-%m-%dT%H:%M:%SZ")

    sent_emails: list[dict] = []
    async with httpx.AsyncClient(timeout=60) as client:
        url = f"{GRAPH_BASE}/users/{mailbox}/mailFolders/SentItems/messages"
        params = {
            "$top": "100",
            "$filter": f"sentDateTime ge {since_str}",
            "$select": "id,subject,body,sentDateTime",
            "$orderby": "sentDateTime desc",
        }

        resp = await client.get(url, headers=headers, params=params)
        if resp.status_code != 200:
            return {"error": f"API error: {resp.status_code}"}

        data = resp.json()
        sent_emails.extend(data.get("value", []))

        next_link = data.get("@odata.nextLink")
        pages = 0
        while next_link and pages < 9 and len(sent_emails) < 500:
            resp = await client.get(next_link, headers=headers)
            if resp.status_code != 200:
                break
            data = resp.json()
            sent_emails.extend(data.get("value", []))
            next_link = data.get("@odata.nextLink")
            pages += 1

    # Group by category
    by_category: dict[str, list[str]] = defaultdict(list)
    for email in sent_emails:
        body_html = email.get("body", {}).get("content", "")
        body_text = _html_to_text(body_html)
        subject = email.get("subject", "")
        stripped = _strip_quoted(body_text)
        if len(stripped) < 30:
            continue
        cat = _categorize_email(subject, body_text)
        by_category[cat].append(stripped[:800])

    templates = {}
    for cat, examples in by_category.items():
        if len(examples) < min_examples:
            continue
        # Take the 5 most recent as representative examples
        templates[cat] = {
            "count": len(examples),
            "examples": examples[:5],
            "avg_length": round(sum(len(e) for e in examples) / len(examples)),
        }

    return {
        "total_emails": len(sent_emails),
        "categories_with_templates": len(templates),
        "templates": templates,
    }
