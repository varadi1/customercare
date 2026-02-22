"""Historical email ingestion — fetch sent items and ingest as Q&A knowledge.

Strategy: Sent replies contain both the NEÜ answer and the quoted original
question in the body. We ingest the full body as-is — the RAG pipeline benefits
from seeing the complete Q&A context together.

For shared mailboxes with thousands of daily emails, fetching+matching inbox
messages is impractical. The sent body approach is simpler and more reliable.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta, timezone

import httpx
from bs4 import BeautifulSoup

from ..config import settings
from ..rag.ingest import ingest_text_async
from .auth import get_auth_headers

GRAPH_BASE = "https://graph.microsoft.com/v1.0"

_RE_PREFIX = re.compile(r"^(RE|FW|Fwd|Vá|VS|AW|SV)\s*:\s*", re.IGNORECASE)


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n", strip=True)


def _strip_subject_prefix(subject: str) -> str:
    """Strip RE:/FW:/etc. prefixes."""
    s = subject.strip()
    while True:
        new_s = _RE_PREFIX.sub("", s).strip()
        if new_s == s:
            break
        s = new_s
    return s


def _normalize_since(since: str | None) -> str:
    if not since:
        return (datetime.now(timezone.utc) - timedelta(days=90)).strftime("%Y-%m-%dT00:00:00Z")
    if "T" not in since:
        return f"{since}T00:00:00Z"
    return since


def _is_internal(email: str) -> bool:
    """Check if an email address is internal (NEÜ)."""
    email = (email or "").lower()
    return any(email.endswith(d) for d in [
        "@neuzrt.hu", "@nffku.hu", "@nffku.onmicrosoft.com", "@norvegalap.hu"
    ])


def _is_auto_reply(subject: str, body: str) -> bool:
    """Detect auto-replies and delivery notifications."""
    subj_lower = subject.lower()
    auto_keywords = [
        "automatic reply", "automatikus válasz", "out of office",
        "delivery status", "undeliverable", "mailer-daemon",
        "nem kézbesíthető", "kézbesítési állapot",
    ]
    return any(kw in subj_lower for kw in auto_keywords) or len(body) < 30


async def fetch_sent_items(
    mailbox: str,
    since: str,
    max_items: int = 500,
) -> list[dict]:
    """Fetch sent items with pagination."""
    headers = get_auth_headers()
    url = f"{GRAPH_BASE}/users/{mailbox}/mailFolders/sentItems/messages"
    params = {
        "$filter": f"sentDateTime ge {since}",
        "$orderby": "sentDateTime desc",
        "$top": min(max_items, 50),
        "$select": "id,subject,from,toRecipients,body,sentDateTime",
    }

    all_messages = []
    async with httpx.AsyncClient(timeout=60) as client:
        while url and len(all_messages) < max_items:
            resp = await client.get(url, headers=headers, params=params)
            if resp.status_code != 200:
                print(f"[history] Error fetching sentItems from {mailbox}: {resp.status_code} {resp.text[:300]}")
                break

            data = resp.json()
            all_messages.extend(data.get("value", []))
            url = data.get("@odata.nextLink")
            params = {}

            if not url or len(all_messages) >= max_items:
                break

    return all_messages[:max_items]


async def ingest_historical_emails(
    mailbox: str,
    since: str | None = None,
    max_items: int = 200,
    dry_run: bool = False,
) -> dict:
    """Fetch sent items and ingest as Q&A knowledge.

    Each sent reply contains the NEÜ answer + quoted original question.
    We ingest the full body with proper metadata.
    """
    since_norm = _normalize_since(since)
    stats = {
        "mailbox": mailbox,
        "sent_items_fetched": 0,
        "ingested": 0,
        "chunks_created": 0,
        "skipped_internal": 0,
        "skipped_auto_reply": 0,
        "skipped_short": 0,
        "skipped_duplicate_subject": 0,
        "errors": 0,
        "samples": [],
    }

    print(f"[history] Fetching sent items from {mailbox} (since={since_norm})...")
    sent_items = await fetch_sent_items(mailbox, since_norm, max_items)
    stats["sent_items_fetched"] = len(sent_items)
    print(f"[history] Got {len(sent_items)} sent items")

    if not sent_items:
        return stats

    # Track content hashes to avoid true duplicates (same answer to different people)
    import hashlib as _hashlib
    seen_content_hashes = set()

    for msg in sent_items:
        try:
            subject = msg.get("subject", "")
            to_addrs = [r.get("emailAddress", {}).get("address", "")
                        for r in msg.get("toRecipients", [])]

            # Skip internal-only (forwards to colleagues)
            if to_addrs and all(_is_internal(a) for a in to_addrs):
                stats["skipped_internal"] += 1
                continue

            # Extract body text
            body_html = msg.get("body", {}).get("content", "")
            body_text = _html_to_text(body_html).strip()

            # Skip auto-replies
            if _is_auto_reply(subject, body_text):
                stats["skipped_auto_reply"] += 1
                continue

            # Skip very short
            if len(body_text) < 50:
                stats["skipped_short"] += 1
                continue

            # Skip signature-only emails (no real content beyond NEÜ signature)
            sig_patterns = ["Nemzeti Energetikai Ügynökség", "Montevideo u.", "1037- Budapest"]
            content_lines = [l.strip() for l in body_text.split("\n") 
                           if l.strip() and not any(p in l for p in sig_patterns) and len(l.strip()) > 10]
            if len(" ".join(content_lines)) < 50:
                stats["skipped_short"] += 1
                continue

            # Skip true duplicates: same answer content (first 500 chars of body)
            # This allows different answers with the same subject to be ingested
            content_hash = _hashlib.sha256(body_text[:500].encode()).hexdigest()[:16]
            if content_hash in seen_content_hashes:
                stats["skipped_duplicate_subject"] += 1  # reuse counter name for compat
                continue
            seen_content_hashes.add(content_hash)

            sent_date = msg.get("sentDateTime", "")[:10]
            msg_id_short = msg.get("id", "unknown")[:20]
            source = f"email_reply:{mailbox}:{msg_id_short}"

            # Format for ingestion: subject + full body (answer + quoted question)
            full_text = f"Tárgy: {subject}\nMailbox: {mailbox}\nDátum: {sent_date}\n\n{body_text}"

            if dry_run:
                stats["ingested"] += 1
                if len(stats["samples"]) < 10:
                    stats["samples"].append({
                        "subject": subject,
                        "to": to_addrs,
                        "date": sent_date,
                        "body_len": len(body_text),
                        "body_preview": body_text[:400],
                    })
            else:
                chunks = await ingest_text_async(
                    text=full_text,
                    source=source,
                    category="ügyfélszolgálat",
                    chunk_type="email_reply",
                    valid_from=sent_date or None,
                )
                stats["ingested"] += 1
                stats["chunks_created"] += chunks
                if len(stats["samples"]) < 10:
                    stats["samples"].append({
                        "subject": subject,
                        "date": sent_date,
                        "chunks": chunks,
                    })

        except Exception as e:
            print(f"[history] Error processing msg: {e}")
            stats["errors"] += 1

    return stats
