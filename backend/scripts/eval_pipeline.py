#!/usr/bin/env python3
"""Offline evaluation pipeline — test CustomerCare against real sent emails.

Fetches sent emails, extracts the original question (from quoted thread),
sends it to CC /draft/generate, and compares the draft to the actual answer.

Usage (inside cc-backend container):
    python3 /app/scripts/eval_pipeline.py --limit 250 [--dry-run] [--output /app/data/eval_results.json]

Or from host:
    docker exec cc-backend python3 /app/scripts/eval_pipeline.py --limit 250
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

import httpx
from bs4 import BeautifulSoup

sys.path.insert(0, "/app")

GRAPH_BASE = "https://graph.microsoft.com/v1.0"
CC_URL = "http://localhost:8000"

# ── Email parsing ──

# Patterns that mark the start of a quoted/forwarded section
QUOTE_PATTERNS = [
    re.compile(r"^-{3,}\s*(Eredeti üzenet|Original Message|Továbbított üzenet|Forwarded message)", re.MULTILINE | re.IGNORECASE),
    # Outlook Desktop: "From:" or "Feladó:" on its own line (content may be next line)
    re.compile(r"^From:$", re.MULTILINE),
    re.compile(r"^From:\s+\S", re.MULTILINE),
    re.compile(r"^Feladó:$", re.MULTILINE),
    re.compile(r"^Feladó:\s+\S", re.MULTILINE),
    # Line of underscores or equals
    re.compile(r"^[_=]{10,}$", re.MULTILINE),
]

# NEÜ signature patterns to strip from answer
NEU_SIG_PATTERNS = [
    "Nemzeti Energetikai Ügynökség",
    "Zártkörűen Működő Részvénytársaság",
    "1037- Budapest",
    "1037 Budapest",
    "Montevideo u.",
    "NEÜ Zrt.",
]

AUTO_REPLY_KEYWORDS = [
    "automatic reply", "automatikus válasz", "out of office",
    "delivery status", "undeliverable", "mailer-daemon",
    "nem kézbesíthető", "szabadságom",
]


def html_to_text(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n", strip=True)


def is_auto_reply(subject: str, body: str) -> bool:
    subj_lower = (subject or "").lower()
    body_lower = (body or "").lower()[:200]
    return any(kw in subj_lower or kw in body_lower for kw in AUTO_REPLY_KEYWORDS)


def is_internal(email: str) -> bool:
    email = (email or "").lower()
    return any(email.endswith(d) for d in ["@neuzrt.hu", "@nffku.hu", "@norvegalap.hu"])


def split_answer_and_question(body: str) -> tuple[str, str]:
    """Split email body into (NEÜ answer, original customer question).

    The sent email typically looks like:
        [NEÜ answer]
        ---
        [signature]
        ---
        From: customer@example.com
        [original question]

    Returns (answer, question). Either can be empty.
    """
    lines = body.split("\n")

    # Find the first quote boundary
    split_idx = None
    for i, line in enumerate(lines):
        for pattern in QUOTE_PATTERNS:
            if pattern.search(line):
                split_idx = i
                break
        if split_idx is not None:
            break

    if split_idx is None:
        # No quoted section found — the whole body is the answer
        return _strip_signature(body), ""

    answer_part = "\n".join(lines[:split_idx])
    question_part = "\n".join(lines[split_idx:])

    # Clean up the question: remove From:/Sent:/To: headers and metadata lines
    question_lines = []
    skip_headers = True
    for line in question_part.split("\n"):
        if skip_headers:
            stripped = line.strip()
            # Skip separator lines
            if re.match(r"^[-_=]{3,}", stripped):
                continue
            # Skip header fields (may span: "From:" on one line, value on next)
            if re.match(r"^(From|Feladó|Sent|Elküldve|To|Címzett|Subject|Tárgy|Date|Dátum|Cc|Másolat)\s*:", stripped, re.IGNORECASE):
                continue
            # Skip email addresses on their own line (from the header block)
            if re.match(r"^<?[\w.+-]+@[\w.-]+>?$", stripped):
                continue
            # Skip date-like lines (e.g. "Tuesday, April 7, 2026 9:46 AM")
            if re.match(r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Hétfő|Kedd|Szerda|Csütörtök|Péntek|Szombat|Vasárnap)", stripped, re.IGNORECASE):
                continue
            if re.match(r"^\d{4}\.\s*(január|február|március|április|május|június|július|augusztus|szeptember|október|november|december)", stripped, re.IGNORECASE):
                continue
            # Skip empty lines in the header block
            if not stripped:
                continue
            # Skip "Forwarded message" marker
            if "forwarded message" in stripped.lower():
                continue
            # Found actual content — stop skipping
            skip_headers = False
        question_lines.append(line)

    return _strip_signature(answer_part), "\n".join(question_lines).strip()


def _strip_signature(text: str) -> str:
    """Remove NEÜ signature from answer text."""
    lines = text.split("\n")
    # Find where signature starts
    sig_start = None
    for i, line in enumerate(lines):
        if any(p in line for p in NEU_SIG_PATTERNS):
            sig_start = i
            break

    if sig_start is not None and sig_start > 2:
        # Also remove "Üdvözlettel" line before signature
        if sig_start > 0 and "dvözlettel" in lines[sig_start - 1]:
            sig_start -= 1
        text = "\n".join(lines[:sig_start])

    return text.strip()


# ── Graph API ──

async def fetch_sent_emails(mailbox: str, max_items: int = 250) -> list[dict]:
    """Fetch sent emails with conversationId for thread matching."""
    from app.email.auth import get_auth_headers
    headers = get_auth_headers()

    url = f"{GRAPH_BASE}/users/{mailbox}/mailFolders/sentItems/messages"
    params = {
        "$orderby": "sentDateTime desc",
        "$top": min(max_items, 50),
        "$select": "id,subject,from,toRecipients,body,sentDateTime,conversationId,hasAttachments",
    }

    all_messages = []
    async with httpx.AsyncClient(timeout=60) as client:
        while url and len(all_messages) < max_items:
            resp = await client.get(url, headers=headers, params=params)
            if resp.status_code != 200:
                print(f"[eval] Error fetching: {resp.status_code}")
                break
            data = resp.json()
            all_messages.extend(data.get("value", []))
            url = data.get("@odata.nextLink")
            params = {}
            time.sleep(0.1)

    return all_messages[:max_items]


# ── CustomerCare eval ──

async def generate_cc_draft(question: str, subject: str, sender: str = "Teszt Pályázó") -> dict:
    """Call CC /draft/generate with the extracted question."""
    async with httpx.AsyncClient(timeout=120) as client:
        payload = {
            "email_text": question[:3000],
            "email_subject": subject,
            "sender_name": sender,
            "sender_email": "eval@test.hu",
            "top_k": 5,
        }
        try:
            resp = await client.post(f"{CC_URL}/draft/generate", json=payload)
            if resp.status_code == 200:
                return resp.json()
            else:
                return {"error": f"HTTP {resp.status_code}", "detail": resp.text[:200]}
        except Exception as e:
            return {"error": str(e)}


def compute_similarity(text_a: str, text_b: str) -> float:
    """Quick similarity using difflib (no embedding needed for first pass)."""
    from difflib import SequenceMatcher
    if not text_a or not text_b:
        return 0.0
    # Normalize
    a = re.sub(r"\s+", " ", text_a.lower().strip())[:2000]
    b = re.sub(r"\s+", " ", text_b.lower().strip())[:2000]
    return SequenceMatcher(None, a, b).ratio()


def extract_app_ids(text: str) -> set[str]:
    """Extract OETP-2026-XXXXXX identifiers."""
    return set(re.findall(r"OETP-\d{4}-\d+", text))


def extract_key_terms(text: str) -> set[str]:
    """Extract domain-specific terms for content comparison."""
    terms = set()
    # OETP IDs
    terms.update(re.findall(r"OETP-\d{4}-\d+", text))
    # Law references
    terms.update(re.findall(r"\d+/\d{4}", text))
    # Key domain words
    domain_words = [
        "szaldó", "bruttó", "inverter", "akkumulátor", "tároló",
        "meghatalmazás", "meghatalmazott", "igazolási", "fenntartási",
        "kivitelező", "vállalkozási", "felhívás", "pályázat",
        "tulajdon", "társasház", "végrehajtás", "elutasít",
        "hiánypótl", "támogatás", "2,5 millió", "POD",
    ]
    text_lower = text.lower()
    for w in domain_words:
        if w.lower() in text_lower:
            terms.add(w.lower())
    return terms


@dataclass
class EvalResult:
    """Single evaluation result."""
    msg_id: str
    subject: str
    sent_date: str
    question_len: int
    answer_len: int
    draft_len: int
    similarity: float  # difflib ratio
    confidence: str  # CC confidence
    app_ids_match: bool  # Did CC mention the same OETP IDs?
    key_terms_overlap: float  # Jaccard of domain terms
    question_preview: str
    answer_preview: str
    draft_preview: str
    error: str | None = None
    sources_used: list[str] = field(default_factory=list)


async def run_eval(
    mailbox: str = "lakossagitarolo@neuzrt.hu",
    max_items: int = 250,
    dry_run: bool = False,
    output_path: str = "/app/data/eval_results.json",
    skip_existing: bool = True,
) -> dict:
    """Run the full evaluation pipeline."""

    print(f"[eval] Fetching {max_items} sent emails from {mailbox}...")
    sent_emails = await fetch_sent_emails(mailbox, max_items)
    print(f"[eval] Got {len(sent_emails)} emails")

    results: list[EvalResult] = []
    stats = {
        "total_fetched": len(sent_emails),
        "usable": 0,
        "skipped_auto": 0,
        "skipped_internal": 0,
        "skipped_no_question": 0,
        "skipped_short": 0,
        "evaluated": 0,
        "errors": 0,
        "avg_similarity": 0.0,
        "avg_key_overlap": 0.0,
        "high_similarity": 0,  # > 0.6
        "medium_similarity": 0,  # 0.3-0.6
        "low_similarity": 0,  # < 0.3
    }

    for i, msg in enumerate(sent_emails):
        subject = msg.get("subject", "")
        to_addrs = [r.get("emailAddress", {}).get("address", "")
                    for r in msg.get("toRecipients", [])]
        body_html = msg.get("body", {}).get("content", "")
        body_text = html_to_text(body_html)
        sent_date = msg.get("sentDateTime", "")[:10]
        msg_id = hashlib.sha256(msg.get("id", "").encode()).hexdigest()[:12]

        # Filters
        if is_auto_reply(subject, body_text):
            stats["skipped_auto"] += 1
            continue
        if to_addrs and all(is_internal(a) for a in to_addrs):
            stats["skipped_internal"] += 1
            continue
        if len(body_text) < 50:
            stats["skipped_short"] += 1
            continue

        # Split answer/question
        actual_answer, question = split_answer_and_question(body_text)

        if not question or len(question) < 30:
            stats["skipped_no_question"] += 1
            continue

        if len(actual_answer) < 30:
            stats["skipped_short"] += 1
            continue

        stats["usable"] += 1

        if dry_run:
            print(f"  [{i+1}] {subject[:60]} — Q:{len(question)} A:{len(actual_answer)}")
            if stats["usable"] <= 3:
                print(f"       Q: {question[:120]}...")
                print(f"       A: {actual_answer[:120]}...")
            continue

        # Generate CC draft
        print(f"  [{i+1}/{len(sent_emails)}] {subject[:50]}...", end=" ", flush=True)
        draft_result = await generate_cc_draft(question, subject)

        if "error" in draft_result:
            stats["errors"] += 1
            results.append(EvalResult(
                msg_id=msg_id, subject=subject, sent_date=sent_date,
                question_len=len(question), answer_len=len(actual_answer),
                draft_len=0, similarity=0, confidence="error",
                app_ids_match=False, key_terms_overlap=0,
                question_preview=question[:200],
                answer_preview=actual_answer[:200],
                draft_preview="",
                error=draft_result.get("error", "unknown"),
            ))
            print(f"ERROR: {draft_result.get('error', '')[:50]}")
            continue

        draft_text = (
            draft_result.get("body_html", "")
            or draft_result.get("draft_text", "")
            or draft_result.get("draft", "")
        )
        if isinstance(draft_text, dict):
            draft_text = draft_text.get("body", "")

        # Strip HTML from draft if needed
        if "<" in draft_text:
            draft_text = html_to_text(draft_text)

        confidence = draft_result.get("confidence", "unknown")

        # Compare
        sim = compute_similarity(actual_answer, draft_text)

        answer_ids = extract_app_ids(actual_answer)
        draft_ids = extract_app_ids(draft_text)
        app_match = bool(answer_ids & draft_ids) if answer_ids else True

        answer_terms = extract_key_terms(actual_answer)
        draft_terms = extract_key_terms(draft_text)
        union = answer_terms | draft_terms
        key_overlap = len(answer_terms & draft_terms) / len(union) if union else 1.0

        # Sources
        sources = []
        for chunk in draft_result.get("top_chunks", draft_result.get("sources", [])):
            if isinstance(chunk, dict):
                sources.append(chunk.get("chunk_type", chunk.get("source", "?")))

        result = EvalResult(
            msg_id=msg_id, subject=subject, sent_date=sent_date,
            question_len=len(question), answer_len=len(actual_answer),
            draft_len=len(draft_text), similarity=round(sim, 3),
            confidence=str(confidence),
            app_ids_match=app_match,
            key_terms_overlap=round(key_overlap, 3),
            question_preview=question[:300],
            answer_preview=actual_answer[:300],
            draft_preview=draft_text[:300],
            sources_used=sources[:5],
        )
        results.append(result)
        stats["evaluated"] += 1

        if sim >= 0.6:
            stats["high_similarity"] += 1
        elif sim >= 0.3:
            stats["medium_similarity"] += 1
        else:
            stats["low_similarity"] += 1

        print(f"sim={sim:.2f} conf={confidence} terms={key_overlap:.2f}")

        # Rate limit (be nice to the API)
        await asyncio.sleep(1)

    # Aggregate stats
    if results:
        sims = [r.similarity for r in results if r.error is None]
        overlaps = [r.key_terms_overlap for r in results if r.error is None]
        stats["avg_similarity"] = round(sum(sims) / len(sims), 3) if sims else 0
        stats["avg_key_overlap"] = round(sum(overlaps) / len(overlaps), 3) if overlaps else 0

    # Save results
    if not dry_run and results:
        output = {
            "run_date": datetime.now(timezone.utc).isoformat(),
            "mailbox": mailbox,
            "stats": stats,
            "results": [asdict(r) for r in results],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n[eval] Results saved to {output_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"  EVAL ÖSSZESÍTÉS {'(DRY RUN)' if dry_run else ''}")
    print(f"{'='*60}")
    print(f"  Letöltve:           {stats['total_fetched']}")
    print(f"  Használható:        {stats['usable']}")
    print(f"  Skip (auto-reply):  {stats['skipped_auto']}")
    print(f"  Skip (internal):    {stats['skipped_internal']}")
    print(f"  Skip (no question): {stats['skipped_no_question']}")
    print(f"  Skip (short):       {stats['skipped_short']}")
    if not dry_run:
        print(f"  Értékelve:          {stats['evaluated']}")
        print(f"  Hibák:              {stats['errors']}")
        print(f"  Átlag hasonlóság:   {stats['avg_similarity']}")
        print(f"  Átlag term overlap: {stats['avg_key_overlap']}")
        print(f"  Magas (>0.6):       {stats['high_similarity']}")
        print(f"  Közepes (0.3-0.6):  {stats['medium_similarity']}")
        print(f"  Alacsony (<0.3):    {stats['low_similarity']}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="CustomerCare offline eval pipeline")
    parser.add_argument("--limit", type=int, default=250, help="Max emails to fetch")
    parser.add_argument("--mailbox", default="lakossagitarolo@neuzrt.hu")
    parser.add_argument("--dry-run", action="store_true", help="Only fetch+parse, don't call CC API")
    parser.add_argument("--output", default="/app/data/eval_results.json")
    args = parser.parse_args()

    asyncio.run(run_eval(
        mailbox=args.mailbox,
        max_items=args.limit,
        dry_run=args.dry_run,
        output_path=args.output,
    ))


if __name__ == "__main__":
    main()
