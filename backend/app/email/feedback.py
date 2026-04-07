"""Feedback loop: compare sent emails with stored drafts.

v5 — 2026-03-01:
  - Feedback diff storage for heavily modified drafts
  - Max 200 entries, 30 day retention

v4 — 2026-02-27 fixes:
  - Body match threshold raised to 0.5 (was 0.15)
  - Pending status for <24h drafts without conv match
  - Reliable vs uncertain match separation in stats
  - HTML-level Hanna banner strip (before html→text)
  - Hungarian Outlook Web quote pattern added
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from .auth import get_auth_headers
from .draft_store import get_recent_drafts

import logging
_logger = logging.getLogger(__name__)

GRAPH_BASE = "https://graph.microsoft.com/v1.0"

# ─── Text Processing ─────────────────────────────────────────────────────────


def _strip_hanna_banner_html(html: str) -> str:
    """Remove Hanna AI Draft banner at HTML level BEFORE text extraction.
    
    The banner is a <div style="background:#f0f0f0; border-left:3px solid #999">
    block at the top of the draft.
    """
    if not html:
        return html
    soup = BeautifulSoup(html, "html.parser")

    # Strategy 1: find div with background:#f0f0f0 style (Hanna banner)
    for div in soup.find_all("div"):
        style = (div.get("style") or "").replace(" ", "").lower()
        if "background:#f0f0f0" in style or "background-color:#f0f0f0" in style:
            div.decompose()
            break

    # Strategy 2: find any element containing "Hanna AI Draft" or "🤖 Hanna"
    for el in soup.find_all(string=re.compile(r"(Hanna AI Draft|🤖\s*Hanna)", re.IGNORECASE)):
        parent = el.find_parent("div") or el.find_parent("p") or el.find_parent("td")
        if parent:
            parent.decompose()

    return str(soup)


def _html_to_text(html: str) -> str:
    """Strip HTML to plain text for comparison."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n", strip=True)


def _extract_reply_text(html: str) -> str:
    """Extract only the reply portion from a sent email HTML, removing quoted original.
    
    Handles: divRplyFwdMsg, border-top #E1E1E1, <hr> dividers, Original Message,
    Hungarian OWA "Feladó:/Küldés ideje:" pattern.
    """
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")

    def _text_before(element) -> str:
        texts = []
        for el in element.previous_siblings:
            t = el.get_text(separator="\n", strip=True) if hasattr(el, "get_text") else str(el).strip()
            if t:
                texts.append(t)
        return "\n".join(reversed(texts)).strip()

    # Strategy 1: <div id="divRplyFwdMsg"> (Desktop Outlook)
    reply_div = soup.find(id="divRplyFwdMsg")
    if reply_div:
        return _text_before(reply_div)

    # Strategy 2: border-top solid #E1E1E1 (OWA/new Outlook)
    for div in soup.find_all("div"):
        style = div.get("style", "") or ""
        if "border-top" in style and "#E1E1E1" in style:
            return _text_before(div)

    # Strategy 3: <hr> followed by quote block
    for hr in soup.find_all("hr"):
        result = _text_before(hr)
        if result and len(result) > 20:
            return result

    # Strategy 4: text-level patterns
    full_text = soup.get_text(separator="\n", strip=True)

    patterns = [
        r"\n\s*From:.*?\nSent:",                          # English Outlook
        r"\n\s*Feladó:.*?\nKüldés ideje:",                 # Hungarian OWA
        r"\n\s*Feladó:.*?\nElküldve:",                     # Hungarian Desktop Outlook
        r"\n\s*Feladó:.*?\nDátum:",                        # Hungarian alt
        r"-{3,}\s*Original Message\s*-{3,}",               # Original Message marker
        r"-{3,}\s*Eredeti üzenet\s*-{3,}",                 # Hungarian Original Message
    ]

    for pattern in patterns:
        match = re.search(pattern, full_text, re.IGNORECASE | re.DOTALL)
        if match:
            reply_text = full_text[: match.start()]
            reply_text = re.sub(r"\n\s*[-_=]{3,}\s*$", "", reply_text)
            return reply_text.strip()

    return full_text


def _similarity(a: str, b: str) -> float:
    """Return similarity ratio between two strings (0-1)."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _norm_subject(s: str) -> str:
    """Normalize email subject for comparison."""
    return re.sub(r"^(re:|fw:|fwd:)\s*", "", (s or "").lower().strip())


# ─── Feedback Diff Store ─────────────────────────────────────────────────────

FEEDBACK_DIFF_PATH = Path("/app/data/feedback_diffs.json")
FEEDBACK_DIFF_MAX = 200
FEEDBACK_DIFF_RETENTION_DAYS = 30


def _load_feedback_diffs() -> list[dict]:
    """Load existing feedback diffs."""
    if not FEEDBACK_DIFF_PATH.exists():
        return []
    try:
        with open(FEEDBACK_DIFF_PATH) as f:
            data = json.load(f)
        return data.get("diffs", [])
    except (json.JSONDecodeError, OSError):
        return []


def _store_feedback_diff(entry: dict) -> None:
    """Append a feedback diff entry. FIFO, max 200, 30-day retention."""
    diffs = _load_feedback_diffs()

    # Retention: remove entries older than 30 days
    cutoff = (datetime.now(timezone.utc) - timedelta(days=FEEDBACK_DIFF_RETENTION_DAYS)).isoformat()
    diffs = [d for d in diffs if d.get("created_at", "") >= cutoff]

    # Dedup: skip if same subject already exists
    subj = entry.get("subject", "")
    if any(d.get("subject") == subj for d in diffs):
        return

    diffs.append(entry)

    # FIFO: keep max 200
    if len(diffs) > FEEDBACK_DIFF_MAX:
        diffs = diffs[-FEEDBACK_DIFF_MAX:]

    FEEDBACK_DIFF_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_DIFF_PATH, "w") as f:
        json.dump({"diffs": diffs}, f, ensure_ascii=False, indent=2)


# ─── Main Feedback Check ─────────────────────────────────────────────────────

# Thresholds
BODY_MATCH_MIN_SIM = 0.50       # body-match must exceed this to count
PENDING_HOURS = 24              # drafts younger than this without conv-match → pending
UNCHANGED_THRESHOLD = 0.85
MODIFIED_THRESHOLD = 0.30


async def check_feedback(mailbox: str, hours: int = 48) -> dict:
    """Compare sent emails with stored drafts.

    Match strategies (in order):
      1. conversationId match (reliable)
      2. subject match (reliable)
      3. body similarity brute-force (only if sim ≥ 0.50)
      4. draft_consumed check (draft disappeared from Drafts folder)

    Drafts < 24h old without a reliable match → 'pending' (not counted).
    """
    headers = get_auth_headers()
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    since_str = since.strftime("%Y-%m-%dT%H:%M:%SZ")
    now_utc = datetime.now(timezone.utc)

    drafts = get_recent_drafts(hours=hours, mailbox=mailbox)
    if not drafts:
        return {
            "status": "no_drafts",
            "message": f"No drafts found in the last {hours} hours for mailbox {mailbox}",
            "drafts_checked": 0,
            "mailbox": mailbox,
        }

    # ── Fetch sent emails (paginate up to 500) ──
    sent_emails: list[dict] = []
    async with httpx.AsyncClient(timeout=60) as client:
        sent_url = f"{GRAPH_BASE}/users/{mailbox}/mailFolders/SentItems/messages"
        sent_params = {
            "$top": "100",
            "$filter": f"sentDateTime ge {since_str}",
            "$select": "id,conversationId,subject,body,sentDateTime",
            "$orderby": "sentDateTime desc",
        }
        resp = await client.get(sent_url, headers=headers, params=sent_params)
        if resp.status_code != 200:
            return {"status": "error", "message": f"Failed to get sent items: {resp.status_code}"}
        data = resp.json()
        sent_emails.extend(data.get("value", []))
        next_link = data.get("@odata.nextLink")
        pages = 0
        while next_link and pages < 4:
            resp = await client.get(next_link, headers=headers)
            if resp.status_code != 200:
                break
            data = resp.json()
            sent_emails.extend(data.get("value", []))
            next_link = data.get("@odata.nextLink")
            pages += 1

    # ── Index sent emails ──
    sent_by_conv: dict[str, list[dict]] = {}
    sent_by_subj: dict[str, list[dict]] = {}
    sent_text_cache: dict[str, str] = {}

    for email in sent_emails:
        eid = email.get("id", "")
        conv_id = email.get("conversationId", "")
        if conv_id:
            sent_by_conv.setdefault(conv_id, []).append(email)
        subj = _norm_subject(email.get("subject", ""))
        if subj:
            sent_by_subj.setdefault(subj, []).append(email)
        raw_html = email.get("body", {}).get("content", "")
        sent_text_cache[eid] = _extract_reply_text(raw_html)

    # ── Check which drafts still exist in Drafts folder ──
    draft_ids = [d.get("draft_id", "") for d in drafts if d.get("draft_id")]
    existing_draft_ids: set[str] = set()
    if draft_ids:
        async with httpx.AsyncClient(timeout=60) as client:
            for did in draft_ids:
                try:
                    check_url = f"{GRAPH_BASE}/users/{mailbox}/messages/{did}"
                    resp = await client.get(check_url, headers=headers, params={"$select": "id,isDraft"})
                    if resp.status_code == 200 and resp.json().get("isDraft", False):
                        existing_draft_ids.add(did)
                except Exception:
                    pass

    # ── Results structure ──
    results = {
        "drafts_checked": len(drafts),
        "sent_found": 0,
        "accepted_unchanged": 0,
        "accepted_modified": 0,
        "heavily_modified": 0,
        "not_sent": 0,
        "pending": 0,
        "draft_consumed_no_match": 0,
        # Reliable vs uncertain breakdown
        "reliable_matches": 0,       # conv or subject match
        "uncertain_matches": 0,      # body match (sim ≥ 0.50)
        "match_by_conv": 0,
        "match_by_subject": 0,
        "match_by_body": 0,
        "details": [],
    }

    matched_sent_ids: set[str] = set()

    for draft in drafts:
        conv_id = draft.get("conversation_id", "")
        draft_subj = _norm_subject(draft.get("subject", ""))

        # Clean draft text: strip Hanna banner at HTML level, then extract text
        draft_html = draft.get("draft_html", "")
        clean_html = _strip_hanna_banner_html(draft_html)
        draft_text = _html_to_text(clean_html)

        # Determine draft age
        draft_ts = draft.get("created_at")
        draft_age_hours = None
        if draft_ts:
            try:
                if isinstance(draft_ts, str):
                    dt = datetime.fromisoformat(draft_ts.replace("Z", "+00:00"))
                else:
                    dt = draft_ts
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                draft_age_hours = (now_utc - dt).total_seconds() / 3600
            except Exception:
                pass

        # ── Strategy 1: conversationId match (RELIABLE) ──
        candidates = sent_by_conv.get(conv_id, []) if conv_id else []
        match_method = "conv"

        # ── Strategy 2: subject match (RELIABLE) ──
        if not candidates and draft_subj:
            candidates = sent_by_subj.get(draft_subj, [])
            match_method = "subject"

        # Find best similarity among candidates
        best_sim = 0.0
        best_sent_id = ""
        if candidates:
            for sent in candidates:
                sid = sent.get("id", "")
                if sid in matched_sent_ids:
                    continue
                sent_text = sent_text_cache.get(sid, "")
                sim = _similarity(draft_text, sent_text)
                if sim > best_sim:
                    best_sim = sim
                    best_sent_id = sid

        reliable_match = match_method in ("conv", "subject") and best_sim >= 0.10

        # ── Strategy 3: body brute-force (UNRELIABLE, high threshold) ──
        if not reliable_match and draft_text and len(draft_text) > 30:
            body_best_sim = 0.0
            body_best_id = ""
            for email in sent_emails:
                sid = email.get("id", "")
                if sid in matched_sent_ids:
                    continue
                sent_text = sent_text_cache.get(sid, "")
                if not sent_text or len(sent_text) < 30:
                    continue
                sim = _similarity(draft_text, sent_text)
                if sim > body_best_sim:
                    body_best_sim = sim
                    body_best_id = sid

            # Only accept body match if similarity is high enough
            if body_best_sim >= BODY_MATCH_MIN_SIM:
                best_sim = body_best_sim
                best_sent_id = body_best_id
                match_method = "body"
                reliable_match = False
            # else: don't use body match at all

        # ── Strategy 4: draft consumed check ──
        did = draft.get("draft_id", "")
        draft_consumed = did and did not in existing_draft_ids

        # ── Classify result ──

        has_match = (reliable_match and best_sim >= 0.10) or (match_method == "body" and best_sim >= BODY_MATCH_MIN_SIM)

        if not has_match and not draft_consumed:
            # No match found — is it pending or not_sent?
            if draft_age_hours is not None and draft_age_hours < PENDING_HOURS:
                results["pending"] += 1
                results["details"].append({
                    "subject": draft.get("subject", "")[:60],
                    "confidence": draft.get("confidence", ""),
                    "similarity": 0.0,
                    "match_method": "none",
                    "status": "pending",
                    "draft_age_hours": round(draft_age_hours, 1),
                })
            else:
                results["not_sent"] += 1
            continue

        if draft_consumed and not has_match:
            # Draft disappeared but no similar sent email found
            results["draft_consumed_no_match"] += 1
            results["sent_found"] += 1
            results["details"].append({
                "subject": draft.get("subject", "")[:60],
                "confidence": draft.get("confidence", ""),
                "similarity": 0.0,
                "match_method": "draft_consumed",
                "status": "sent_or_deleted",
            })
            continue

        # We have a match
        results["sent_found"] += 1
        if best_sent_id:
            matched_sent_ids.add(best_sent_id)

        if match_method == "conv":
            results["match_by_conv"] += 1
            results["reliable_matches"] += 1
        elif match_method == "subject":
            results["match_by_subject"] += 1
            results["reliable_matches"] += 1
        else:
            results["match_by_body"] += 1
            results["uncertain_matches"] += 1

        sent_text_matched = sent_text_cache.get(best_sent_id, "")
        detail = {
            "subject": draft.get("subject", "")[:60],
            "confidence": draft.get("confidence", ""),
            "similarity": round(best_sim, 3),
            "match_method": match_method,
            "reliable": match_method in ("conv", "subject"),
            "category": draft.get("category", ""),
            "draft_text": draft_text if best_sim < 0.85 else "",
            "sent_text": sent_text_matched if best_sim < 0.85 else "",
        }

        if best_sim >= UNCHANGED_THRESHOLD:
            results["accepted_unchanged"] += 1
            detail["status"] = "accepted_unchanged"
        elif best_sim >= MODIFIED_THRESHOLD:
            results["accepted_modified"] += 1
            detail["status"] = "modified"
        else:
            results["heavily_modified"] += 1
            detail["status"] = "heavily_modified"

            # Store feedback diff for heavily modified reliable matches
            if match_method in ("conv", "subject") and best_sent_id:
                sent_text = sent_text_cache.get(best_sent_id, "")
                _store_feedback_diff({
                    "id": f"diff_{now_utc.strftime('%Y%m%d_%H%M')}_{draft_subj[:20]}",
                    "created_at": now_utc.isoformat(),
                    "category": draft.get("category", ""),
                    "subject": draft.get("subject", "")[:80],
                    "similarity": round(best_sim, 3),
                    "draft_text": draft_text[:500],
                    "sent_text": sent_text[:500],
                    "key_changes": [],
                    "lesson": "",
                    "match_method": match_method,
                })

        results["details"].append(detail)

    results["status"] = "ok"

    # ── Sync resolved drafts to reasoning_traces (non-blocking) ──
    try:
        await _sync_feedback_to_traces(results.get("details", []), drafts)
    except Exception as e:
        _logger.warning("Failed to sync feedback to reasoning traces: %s", e)

    return results


async def _sync_feedback_to_traces(details: list[dict], drafts: list[dict]) -> None:
    """Write feedback results to reasoning_traces table for learning.

    v6 — 2026-04-07:
      - Find EXISTING trace by email_message_id or subject instead of creating duplicates
      - Run feedback analytics (categorize_changes, chunk_survival, Langfuse export)

    Only syncs drafts that have a definitive outcome (not pending).
    """
    import asyncpg
    from ..reasoning.traces import resolve_trace, create_trace

    resolved_statuses = {"accepted_unchanged", "modified", "heavily_modified"}
    to_sync = [d for d in details if d.get("status") in resolved_statuses]

    if not to_sync:
        return

    conn = await asyncpg.connect(
        os.environ.get("HANNA_PG_DSN", "postgresql://klara:klara_docs_2026@hanna-db:5432/hanna_oetp")
    )
    try:
        for detail in to_sync:
            subject = detail.get("subject", "")
            sim = detail.get("similarity", 0.0)
            draft_text = detail.get("draft_text", "")
            sent_text = detail.get("sent_text", "")

            # Find matching draft for metadata
            matching_draft = None
            for d in drafts:
                if _norm_subject(d.get("subject", ""))[:50] == _norm_subject(subject)[:50]:
                    matching_draft = d
                    break

            # Try to find EXISTING trace (created by processor.py during draft generation)
            # instead of creating a duplicate without top_chunks/query_embedding
            message_id = (matching_draft or {}).get("message_id", "")
            norm_subj = _norm_subject(subject)[:40]
            trace_id = None

            if message_id:
                row = await conn.fetchrow(
                    """SELECT id FROM reasoning_traces
                       WHERE email_message_id = $1 AND outcome = 'PENDING'
                       ORDER BY created_at DESC LIMIT 1""",
                    message_id,
                )
                if row:
                    trace_id = row["id"]

            if not trace_id and norm_subj:
                row = await conn.fetchrow(
                    """SELECT id FROM reasoning_traces
                       WHERE LOWER(LEFT(query_text, 50)) LIKE $1
                         AND outcome = 'PENDING'
                         AND created_at >= NOW() - INTERVAL '72 hours'
                       ORDER BY created_at DESC LIMIT 1""",
                    f"%{norm_subj}%",
                )
                if row:
                    trace_id = row["id"]

            # Fallback: create new trace if no existing one found
            if not trace_id:
                trace_id = await create_trace(
                    conn=conn,
                    query_text=subject,
                    category=detail.get("category", (matching_draft or {}).get("category", "")),
                    confidence=detail.get("confidence", (matching_draft or {}).get("confidence", "")),
                    draft_text=draft_text if draft_text else None,
                    sender_name=(matching_draft or {}).get("sender_name"),
                    sender_email=(matching_draft or {}).get("sender_email"),
                )

            await resolve_trace(
                conn=conn,
                trace_id=trace_id,
                sent_text=sent_text or "",
                similarity_score=sim,
            )

            # ── Run feedback analytics (non-blocking for individual failures) ──
            try:
                from ..reasoning.feedback_analytics import run_analytics_for_feedback
                top_chunks = (matching_draft or {}).get("top_chunks", [])
                category = detail.get("category", (matching_draft or {}).get("category", ""))
                await run_analytics_for_feedback(
                    trace_id=trace_id,
                    draft_text=draft_text,
                    sent_text=sent_text,
                    category=category,
                    top_chunks=top_chunks,
                    metadata={
                        "query_text": subject,
                        "subject": (matching_draft or {}).get("subject", subject),
                        "category": category,
                        "confidence": detail.get("confidence", ""),
                        "similarity": sim,
                        "outcome": detail.get("status", ""),
                        "match_method": detail.get("match_method", ""),
                        "top_chunks": top_chunks,
                    },
                )
            except Exception as e:
                _logger.warning("Feedback analytics failed for %s: %s", subject[:40], e)

        _logger.info("Synced %d feedback results to reasoning_traces", len(to_sync))
    finally:
        await conn.close()
