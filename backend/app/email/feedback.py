"""Feedback loop: compare sent emails with stored drafts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher

import httpx
from bs4 import BeautifulSoup

from .auth import get_auth_headers
from .draft_store import get_recent_drafts

GRAPH_BASE = "https://graph.microsoft.com/v1.0"


def _html_to_text(html: str) -> str:
    """Strip HTML to plain text for comparison."""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n", strip=True)


def _similarity(a: str, b: str) -> float:
    """Return similarity ratio between two strings (0-1)."""
    return SequenceMatcher(None, a, b).ratio()


def _norm_subject(s: str) -> str:
    """Normalize email subject for comparison."""
    import re
    return re.sub(r"^(re:|fw:|fwd:)\s*", "", (s or "").lower().strip())


async def check_feedback(mailbox: str, hours: int = 48) -> dict:
    """Compare sent emails with stored drafts.

    Matches by conversationId first, then falls back to subject + body
    similarity (colleagues often send from a new thread, not the draft).

    Returns stats on how many drafts were accepted unchanged vs modified.
    """
    headers = get_auth_headers()
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    since_str = since.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Get recent drafts from store
    drafts = get_recent_drafts(hours=hours)
    if not drafts:
        return {
            "status": "no_drafts",
            "message": f"No drafts found in the last {hours} hours",
            "drafts_checked": 0,
        }

    # Get sent emails (paginate up to 500)
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

        # Paginate if needed (up to 500)
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

    # Index sent emails by conversationId, normalized subject, and body text
    sent_by_conv: dict[str, list[dict]] = {}
    sent_by_subj: dict[str, list[dict]] = {}
    all_sent_with_text: list[tuple[dict, str]] = []
    for email in sent_emails:
        conv_id = email.get("conversationId", "")
        if conv_id:
            sent_by_conv.setdefault(conv_id, []).append(email)
        subj = _norm_subject(email.get("subject", ""))
        if subj:
            sent_by_subj.setdefault(subj, []).append(email)
        sent_text = _html_to_text(email.get("body", {}).get("content", ""))
        all_sent_with_text.append((email, sent_text))

    # Compare
    results = {
        "drafts_checked": len(drafts),
        "sent_found": 0,
        "accepted_unchanged": 0,
        "accepted_modified": 0,
        "not_sent": 0,
        "match_by_conv": 0,
        "match_by_subject": 0,
        "details": [],
    }

    # Check which drafts still exist in Drafts folder (if gone = sent or deleted)
    draft_ids = [d.get("draft_id", "") for d in drafts if d.get("draft_id")]
    existing_draft_ids: set[str] = set()
    if draft_ids:
        async with httpx.AsyncClient(timeout=60) as client:
            for did in draft_ids:
                try:
                    check_url = f"{GRAPH_BASE}/users/{mailbox}/messages/{did}"
                    resp = await client.get(check_url, headers=headers, params={"$select": "id,isDraft"})
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("isDraft", False):
                            existing_draft_ids.add(did)
                except Exception:
                    pass

    matched_sent_ids: set[str] = set()

    for draft in drafts:
        conv_id = draft.get("conversation_id", "")
        draft_text = _html_to_text(draft.get("draft_html", ""))
        draft_subj = _norm_subject(draft.get("subject", ""))

        # Strategy 1: match by conversationId
        candidates = sent_by_conv.get(conv_id, []) if conv_id else []
        match_method = "conv"

        # Strategy 2: fallback to subject match
        if not candidates and draft_subj:
            candidates = sent_by_subj.get(draft_subj, [])
            match_method = "subject"

        # Find best matching sent email by body similarity
        best_sim = 0.0
        best_sent_id = ""

        if candidates:
            for sent in candidates:
                sid = sent.get("id", "")
                if sid in matched_sent_ids:
                    continue
                sent_html = sent.get("body", {}).get("content", "")
                sent_text = _html_to_text(sent_html)
                sim = _similarity(draft_text, sent_text)
                if sim > best_sim:
                    best_sim = sim
                    best_sent_id = sid

        # Strategy 3: brute-force body similarity against ALL sent emails
        if best_sim < 0.3 and draft_text and len(draft_text) > 30:
            match_method = "body"
            for sent, sent_text in all_sent_with_text:
                sid = sent.get("id", "")
                if sid in matched_sent_ids:
                    continue
                if not sent_text or len(sent_text) < 30:
                    continue
                sim = _similarity(draft_text, sent_text)
                if sim > best_sim:
                    best_sim = sim
                    best_sent_id = sid

        # Check if draft was consumed (no longer in Drafts folder)
        did = draft.get("draft_id", "")
        draft_consumed = did and did not in existing_draft_ids

        if best_sim < 0.15 and not draft_consumed:
            results["not_sent"] += 1
            continue

        if draft_consumed and best_sim < 0.15:
            # Draft was sent/deleted but we can't find similar sent email
            # → colleague likely edited heavily or deleted
            results["sent_found"] += 1
            detail = {
                "subject": draft.get("subject", "")[:60],
                "confidence": draft.get("confidence", ""),
                "similarity": 0.0,
                "match_method": "draft_consumed",
                "status": "sent_or_deleted",
            }
            results["accepted_modified"] += 1
            results["details"].append(detail)
            continue

        results["sent_found"] += 1
        if best_sent_id:
            matched_sent_ids.add(best_sent_id)

        if match_method == "conv":
            results["match_by_conv"] += 1
        else:
            results["match_by_subject"] += 1

        detail = {
            "subject": draft.get("subject", "")[:60],
            "confidence": draft.get("confidence", ""),
            "similarity": round(best_sim, 3),
            "match_method": match_method,
        }

        if best_sim > 0.85:
            results["accepted_unchanged"] += 1
            detail["status"] = "accepted_unchanged"
        elif best_sim > 0.3:
            results["accepted_modified"] += 1
            detail["status"] = "modified"
        else:
            results["accepted_modified"] += 1
            detail["status"] = "heavily_modified"

        results["details"].append(detail)

    results["status"] = "ok"
    return results
