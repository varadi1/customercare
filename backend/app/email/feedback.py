"""Feedback loop: compare sent emails with stored drafts."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher

import httpx
from bs4 import BeautifulSoup

from .auth import get_auth_headers
from .draft_store import get_recent_drafts

GRAPH_BASE = "https://graph.microsoft.com/v1.0"


def _html_to_text(html: str) -> str:
    """Strip HTML to plain text for comparison."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n", strip=True)


def _strip_hanna_banner_text(text: str) -> str:
    """Remove Hanna AI Draft banner from plain text."""
    if not text:
        return text
    # Remove lines containing Hanna banner markers
    lines = text.split("\n")
    clean_lines = []
    skip = False
    for line in lines:
        stripped = line.strip()
        # Start of Hanna banner
        if "Hanna AI Draft" in stripped and ("Confidence" in stripped or "Forrás" in stripped):
            skip = True
            continue
        if skip:
            # Banner typically ends before "Tisztelt" or after an empty line following source info
            if stripped.startswith("Tisztelt") or stripped.startswith("Köszön"):
                skip = False
                clean_lines.append(line)
            elif "Forrás:" in stripped or "Dash:" in stripped or "Pályázó kérdése:" in stripped:
                continue  # still in banner
            elif stripped == "":
                skip = False  # empty line = end of banner
            else:
                skip = False
                clean_lines.append(line)
        else:
            # Also catch single-line banner format
            if "🤖 Hanna AI Draft" in stripped:
                continue
            clean_lines.append(line)
    return "\n".join(clean_lines)


def _extract_reply_text(html: str) -> str:
    """Extract only the reply portion from a sent email HTML, removing quoted original.
    
    Works at HTML level to identify quote boundaries, then extracts text.
    Handles: divRplyFwdMsg, border-top #E1E1E1, <hr> dividers, Original Message.
    """
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    
    # Strategy 1: <div id="divRplyFwdMsg"> (Desktop Outlook)
    reply_div = soup.find(id="divRplyFwdMsg")
    if reply_div:
        # Get all text BEFORE this div
        texts = []
        for el in reply_div.previous_siblings:
            t = el.get_text(separator="\n", strip=True) if hasattr(el, 'get_text') else str(el).strip()
            if t:
                texts.append(t)
        # Also remove <hr> separators from the text
        result = "\n".join(reversed(texts))
        return result.strip()
    
    # Strategy 2: border-top solid #E1E1E1 (OWA/new Outlook)
    for div in soup.find_all("div"):
        style = div.get("style", "") or ""
        if "border-top" in style and "#E1E1E1" in style:
            texts = []
            for el in div.previous_siblings:
                t = el.get_text(separator="\n", strip=True) if hasattr(el, 'get_text') else str(el).strip()
                if t:
                    texts.append(t)
            result = "\n".join(reversed(texts))
            return result.strip()
    
    # Strategy 3: text-level — find "From:" / "Feladó:" pattern
    full_text = soup.get_text(separator="\n", strip=True)
    
    # Look for common reply divider patterns
    patterns = [
        r"\n\s*From:.*?\nSent:",  # English Outlook
        r"\n\s*Feladó:.*?\nElküldve:",  # Hungarian Outlook
        r"-{3,}\s*Original Message\s*-{3,}",  # Original Message marker
    ]
    
    for pattern in patterns:
        match = re.search(pattern, full_text, re.IGNORECASE | re.DOTALL)
        if match:
            # Return everything before the match
            # Also strip any trailing <hr> equivalent (dashes, underscores)
            reply_text = full_text[:match.start()]
            reply_text = re.sub(r"\n\s*[-_=]{3,}\s*$", "", reply_text)
            return reply_text.strip()
    
    # No quote found — return full text
    return full_text


def _similarity(a: str, b: str) -> float:
    """Return similarity ratio between two strings (0-1)."""
    return SequenceMatcher(None, a, b).ratio()


def _norm_subject(s: str) -> str:
    """Normalize email subject for comparison."""
    return re.sub(r"^(re:|fw:|fwd:)\s*", "", (s or "").lower().strip())


async def check_feedback(mailbox: str, hours: int = 48) -> dict:
    """Compare sent emails with stored drafts."""
    headers = get_auth_headers()
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    since_str = since.strftime("%Y-%m-%dT%H:%M:%SZ")

    drafts = get_recent_drafts(hours=hours, mailbox=mailbox)
    if not drafts:
        return {
            "status": "no_drafts",
            "message": f"No drafts found in the last {hours} hours for mailbox {mailbox}",
            "drafts_checked": 0,
            "mailbox": mailbox,
        }

    # Get sent emails
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

    # Index sent emails
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
        # Extract reply-only text
        raw_html = email.get("body", {}).get("content", "")
        sent_text_cache[eid] = _extract_reply_text(raw_html)

    results = {
        "drafts_checked": len(drafts),
        "sent_found": 0,
        "accepted_unchanged": 0,
        "accepted_modified": 0,
        "not_sent": 0,
        "match_by_conv": 0,
        "match_by_subject": 0,
        "match_by_body": 0,
        "details": [],
    }

    # Check which drafts still exist
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

    matched_sent_ids: set[str] = set()

    for draft in drafts:
        conv_id = draft.get("conversation_id", "")
        # Clean draft text: strip Hanna banner
        draft_html = draft.get("draft_html", "")
        draft_text_raw = _html_to_text(draft_html)
        draft_text = _strip_hanna_banner_text(draft_text_raw)
        draft_subj = _norm_subject(draft.get("subject", ""))

        candidates = sent_by_conv.get(conv_id, []) if conv_id else []
        match_method = "conv"

        if not candidates and draft_subj:
            candidates = sent_by_subj.get(draft_subj, [])
            match_method = "subject"

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

        if best_sim < 0.3 and draft_text and len(draft_text) > 30:
            for email in sent_emails:
                sid = email.get("id", "")
                if sid in matched_sent_ids:
                    continue
                sent_text = sent_text_cache.get(sid, "")
                if not sent_text or len(sent_text) < 30:
                    continue
                sim = _similarity(draft_text, sent_text)
                if sim > best_sim:
                    best_sim = sim
                    best_sent_id = sid
                    match_method = "body"

        did = draft.get("draft_id", "")
        draft_consumed = did and did not in existing_draft_ids

        if best_sim < 0.15 and not draft_consumed:
            results["not_sent"] += 1
            continue

        if draft_consumed and best_sim < 0.15:
            results["sent_found"] += 1
            results["accepted_modified"] += 1
            results["details"].append({
                "subject": draft.get("subject", "")[:60],
                "confidence": draft.get("confidence", ""),
                "similarity": 0.0,
                "match_method": "draft_consumed",
                "status": "sent_or_deleted",
            })
            continue

        results["sent_found"] += 1
        if best_sent_id:
            matched_sent_ids.add(best_sent_id)

        if match_method == "conv":
            results["match_by_conv"] += 1
        elif match_method == "subject":
            results["match_by_subject"] += 1
        else:
            results["match_by_body"] += 1

        # Body matches are less reliable — flag them
        match_reliable = match_method in ("conv", "subject")
        detail = {
            "subject": draft.get("subject", "")[:60],
            "confidence": draft.get("confidence", ""),
            "similarity": round(best_sim, 3),
            "match_method": match_method,
            "match_reliable": match_reliable,
        }

        if best_sim > 0.85:
            results["accepted_unchanged"] += 1
            detail["status"] = "accepted_unchanged"
        elif best_sim > 0.3:
            results["accepted_modified"] += 1
            detail["status"] = "modified"
        elif not match_reliable:
            # Body match with low similarity — likely wrong match
            results["accepted_modified"] += 1
            detail["status"] = "uncertain_match"
        else:
            results["accepted_modified"] += 1
            detail["status"] = "heavily_modified"

        results["details"].append(detail)

    results["status"] = "ok"
    return results
