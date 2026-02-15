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


async def check_feedback(mailbox: str, hours: int = 48) -> dict:
    """Compare sent emails with stored drafts.
    
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
    
    # Get sent emails
    async with httpx.AsyncClient(timeout=60) as client:
        sent_url = f"{GRAPH_BASE}/users/{mailbox}/mailFolders/SentItems/messages"
        sent_params = {
            "$top": "100",
            "$filter": f"sentDateTime ge {since_str}",
            "$select": "id,conversationId,subject,body,sentDateTime",
        }
        
        resp = await client.get(sent_url, headers=headers, params=sent_params)
        if resp.status_code != 200:
            return {"status": "error", "message": f"Failed to get sent items: {resp.status_code}"}
        
        sent_emails = resp.json().get("value", [])
    
    # Index sent emails by conversationId
    sent_by_conv: dict[str, list[dict]] = {}
    for email in sent_emails:
        conv_id = email.get("conversationId", "")
        if conv_id:
            sent_by_conv.setdefault(conv_id, []).append(email)
    
    # Compare
    results = {
        "drafts_checked": len(drafts),
        "sent_found": 0,
        "accepted_unchanged": 0,
        "accepted_modified": 0,
        "not_sent": 0,
        "details": [],
    }
    
    for draft in drafts:
        conv_id = draft.get("conversation_id", "")
        if not conv_id or conv_id not in sent_by_conv:
            results["not_sent"] += 1
            continue
        
        results["sent_found"] += 1
        draft_text = _html_to_text(draft.get("draft_html", ""))
        
        # Find best matching sent email
        best_sim = 0.0
        for sent in sent_by_conv[conv_id]:
            sent_html = sent.get("body", {}).get("content", "")
            sent_text = _html_to_text(sent_html)
            sim = _similarity(draft_text, sent_text)
            best_sim = max(best_sim, sim)
        
        detail = {
            "subject": draft.get("subject", "")[:60],
            "confidence": draft.get("confidence", ""),
            "similarity": round(best_sim, 3),
        }
        
        if best_sim > 0.85:
            results["accepted_unchanged"] += 1
            detail["status"] = "accepted_unchanged"
        else:
            results["accepted_modified"] += 1
            detail["status"] = "modified"
        
        results["details"].append(detail)
    
    results["status"] = "ok"
    return results
