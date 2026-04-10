"""Draft creation via Microsoft Graph API."""

from __future__ import annotations

import re

import httpx

from ..config import settings
from ..models import DraftResult
from .auth import get_auth_headers

from bs4 import BeautifulSoup

GRAPH_BASE = "https://graph.microsoft.com/v1.0"


def _get_fallback_html() -> str:
    """Get fallback draft HTML from program.yaml signature."""
    from ..config import get_program_config
    pcfg = get_program_config()
    sig = pcfg.get("email", {}).get("signature", "")
    if sig:
        lines = sig.strip().split("\n")
        sig_html = "<p>" + "<br>".join(l.strip() for l in lines if l.strip()) + "</p>"
    else:
        sig_html = "<p>Üdvözlettel:<br>Ügyfélszolgálat</p>"
    return f"<p>Kérdésére kollégánk hamarosan válaszol.</p>{sig_html}"


def _final_safety_check(body_html: str, confidence: str) -> str:
    """Last-resort safety check on the draft HTML before saving to Outlook.

    Catches:
    1. Accent-free Hungarian text (should never happen after all guards)
    2. Empty or whitespace-only body

    If any issue found, replaces with a safe fallback message.
    """
    if not body_html or not body_html.strip():
        return _get_fallback_html()

    # Check for accent-free Hungarian text
    plain = BeautifulSoup(body_html, "html.parser").get_text()
    accent_chars = set("áéíóöőúüűÁÉÍÓÖŐÚÜŰ")

    if len(plain) > 80 and not any(c in accent_chars for c in plain):
        print("[drafts] BLOCKED: accent-free draft detected at final gate!")
        return _get_fallback_html()

    return body_html

# Outlook categories applied to the original email after draft creation
CATEGORY_MAP = {
    "high": "CC - draft kész",
    "medium": "CC - review kell",
    "low": "CC - emberi válasz kell",
}

# All CC category values (for filtering)
CC_CATEGORIES = {
    "CC - draft kész",
    "CC - review kell",
    "CC - emberi válasz kell",
    "CC - elküldve",
    "CC - nem kell válasz",
}


async def _set_email_category(
    client: httpx.AsyncClient,
    headers: dict,
    mailbox: str,
    message_id: str,
    confidence: str,
) -> None:
    """Add a CC category to the original email WITHOUT removing existing categories."""
    category = CATEGORY_MAP.get(confidence, CATEGORY_MAP["medium"])
    url = f"{GRAPH_BASE}/users/{mailbox}/messages/{message_id}"
    try:
        # First, GET existing categories so we don't overwrite them
        get_resp = await client.get(
            url, headers=headers, params={"$select": "categories"},
        )
        existing: list[str] = []
        if get_resp.status_code == 200:
            existing = get_resp.json().get("categories", [])

        # Remove any previous CC categories, then add the new one
        cc_prefixes = tuple(CATEGORY_MAP.values())
        merged = [c for c in existing if c not in cc_prefixes]
        merged.append(category)

        resp = await client.patch(url, headers=headers, json={"categories": merged})
        if resp.status_code == 200:
            print(f"[drafts] Category '{category}' added on {message_id[:20]}... (kept {len(merged)-1} existing)")
        else:
            print(f"[drafts] Failed to set category: {resp.status_code}")
    except Exception as e:
        print(f"[drafts] Category error (non-fatal): {e}")


async def create_reply_draft(
    mailbox: str,
    reply_to_message_id: str,
    body_html: str,
    confidence: str = "medium",
    top_chunks: list[dict] | None = None,
    category: str = "",
    sender_name: str = "",
    sender_email: str = "",
) -> DraftResult:
    """Create a draft reply in a shared mailbox.

    This uses the Graph API createReply endpoint which:
    - Creates a draft in the Drafts folder
    - Sets the proper In-Reply-To headers
    - Includes the original message in the body
    - Does NOT send the email

    An operator then reviews and sends (or discards) the draft.
    """
    headers = get_auth_headers()

    # Step 1: Create reply draft
    url = f"{GRAPH_BASE}/users/{mailbox}/messages/{reply_to_message_id}/createReply"

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, headers=headers, json={})

        if resp.status_code not in (200, 201):
            raise RuntimeError(
                f"Failed to create reply draft: {resp.status_code} {resp.text}"
            )

        draft = resp.json()
        draft_id = draft["id"]
        subject = draft.get("subject", "")

        # Preserve the original quoted email from the createReply response
        # Strip <html><head>...</head><body> wrapper to avoid nested HTML
        raw_body = draft.get("body", {}).get("content", "")
        original_body = re.sub(
            r"<html[^>]*>.*?<body[^>]*>", "", raw_body,
            count=1, flags=re.DOTALL | re.IGNORECASE,
        )
        original_body = re.sub(
            r"</body>\s*</html>\s*$", "", original_body,
            flags=re.IGNORECASE,
        )

        # Step 2: Update the draft body with our generated content
        # Add confidence icon at the top (icon only, no text — operator sees the Outlook category for details)
        confidence_icon = {
            "high": '<div style="padding:4px;margin-bottom:12px;font-size:18px" title="CC - Magabiztos válasz">🟢</div>',
            "medium": '<div style="padding:4px;margin-bottom:12px;font-size:18px" title="CC - Részben biztos, kérlek ellenőrizd">🟡</div>',
            "low": '<div style="padding:4px;margin-bottom:12px;font-size:18px" title="CC - Bizonytalan, emberi válasz javasolt">🔴</div>',
        }
        banner = confidence_icon.get(confidence, confidence_icon["medium"])

        # Final safety checks before saving
        body_html = _final_safety_check(body_html, confidence)

        # Ensure proper UTF-8 encoding for Hungarian characters
        meta_charset = '<meta http-equiv="Content-Type" content="text/html; charset=utf-8">'
        # Prepend our response before the original quoted email
        full_body = f"{meta_charset}\n{banner}\n{body_html}\n<br/>\n<hr/>\n{original_body}"

        update_url = f"{GRAPH_BASE}/users/{mailbox}/messages/{draft_id}"
        update_resp = await client.patch(
            update_url,
            headers=headers,
            json={
                "body": {
                    "contentType": "HTML",
                    "content": full_body,
                },
            },
        )

        if update_resp.status_code != 200:
            raise RuntimeError(
                f"Failed to update draft body: {update_resp.status_code} {update_resp.text}"
            )

        # Step 3: Mark the original email with an Outlook category
        await _set_email_category(
            client, headers, mailbox, reply_to_message_id, confidence
        )

    # Save to draft store for feedback tracking
    try:
        from .draft_store import save_draft
        conversation_id = draft.get("conversationId", "")
        save_draft(
            conversation_id=conversation_id,
            message_id=reply_to_message_id,
            mailbox=mailbox,
            draft_html=body_html,
            confidence=confidence,
            draft_id=draft_id,
            subject=subject,
            top_chunks=top_chunks,
            category=category,
            sender_name=sender_name,
            sender_email=sender_email,
        )
        print(f"[drafts] Saved to draft store: {subject[:40]}...")
    except Exception as e:
        print(f"[drafts] Draft store save failed (non-fatal): {e}")

    return DraftResult(
        draft_id=draft_id,
        mailbox=mailbox,
        subject=subject,
        confidence=confidence,
    )


async def list_drafts(mailbox: str, limit: int = 20) -> list[dict]:
    """List recent drafts in a shared mailbox."""
    headers = get_auth_headers()

    url = f"{GRAPH_BASE}/users/{mailbox}/mailFolders/Drafts/messages"
    params = {
        "$top": limit,
        "$orderby": "lastModifiedDateTime desc",
        "$select": "id,subject,lastModifiedDateTime,body",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, headers=headers, params=params)

        if resp.status_code == 200:
            return resp.json().get("value", [])

    return []


async def mark_sent_emails(mailbox: str, hours: int = 4) -> dict:
    """Check Sent Items and mark original emails as 'CC - elküldve'.
    
    This function:
    1. Fetches recent sent emails (last N hours)
    2. For each sent email that is a reply (has conversationId)
    3. Finds the original email in Inbox with matching conversationId
    4. If the original has a CC category (draft kész, review kell, etc.)
    5. Updates it to 'CC - elküldve' while preserving non-CC categories
    
    Returns dict with counts of processed/updated/errors.
    """
    from datetime import datetime, timedelta, timezone
    
    headers = get_auth_headers()
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    since_str = since.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    results = {"checked": 0, "updated": 0, "already_sent": 0, "no_cc_draft": 0, "errors": 0}
    
    async with httpx.AsyncClient(timeout=60) as client:
        # Get recent sent emails
        sent_url = f"{GRAPH_BASE}/users/{mailbox}/mailFolders/SentItems/messages"
        sent_params = {
            "$top": "100",
            "$filter": f"sentDateTime ge {since_str}",
            "$select": "id,conversationId,subject,sentDateTime",
        }
        
        sent_resp = await client.get(sent_url, headers=headers, params=sent_params)
        if sent_resp.status_code != 200:
            print(f"[mark_sent] Failed to get sent items: {sent_resp.status_code}")
            return results
        
        sent_emails = sent_resp.json().get("value", [])
        results["checked"] = len(sent_emails)
        
        # Group by conversationId
        conversation_ids = set()
        for email in sent_emails:
            conv_id = email.get("conversationId")
            if conv_id:
                conversation_ids.add(conv_id)
        
        if not conversation_ids:
            return results
        
        # For each conversation, find the original email in Inbox
        inbox_url = f"{GRAPH_BASE}/users/{mailbox}/mailFolders/Inbox/messages"
        
        for conv_id in conversation_ids:
            try:
                # Find original emails in this conversation
                inbox_params = {
                    "$filter": f"conversationId eq '{conv_id}'",
                    "$select": "id,subject,categories,receivedDateTime",
                    "$orderby": "receivedDateTime asc",
                    "$top": "10",
                }
                
                inbox_resp = await client.get(inbox_url, headers=headers, params=inbox_params)
                if inbox_resp.status_code != 200:
                    # Try without $orderby (shared mailbox limitation)
                    del inbox_params["$orderby"]
                    inbox_resp = await client.get(inbox_url, headers=headers, params=inbox_params)
                    if inbox_resp.status_code != 200:
                        results["errors"] += 1
                        continue
                
                inbox_emails = inbox_resp.json().get("value", [])
                
                for orig_email in inbox_emails:
                    orig_id = orig_email["id"]
                    categories = orig_email.get("categories", [])
                    
                    # Check if it has a CC category that should be updated
                    has_cc_draft = any(c in CC_CATEGORIES for c in categories)
                    already_sent = "CC - elküldve" in categories
                    
                    if already_sent:
                        results["already_sent"] += 1
                        continue
                    
                    if not has_cc_draft:
                        results["no_cc_draft"] += 1
                        continue
                    
                    # Update: remove old CC categories, add "elküldve", keep others
                    new_categories = [c for c in categories if c not in CC_CATEGORIES]
                    new_categories.append("CC - elküldve")
                    
                    update_url = f"{GRAPH_BASE}/users/{mailbox}/messages/{orig_id}"
                    update_resp = await client.patch(
                        update_url,
                        headers=headers,
                        json={"categories": new_categories},
                    )
                    
                    if update_resp.status_code == 200:
                        results["updated"] += 1
                        subj = orig_email.get("subject", "")[:40]
                        print(f"[mark_sent] Updated to 'elküldve': {subj}...")
                    else:
                        results["errors"] += 1
                        print(f"[mark_sent] Failed to update {orig_id[:20]}: {update_resp.status_code}")
                        
            except Exception as e:
                print(f"[mark_sent] Error processing conversation: {e}")
                results["errors"] += 1
    
    return results
