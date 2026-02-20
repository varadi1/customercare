"""Draft creation via Microsoft Graph API."""

from __future__ import annotations

import re

import httpx

from ..config import settings
from ..models import DraftResult
from .auth import get_auth_headers

GRAPH_BASE = "https://graph.microsoft.com/v1.0"

# Outlook categories applied to the original email after draft creation
CATEGORY_MAP = {
    "high": "Hanna - draft kész",
    "medium": "Hanna - review kell",
    "low": "Hanna - emberi válasz kell",
}

# All Hanna category values (for filtering)
HANNA_CATEGORIES = {
    "Hanna - draft kész",
    "Hanna - review kell",
    "Hanna - emberi válasz kell",
    "Hanna - elküldve",
    "Hanna - nem kell válasz",
}


async def _set_email_category(
    client: httpx.AsyncClient,
    headers: dict,
    mailbox: str,
    message_id: str,
    confidence: str,
) -> None:
    """Add a Hanna category to the original email WITHOUT removing existing categories."""
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

        # Remove any previous Hanna categories, then add the new one
        hanna_prefixes = tuple(CATEGORY_MAP.values())
        merged = [c for c in existing if c not in hanna_prefixes]
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
        # Add confidence banner at the top
        confidence_banner = {
            "high": '<div style="background:#d4edda;padding:8px;border-radius:4px;margin-bottom:16px">🟢 <b>Hanna - Magabiztos válasz</b></div>',
            "medium": '<div style="background:#fff3cd;padding:8px;border-radius:4px;margin-bottom:16px">🟡 <b>Hanna - Részben biztos, kérlek ellenőrizd</b></div>',
            "low": '<div style="background:#f8d7da;padding:8px;border-radius:4px;margin-bottom:16px">🔴 <b>Hanna - Bizonytalan, emberi válasz javasolt</b></div>',
        }
        banner = confidence_banner.get(confidence, confidence_banner["medium"])

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
    """Check Sent Items and mark original emails as 'Hanna - elküldve'.
    
    This function:
    1. Fetches recent sent emails (last N hours)
    2. For each sent email that is a reply (has conversationId)
    3. Finds the original email in Inbox with matching conversationId
    4. If the original has a Hanna category (draft kész, review kell, etc.)
    5. Updates it to 'Hanna - elküldve' while preserving non-Hanna categories
    
    Returns dict with counts of processed/updated/errors.
    """
    from datetime import datetime, timedelta, timezone
    
    headers = get_auth_headers()
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    since_str = since.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    results = {"checked": 0, "updated": 0, "already_sent": 0, "no_hanna": 0, "errors": 0}
    
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
                    
                    # Check if it has a Hanna category that should be updated
                    has_hanna = any(c in HANNA_CATEGORIES for c in categories)
                    already_sent = "Hanna - elküldve" in categories
                    
                    if already_sent:
                        results["already_sent"] += 1
                        continue
                    
                    if not has_hanna:
                        results["no_hanna"] += 1
                        continue
                    
                    # Update: remove old Hanna categories, add "elküldve", keep others
                    new_categories = [c for c in categories if c not in HANNA_CATEGORIES]
                    new_categories.append("Hanna - elküldve")
                    
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
