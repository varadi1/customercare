#!/usr/bin/env python3
"""
Regenerate all existing Hanna drafts in the Drafts folder.

Fetches each Hanna draft → finds the original email it replies to →
calls /draft/generate with the original email text → overwrites the draft body.

Usage:
  docker exec hanna-backend python3 /app/scripts/regenerate_drafts.py
  docker exec hanna-backend python3 /app/scripts/regenerate_drafts.py --dry-run
  docker exec hanna-backend python3 /app/scripts/regenerate_drafts.py --mailbox info@neuzrt.hu
"""
import argparse
import asyncio
import json
import logging
import os
import re
import sys

sys.path.insert(0, "/app")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("regenerate_drafts")

GRAPH_BASE = "https://graph.microsoft.com/v1.0"


async def get_hanna_drafts(headers: dict, mailbox: str) -> list[dict]:
    """Fetch all drafts from the Drafts folder."""
    import httpx

    drafts = []
    async with httpx.AsyncClient(timeout=60) as client:
        url = f"{GRAPH_BASE}/users/{mailbox}/mailFolders/Drafts/messages"
        params = {
            "$top": "50",
            "$orderby": "lastModifiedDateTime desc",
            "$select": "id,subject,conversationId,body,lastModifiedDateTime,categories",
        }
        resp = await client.get(url, headers=headers, params=params)
        if resp.status_code != 200:
            logger.error("Failed to list drafts: %s", resp.status_code)
            return []
        drafts = resp.json().get("value", [])

    # Filter: only drafts where the conversation does NOT already have an internal reply
    # (if colleague already replied, don't regenerate)
    result = []
    our_domains = {"neuzrt.hu", "nffku.hu"}

    async with httpx.AsyncClient(timeout=30) as client:
        for d in drafts:
            conv_id = d.get("conversationId", "")
            subj = d.get("subject", "")
            clean_subj = re.sub(r"^(RE|FW|Fwd):\s*", "", subj, flags=re.IGNORECASE).strip()

            # Check: has a colleague already sent a reply in this conversation?
            has_colleague_reply = False
            if clean_subj:
                params = {
                    "$search": f'"subject:{clean_subj[:50]}"',
                    "$top": "20",
                    "$select": "id,subject,from,isDraft,conversationId,parentFolderId",
                }
                resp = await client.get(
                    f"{GRAPH_BASE}/users/{mailbox}/messages",
                    headers=headers,
                    params=params,
                )
                if resp.status_code == 200:
                    for msg in resp.json().get("value", []):
                        if msg.get("isDraft"):
                            continue
                        # Same conversation?
                        if conv_id and msg.get("conversationId") != conv_id:
                            continue
                        sender = msg.get("from", {}).get("emailAddress", {}).get("address", "")
                        domain = sender.split("@")[-1].lower() if "@" in sender else ""
                        if domain in our_domains:
                            has_colleague_reply = True
                            break

            if has_colleague_reply:
                logger.info("  Skipping %s — colleague already replied", subj[:50])
            else:
                result.append(d)

    logger.info("Filtered: %d drafts to regenerate (skipped %d with colleague reply)",
                len(result), len(drafts) - len(result))
    return result


async def find_original_email(headers: dict, mailbox: str, conversation_id: str, draft_subject: str) -> dict | None:
    """Find the original email that a draft replies to.

    Strategy: use the conversation thread endpoint (not $filter on conversationId).
    """
    import httpx

    our_domains = {"neuzrt.hu", "nffku.hu"}

    async with httpx.AsyncClient(timeout=30) as client:
        # Strategy 1: Search by subject in Inbox (strip RE:/FW: prefix)
        clean_subject = re.sub(r"^(RE|FW|Fwd):\s*", "", draft_subject, flags=re.IGNORECASE).strip()
        if clean_subject:
            url = f"{GRAPH_BASE}/users/{mailbox}/messages"
            # $search cannot be combined with $orderby in Graph API
            params = {
                "$search": f'"subject:{clean_subject[:50]}"',
                "$top": "10",
                "$select": "id,subject,from,body,receivedDateTime,isDraft,conversationId",
            }
            resp = await client.get(url, headers=headers, params=params)
            if resp.status_code == 200:
                messages = resp.json().get("value", [])
                # Filter: same conversation, not draft, from external sender
                for msg in messages:
                    if msg.get("isDraft"):
                        continue
                    # Prefer same conversation
                    if conversation_id and msg.get("conversationId") != conversation_id:
                        continue
                    sender_email = msg.get("from", {}).get("emailAddress", {}).get("address", "")
                    sender_domain = sender_email.split("@")[-1].lower() if "@" in sender_email else ""
                    if sender_domain not in our_domains:
                        return {
                            "id": msg["id"],
                            "subject": msg.get("subject", ""),
                            "sender_name": msg.get("from", {}).get("emailAddress", {}).get("name", ""),
                            "sender_email": sender_email,
                            "body_text": _html_to_text(msg.get("body", {}).get("content", "")),
                        }

                # If no conversation match, try any external sender with matching subject
                for msg in messages:
                    if msg.get("isDraft"):
                        continue
                    sender_email = msg.get("from", {}).get("emailAddress", {}).get("address", "")
                    sender_domain = sender_email.split("@")[-1].lower() if "@" in sender_email else ""
                    if sender_domain not in our_domains:
                        return {
                            "id": msg["id"],
                            "subject": msg.get("subject", ""),
                            "sender_name": msg.get("from", {}).get("emailAddress", {}).get("name", ""),
                            "sender_email": sender_email,
                            "body_text": _html_to_text(msg.get("body", {}).get("content", "")),
                        }

    return None


def _html_to_text(html: str) -> str:
    """Strip HTML to plain text."""
    from bs4 import BeautifulSoup
    if not html:
        return ""
    return BeautifulSoup(html, "html.parser").get_text(separator="\n", strip=True)


def _extract_oetp_ids(text: str) -> list[str]:
    """Extract OETP-YYYY-NNNNN IDs from text."""
    return list(set(re.findall(r"OETP-\d{4}-\d+", text)))


async def regenerate_draft(
    headers: dict,
    mailbox: str,
    draft: dict,
    original: dict,
    dry_run: bool = False,
) -> dict:
    """Regenerate a single draft.

    Strategy: delete old draft → createReply on original → update body.
    This preserves the quoted thread (createReply adds it automatically).
    """
    import httpx

    email_text = original["body_text"]
    email_subject = original["subject"]
    sender_name = original["sender_name"]
    sender_email = original["sender_email"]
    oetp_ids = _extract_oetp_ids(email_text + " " + email_subject)

    # Call /draft/generate endpoint
    async with httpx.AsyncClient(timeout=120, base_url="http://localhost:8000") as client:
        resp = await client.post("/draft/generate", json={
            "email_text": email_text[:3000],
            "email_subject": email_subject,
            "sender_name": sender_name,
            "sender_email": sender_email,
            "oetp_ids": oetp_ids,
        })
        if resp.status_code != 200:
            return {"status": "error", "error": f"generate failed: {resp.status_code}"}

        result = resp.json()

    if result.get("skip"):
        return {"status": "skip", "reason": result.get("skip_reason", "insufficient info")}

    new_body_html = result.get("body_html", "")
    confidence = result.get("confidence", "medium")

    if not new_body_html:
        return {"status": "error", "error": "empty body from generate"}

    if dry_run:
        return {
            "status": "dry_run",
            "confidence": confidence,
            "body_preview": _html_to_text(new_body_html)[:200],
        }

    async with httpx.AsyncClient(timeout=30) as client:
        old_draft_id = draft["id"]
        original_msg_id = original["id"]

        # Step 1: Delete old draft
        del_resp = await client.delete(
            f"{GRAPH_BASE}/users/{mailbox}/messages/{old_draft_id}",
            headers=headers,
        )
        if del_resp.status_code not in (200, 204):
            logger.warning("Failed to delete old draft: %s", del_resp.status_code)

        # Step 2: Create new reply draft (preserves quoted thread)
        reply_resp = await client.post(
            f"{GRAPH_BASE}/users/{mailbox}/messages/{original_msg_id}/createReply",
            headers=headers,
            json={},
        )
        if reply_resp.status_code != 201:
            return {"status": "error", "error": f"createReply failed: {reply_resp.status_code}"}

        new_draft = reply_resp.json()
        new_draft_id = new_draft.get("id", "")

        # Step 3: Extract the quoted thread from the new reply draft
        new_draft_body = new_draft.get("body", {}).get("content", "")
        quoted_thread = _extract_quoted_thread(new_draft_body)

        # Step 4: Update the body with Hanna's response + preserved thread
        combined_body = new_body_html + quoted_thread

        update_resp = await client.patch(
            f"{GRAPH_BASE}/users/{mailbox}/messages/{new_draft_id}",
            headers=headers,
            json={
                "body": {
                    "contentType": "HTML",
                    "content": combined_body,
                },
            },
        )
        if update_resp.status_code != 200:
            return {"status": "error", "error": f"update failed: {update_resp.status_code}"}

    return {"status": "updated", "confidence": confidence, "new_draft_id": new_draft_id}


def _extract_quoted_thread(html: str) -> str:
    """Extract the quoted reply thread from a createReply draft body.

    The createReply response includes an empty body + quoted original.
    We want everything from the reply separator onwards.
    """
    if not html:
        return ""

    # Common separators in Outlook reply drafts
    separators = [
        '<div id="divRplyFwdMsg"',          # Desktop Outlook
        '<div style="border:none;border-top:solid #E1E1E1',  # OWA
        '<hr style="display:inline-block',   # HR separator
    ]

    for sep in separators:
        idx = html.find(sep)
        if idx >= 0:
            return html[idx:]

    # Fallback: look for "From:" or "Feladó:" text pattern
    for pattern in [r'<b>From:</b>', r'<b>Feladó:</b>', r'<b>Küldés ideje:</b>']:
        import re
        match = re.search(pattern, html, re.IGNORECASE)
        if match:
            # Go back to find the containing div
            start = html.rfind('<div', 0, match.start())
            if start >= 0:
                return html[start:]

    return ""


async def main():
    parser = argparse.ArgumentParser(description="Regenerate Hanna drafts")
    parser.add_argument("--mailbox", type=str, default="", help="Specific mailbox (default: all shared)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change, don't update")
    args = parser.parse_args()

    from app.email.auth import get_auth_headers
    from app.config import settings

    headers = get_auth_headers()

    if args.mailbox:
        mailboxes = [args.mailbox]
    else:
        mailboxes = [m.strip() for m in settings.shared_mailboxes.split(",") if m.strip()]

    total = {"found": 0, "updated": 0, "skipped": 0, "errors": 0, "no_original": 0}

    for mb in mailboxes:
        logger.info("=== Mailbox: %s ===", mb)

        drafts = await get_hanna_drafts(headers, mb)
        logger.info("Found %d Hanna drafts", len(drafts))
        total["found"] += len(drafts)

        for i, draft in enumerate(drafts, 1):
            subject = draft.get("subject", "???")[:60]
            conv_id = draft.get("conversationId", "")

            logger.info("[%d/%d] %s", i, len(drafts), subject)

            # Find original email
            original = await find_original_email(headers, mb, conv_id, subject)
            if not original:
                logger.warning("  No original email found — skipping")
                total["no_original"] += 1
                continue

            logger.info("  Original from: %s <%s>", original["sender_name"], original["sender_email"])
            oetp_ids = _extract_oetp_ids(original["body_text"] + " " + subject)
            if oetp_ids:
                logger.info("  OETP IDs: %s", oetp_ids)

            # Regenerate
            result = await regenerate_draft(headers, mb, draft, original, dry_run=args.dry_run)

            status = result["status"]
            if status == "updated":
                logger.info("  UPDATED (confidence: %s)", result["confidence"])
                total["updated"] += 1
            elif status == "dry_run":
                logger.info("  DRY RUN (confidence: %s): %s", result["confidence"], result["body_preview"][:100])
                total["updated"] += 1
            elif status == "skip":
                logger.info("  SKIP: %s", result.get("reason", ""))
                total["skipped"] += 1
            else:
                logger.error("  ERROR: %s", result.get("error", ""))
                total["errors"] += 1

    logger.info("")
    logger.info("=" * 50)
    logger.info("SUMMARY: found=%d, updated=%d, skipped=%d, no_original=%d, errors=%d",
                total["found"], total["updated"], total["skipped"], total["no_original"], total["errors"])


if __name__ == "__main__":
    asyncio.run(main())
