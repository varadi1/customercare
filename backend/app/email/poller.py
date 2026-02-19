"""Email polling via Microsoft Graph API."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from ..config import settings
from ..models import EmailMessage, PollResult
from .auth import get_auth_headers

GRAPH_BASE = "https://graph.microsoft.com/v1.0"
STATE_PATH = Path("/app/data/poll_state.json")


def _load_state() -> dict:
    """Load last poll timestamps per mailbox."""
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def _save_state(state: dict):
    """Save poll state."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def _html_to_text(html: str) -> str:
    """Convert HTML email body to plain text."""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n", strip=True)


async def poll_mailbox(
    mailbox: str,
    since: str | None = None,
    limit: int = 50,
) -> PollResult:
    """Poll a shared mailbox for new emails.

    Args:
        mailbox: email address of shared mailbox
        since: ISO datetime string — fetch emails received after this
        limit: max emails to fetch per request
    """
    headers = get_auth_headers()

    # Default: last 15 minutes
    if not since:
        state = _load_state()
        since = state.get(mailbox)
    if not since:
        since = (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat()

    # Graph API: list messages from Inbox only (not SentItems/Drafts)
    url = f"{GRAPH_BASE}/users/{mailbox}/mailFolders/Inbox/messages"
    params = {
        "$filter": f"receivedDateTime ge {since}",
        "$orderby": "receivedDateTime desc",
        "$top": limit,
        "$select": "id,subject,from,body,receivedDateTime,conversationId,importance,hasAttachments,internetMessageHeaders",
    }

    messages = []
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, headers=headers, params=params)

        if resp.status_code == 200:
            data = resp.json()
            for msg in data.get("value", []):
                sender = msg.get("from", {}).get("emailAddress", {})

                # Extract In-Reply-To from internet headers
                in_reply_to = None
                for header in msg.get("internetMessageHeaders", []):
                    if header.get("name", "").lower() == "in-reply-to":
                        in_reply_to = header.get("value")
                        break

                body_html = msg.get("body", {}).get("content", "")
                body_text = _html_to_text(body_html) if body_html else ""

                messages.append(EmailMessage(
                    id=msg["id"],
                    subject=msg.get("subject") or "(Nincs tárgy)",
                    sender=sender.get("name", ""),
                    sender_email=sender.get("address", ""),
                    body_text=body_text,
                    body_html=body_html,
                    received_at=msg.get("receivedDateTime", ""),
                    conversation_id=msg.get("conversationId"),
                    in_reply_to=in_reply_to,
                    mailbox=mailbox,
                    has_attachments=msg.get("hasAttachments", False),
                    importance=msg.get("importance", "normal"),
                ))
        else:
            print(f"[poller] Error fetching {mailbox}: {resp.status_code} {resp.text}")

    # Update state
    if messages:
        state = _load_state()
        state[mailbox] = datetime.now(timezone.utc).isoformat()
        _save_state(state)

    return PollResult(
        new_emails=len(messages),
        mailbox=mailbox,
        messages=messages,
    )


async def poll_all_mailboxes(hours: float | None = None) -> list[PollResult]:
    """Poll all configured shared mailboxes.

    Args:
        hours: if set, override the saved state and fetch emails from the last N hours.
               This ensures overlap between cron runs so no emails slip through.
    """
    mailboxes = [m.strip() for m in settings.shared_mailboxes.split(",") if m.strip()]
    since_override = None
    if hours:
        since_override = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()

    results = []
    for mb in mailboxes:
        result = await poll_mailbox(mb, since=since_override)
        results.append(result)
    return results


async def get_email_thread(
    mailbox: str,
    conversation_id: str,
    limit: int = 20,
) -> list[EmailMessage]:
    """Fetch full email thread by conversation ID."""
    headers = get_auth_headers()

    url = f"{GRAPH_BASE}/users/{mailbox}/messages"
    params = {
        "$filter": f"conversationId eq '{conversation_id}'",
        "$orderby": "receivedDateTime asc",
        "$top": limit,
        "$select": "id,subject,from,body,receivedDateTime,conversationId,importance,hasAttachments",
    }

    messages = []
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, headers=headers, params=params)

        if resp.status_code == 200:
            data = resp.json()
            for msg in data.get("value", []):
                sender = msg.get("from", {}).get("emailAddress", {})
                body_html = msg.get("body", {}).get("content", "")
                body_text = _html_to_text(body_html)

                messages.append(EmailMessage(
                    id=msg["id"],
                    subject=msg.get("subject") or "(Nincs tárgy)",
                    sender=sender.get("name", ""),
                    sender_email=sender.get("address", ""),
                    body_text=body_text,
                    body_html=body_html,
                    received_at=msg.get("receivedDateTime", ""),
                    conversation_id=msg.get("conversationId"),
                    mailbox=mailbox,
                    has_attachments=msg.get("hasAttachments", False),
                    importance=msg.get("importance", "normal"),
                ))

    return messages
