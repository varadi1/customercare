"""Email polling via Microsoft Graph API."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from ..config import settings
from ..models import EmailMessage, PollResult
from .auth import get_auth_headers

GRAPH_BASE = "https://graph.microsoft.com/v1.0"
STATE_PATH = Path("/app/data/poll_state.json")

# Regex patterns for auto-extracting identifiers
_OETP_RE = re.compile(r"OETP-\d{4}-\d{4,8}", re.IGNORECASE)
_POD_RE = re.compile(r"HU-[A-Z]{2,10}-\d[\w-]{5,30}", re.IGNORECASE)


def _extract_identifiers(text: str) -> tuple[list[str], list[str]]:
    """Extract OETP IDs and POD numbers from email text."""
    oetp_ids = list(set(_OETP_RE.findall(text)))
    pod_numbers = list(set(_POD_RE.findall(text)))
    return oetp_ids, pod_numbers


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
        "$select": "id,subject,from,body,receivedDateTime,conversationId,importance,hasAttachments,internetMessageHeaders,categories",
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

                # Auto-extract identifiers from subject + body
                combined_text = f"{msg.get('subject', '')} {body_text}"
                oetp_ids, pod_numbers = _extract_identifiers(combined_text)

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
                    oetp_ids=oetp_ids,
                    pod_numbers=pod_numbers,
                    categories=msg.get("categories", []),
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


def _parse_messages(raw_messages: list[dict], mailbox: str) -> list[EmailMessage]:
    """Parse Graph API message dicts into EmailMessage objects."""
    messages = []
    for msg in raw_messages:
        sender = msg.get("from", {}).get("emailAddress", {})
        body_html = msg.get("body", {}).get("content", "")
        body_text = _html_to_text(body_html)

        combined_text = f"{msg.get('subject', '')} {body_text}"
        oetp_ids, pod_numbers = _extract_identifiers(combined_text)

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
            oetp_ids=oetp_ids,
            pod_numbers=pod_numbers,
        ))
    return messages


async def get_email_thread(
    mailbox: str,
    conversation_id: str,
    limit: int = 20,
    subject: str | None = None,
    sender_email: str | None = None,
) -> list[EmailMessage]:
    """Fetch full email thread by conversation ID, with subject+sender fallback.

    On shared mailboxes the conversationId filter often returns
    InefficientFilter (400) or empty results.  When that happens and
    subject/sender_email are provided we fall back to a subject-based
    search across Inbox + SentItems.
    """
    headers = get_auth_headers()
    select = "id,subject,from,body,receivedDateTime,conversationId,importance,hasAttachments"

    # --- Attempt 1: conversationId filter (all messages, not just Inbox) ---
    url = f"{GRAPH_BASE}/users/{mailbox}/messages"
    params = {
        "$filter": f"conversationId eq '{conversation_id}'",
        "$orderby": "receivedDateTime asc",
        "$top": limit,
        "$select": select,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, headers=headers, params=params)

        if resp.status_code == 200:
            data = resp.json().get("value", [])
            if data:
                return _parse_messages(data, mailbox)

        # Log the failure reason
        if resp.status_code != 200:
            print(f"[poller] conversationId filter failed ({resp.status_code}), trying fallback...")
        else:
            print("[poller] conversationId filter returned 0 results, trying fallback...")

        # --- Attempt 2: subject-based search (Inbox + SentItems) ---
        if not subject:
            print("[poller] No subject provided for fallback — returning empty")
            return []

        # Normalize subject: strip RE:/FW: prefixes for matching
        clean_subject = subject.strip()
        for prefix in ("RE:", "Re:", "re:", "FW:", "Fw:", "fw:", "VS:", "Vs:"):
            if clean_subject.startswith(prefix):
                clean_subject = clean_subject[len(prefix):].strip()

        # Escape single quotes for OData filter
        escaped_subject = clean_subject.replace("'", "''")

        all_messages: list[dict] = []

        for folder in ("Inbox", "SentItems", "Drafts"):
            folder_url = f"{GRAPH_BASE}/users/{mailbox}/mailFolders/{folder}/messages"
            folder_params = {
                "$filter": f"contains(subject, '{escaped_subject}')",
                "$orderby": "receivedDateTime asc",
                "$top": limit,
                "$select": select,
            }
            try:
                folder_resp = await client.get(folder_url, headers=headers, params=folder_params)
                if folder_resp.status_code == 200:
                    all_messages.extend(folder_resp.json().get("value", []))
                else:
                    # Try without $orderby
                    del folder_params["$orderby"]
                    folder_resp = await client.get(folder_url, headers=headers, params=folder_params)
                    if folder_resp.status_code == 200:
                        all_messages.extend(folder_resp.json().get("value", []))
            except Exception as e:
                print(f"[poller] Fallback folder {folder} error: {e}")

        if not all_messages:
            print(f"[poller] Subject fallback found 0 messages for '{clean_subject[:40]}'")
            return []

        # Deduplicate by message ID
        seen_ids: set[str] = set()
        unique: list[dict] = []
        for m in all_messages:
            if m["id"] not in seen_ids:
                seen_ids.add(m["id"])
                unique.append(m)

        # Sort by receivedDateTime
        unique.sort(key=lambda m: m.get("receivedDateTime", ""))

        # Optional: filter by sender_email if provided (keep only thread participants)
        if sender_email:
            # Keep messages that are FROM or TO this sender
            # (we don't filter here — we want the full thread)
            pass

        print(f"[poller] Subject fallback found {len(unique)} messages for '{clean_subject[:40]}'")
        return _parse_messages(unique, mailbox)
