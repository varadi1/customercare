"""Simple JSON-based draft store for feedback tracking."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

STORE_PATH = Path("/app/data/draft_store.json")


def _load() -> list[dict]:
    if STORE_PATH.exists():
        try:
            return json.loads(STORE_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return []
    return []


def _save(data: list[dict]) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STORE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def save_draft(
    conversation_id: str,
    message_id: str,
    mailbox: str,
    draft_html: str,
    confidence: str,
    draft_id: str = "",
    subject: str = "",
) -> None:
    """Save a draft to the store."""
    data = _load()
    data.append({
        "conversation_id": conversation_id,
        "message_id": message_id,
        "mailbox": mailbox,
        "draft_id": draft_id,
        "subject": subject,
        "draft_html": draft_html,
        "confidence": confidence,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    # Keep last 500 entries
    if len(data) > 500:
        data = data[-500:]
    _save(data)


def get_drafts_by_conversation(conversation_id: str) -> list[dict]:
    """Get all drafts for a conversation."""
    return [d for d in _load() if d.get("conversation_id") == conversation_id]


def get_recent_drafts(hours: int = 48) -> list[dict]:
    """Get drafts from the last N hours."""
    from datetime import timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    results = []
    for d in _load():
        try:
            created = datetime.fromisoformat(d["created_at"])
            if created >= cutoff:
                results.append(d)
        except (KeyError, ValueError):
            continue
    return results
