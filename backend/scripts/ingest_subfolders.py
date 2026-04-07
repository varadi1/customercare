#!/usr/bin/env python3
"""Ingest answered emails from inbox subfolders into RAG.

Strategy: Subfolder emails are categorized customer questions.
The folder name provides the topic/category. We ingest each email
with its topic metadata. Combined with existing sentItems Q&A pairs,
this gives comprehensive RAG coverage.

Run inside the hanna-backend container:
    python3 /app/scripts/ingest_subfolders.py [--dry-run] [--mailbox X] [--limit N]
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import defaultdict

import httpx
from bs4 import BeautifulSoup

sys.path.insert(0, "/app")

from app.email.auth import get_auth_headers
from app.rag.ingest import ingest_text

GRAPH_BASE = "https://graph.microsoft.com/v1.0"

_RE_PREFIX = re.compile(r"^(RE|FW|Fwd|Vá|VS|AW|SV)\s*:\s*", re.IGNORECASE)

# Folders to skip
SKIP_FOLDERS = {
    "kézbesíthetetlen válaszok",
    "MELLÉKLETEK??",
}


def html_to_text(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n", strip=True)


def strip_subject_prefix(subject: str) -> str:
    s = (subject or "").strip()
    while True:
        new_s = _RE_PREFIX.sub("", s).strip()
        if new_s == s:
            break
        s = new_s
    return s


def is_internal(email: str) -> bool:
    email = (email or "").lower()
    return any(email.endswith(d) for d in [
        "@neuzrt.hu", "@nffku.hu", "@nffku.onmicrosoft.com", "@norvegalap.hu"
    ])


def is_auto_reply(subject: str) -> bool:
    subj_lower = (subject or "").lower()
    auto_keywords = [
        "automatic reply", "automatikus válasz", "out of office",
        "delivery status", "undeliverable", "mailer-daemon",
        "nem kézbesíthető", "értesítés oetp",
    ]
    return any(kw in subj_lower for kw in auto_keywords)


def fetch_subfolders(client: httpx.Client, headers: dict, mailbox: str) -> list[dict]:
    """Get all inbox subfolders."""
    r = client.get(f"{GRAPH_BASE}/users/{mailbox}/mailFolders/inbox", headers=headers)
    if r.status_code != 200:
        print(f"[ERROR] Can't get inbox for {mailbox}: {r.status_code}")
        return []

    inbox_id = r.json()["id"]
    r2 = client.get(
        f"{GRAPH_BASE}/users/{mailbox}/mailFolders/{inbox_id}/childFolders",
        headers=headers,
        params={"$select": "id,displayName,totalItemCount", "$top": 100},
    )
    if r2.status_code != 200:
        print(f"[ERROR] Can't list subfolders: {r2.status_code}")
        return []

    folders = r2.json().get("value", [])
    return [f for f in folders
            if f["totalItemCount"] > 0
            and f["displayName"].lower() not in {s.lower() for s in SKIP_FOLDERS}]


def fetch_folder_messages(
    client: httpx.Client, headers: dict, mailbox: str,
    folder_id: str, limit: int = 500
) -> list[dict]:
    """Fetch messages from a folder with pagination."""
    url = f"{GRAPH_BASE}/users/{mailbox}/mailFolders/{folder_id}/messages"
    params = {
        "$select": "id,subject,from,body,receivedDateTime,conversationId",
        "$orderby": "receivedDateTime desc",
        "$top": min(limit, 50),
    }

    all_msgs = []
    while url and len(all_msgs) < limit:
        r = client.get(url, headers=headers, params=params)
        if r.status_code != 200:
            print(f"  [WARN] Fetch error: {r.status_code}")
            break

        data = r.json()
        all_msgs.extend(data.get("value", []))
        url = data.get("@odata.nextLink")
        params = {}

        time.sleep(0.1)

    return all_msgs[:limit]


def get_existing_sources() -> set[str]:
    """Get all existing source prefixes to avoid duplicates (PostgreSQL)."""
    import asyncio
    import asyncpg

    async def _fetch():
        conn = await asyncpg.connect(
            os.environ.get("HANNA_PG_DSN", "postgresql://klara:klara_docs_2026@hanna-db:5432/hanna_oetp")
        )
        rows = await conn.fetch(
            "SELECT DISTINCT doc_id FROM chunks WHERE doc_type IN ('email_reply', 'email_question')"
        )
        await conn.close()
        return {r["doc_id"] for r in rows}

    return asyncio.run(_fetch())


def main():
    parser = argparse.ArgumentParser(description="Ingest inbox subfolder emails into RAG")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually ingest")
    parser.add_argument("--mailbox", type=str, default=None,
                        help="Process only this mailbox")
    parser.add_argument("--limit", type=int, default=200,
                        help="Max messages per folder")
    parser.add_argument("--folder", type=str, default=None,
                        help="Process only this folder name (substring match)")
    args = parser.parse_args()

    headers = get_auth_headers()
    mailboxes = [args.mailbox] if args.mailbox else [
        "info@neuzrt.hu",
        "lakossagitarolo@neuzrt.hu",
    ]

    # Get existing sources to skip duplicates
    if not args.dry_run:
        print("[init] Loading existing sources for dedup...")
        existing_sources = get_existing_sources()
        print(f"[init] {len(existing_sources)} existing email chunks in RAG")
    else:
        existing_sources = set()

    stats = defaultdict(int)

    with httpx.Client(timeout=60) as client:
        for mailbox in mailboxes:
            print(f"\n{'='*60}")
            print(f"  MAILBOX: {mailbox}")
            print(f"{'='*60}")

            subfolders = fetch_subfolders(client, headers, mailbox)
            print(f"  Found {len(subfolders)} non-empty subfolders")

            for folder in subfolders:
                fname = folder["displayName"]
                fcount = folder["totalItemCount"]

                if args.folder and args.folder.lower() not in fname.lower():
                    continue

                print(f"\n  📁 {fname} ({fcount} items)")

                messages = fetch_folder_messages(
                    client, headers, mailbox,
                    folder["id"], limit=args.limit
                )
                print(f"     Fetched {len(messages)} messages")

                f_ingested = 0
                f_skipped = 0
                f_dupes = 0
                seen_subjects = set()

                for msg in messages:
                    subject = msg.get("subject") or "(nincs tárgy)"
                    sender_addr = msg.get("from", {}).get("emailAddress", {}).get("address", "")
                    sender_name = msg.get("from", {}).get("emailAddress", {}).get("name", "")
                    body_html = msg.get("body", {}).get("content", "")
                    body_text = html_to_text(body_html).strip()
                    recv_date = (msg.get("receivedDateTime") or "")[:10]
                    import hashlib as _hl
                    msg_id_hash = _hl.sha256((msg.get("id") or "x").encode()).hexdigest()[:12]
                    source = f"subfolder:{mailbox}:{fname}:{msg_id_hash}"

                    # Skip auto-replies
                    if is_auto_reply(subject):
                        f_skipped += 1
                        continue

                    # Skip internal
                    if is_internal(sender_addr):
                        f_skipped += 1
                        continue

                    # Skip very short
                    if len(body_text) < 40:
                        f_skipped += 1
                        continue

                    # Skip duplicate subjects in same folder
                    clean_subj = strip_subject_prefix(subject).lower()[:80]
                    if clean_subj in seen_subjects:
                        f_dupes += 1
                        continue
                    seen_subjects.add(clean_subj)

                    # Format for ingestion
                    full_text = (
                        f"Téma: {fname}\n"
                        f"Tárgy: {subject}\n"
                        f"Feladó: {sender_name} ({sender_addr})\n"
                        f"Mailbox: {mailbox}\n"
                        f"Dátum: {recv_date}\n\n"
                        f"{body_text}"
                    )

                    if args.dry_run:
                        if f_ingested < 3:
                            print(f"     📧 {subject[:60]} — {sender_name} ({len(body_text)} chars)")
                        f_ingested += 1
                    else:
                        try:
                            chunks = ingest_text(
                                text=full_text,
                                source=source,
                                category=fname,
                                chunk_type="email_question",
                                valid_from=recv_date or None,
                            )
                            f_ingested += 1
                            stats["chunks"] += chunks
                        except Exception as e:
                            print(f"     [ERROR] {e}")
                            stats["errors"] += 1

                print(f"     → Ingested: {f_ingested} | Skipped: {f_skipped} | Dupes: {f_dupes}")
                stats["ingested"] += f_ingested
                stats["skipped"] += f_skipped
                stats["dupes"] += f_dupes

    print(f"\n{'='*60}")
    print(f"  ÖSSZESÍTÉS {'(DRY RUN)' if args.dry_run else ''}")
    print(f"{'='*60}")
    print(f"  Feldolgozva: {stats['ingested']}")
    print(f"  Kihagyva:    {stats['skipped']}")
    print(f"  Duplikált:   {stats['dupes']}")
    if not args.dry_run:
        print(f"  Chunk-ok:    {stats['chunks']}")
        print(f"  Hibák:       {stats['errors']}")
    print()


if __name__ == "__main__":
    main()
