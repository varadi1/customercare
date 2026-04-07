#!/usr/bin/env python3
"""Bulk ingest sent emails — fetches by weekly date ranges to avoid Graph API limits.

Usage (inside container):
    python3 /app/scripts/bulk_ingest_sent.py --mailbox lakossagitarolo@neuzrt.hu --from 2026-02-02 --to 2026-04-07
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from datetime import datetime, timedelta

import httpx

sys.path.insert(0, "/app")

GRAPH_BASE = "https://graph.microsoft.com/v1.0"


async def fetch_sent_range(mailbox: str, since: str, until: str, max_items: int = 500) -> list[dict]:
    """Fetch sent items within a specific date range."""
    from app.email.auth import get_auth_headers
    headers = get_auth_headers()

    url = f"{GRAPH_BASE}/users/{mailbox}/mailFolders/sentItems/messages"
    params = {
        "$filter": f"sentDateTime ge {since} and sentDateTime lt {until}",
        "$orderby": "sentDateTime desc",
        "$top": min(max_items, 50),
        "$select": "id,subject,from,toRecipients,body,sentDateTime",
    }

    all_messages = []
    async with httpx.AsyncClient(timeout=60) as client:
        while url and len(all_messages) < max_items:
            resp = await client.get(url, headers=headers, params=params)
            if resp.status_code != 200:
                print(f"  Error: {resp.status_code}")
                break
            data = resp.json()
            all_messages.extend(data.get("value", []))
            url = data.get("@odata.nextLink")
            params = {}
            time.sleep(0.1)

    return all_messages[:max_items]


async def ingest_batch(mailbox: str, messages: list[dict]) -> dict:
    """Ingest a batch of sent messages (depersonalized)."""
    from app.email.history import _html_to_text, _is_internal, _is_auto_reply, _strip_subject_prefix
    from app.rag.depersonalize import depersonalize
    from app.rag.ingest import ingest_text_async
    import hashlib, re

    stats = {"ingested": 0, "chunks": 0, "skipped": 0}
    seen_hashes = set()

    for msg in messages:
        subject = msg.get("subject", "")
        to_addrs = [r.get("emailAddress", {}).get("address", "")
                    for r in msg.get("toRecipients", [])]
        body_html = msg.get("body", {}).get("content", "")
        body_text = _html_to_text(body_html).strip()
        sent_date = msg.get("sentDateTime", "")[:10]

        # Filters
        if to_addrs and all(_is_internal(a) for a in to_addrs):
            stats["skipped"] += 1
            continue
        if _is_auto_reply(subject, body_text):
            stats["skipped"] += 1
            continue
        if len(body_text) < 50:
            stats["skipped"] += 1
            continue

        # Content dedup
        content_hash = hashlib.sha256(body_text[:500].encode()).hexdigest()[:16]
        if content_hash in seen_hashes:
            stats["skipped"] += 1
            continue
        seen_hashes.add(content_hash)

        # Depersonalize
        clean_body = depersonalize(body_text)
        clean_subject = re.sub(r"OETP-\d{4}-\d{4,8}", "[pályázat]", subject)
        full_text = f"Tárgy: {clean_subject}\nDátum: {sent_date}\n\n{clean_body}"

        msg_id_hash = hashlib.sha256(msg.get("id", "").encode()).hexdigest()[:16]
        source = f"email_reply:{mailbox}:{msg_id_hash}"

        try:
            chunks = await ingest_text_async(
                text=full_text,
                source=source,
                category="ügyfélszolgálat",
                chunk_type="email_reply",
                valid_from=sent_date or None,
            )
            stats["ingested"] += 1
            stats["chunks"] += chunks
        except Exception as e:
            print(f"  Error ingesting: {e}")

    return stats


async def main(mailbox: str, from_date: str, to_date: str):
    from_dt = datetime.fromisoformat(from_date)
    to_dt = datetime.fromisoformat(to_date)

    total_stats = {"fetched": 0, "ingested": 0, "chunks": 0, "skipped": 0, "weeks": 0}

    # Process week by week
    current = from_dt
    while current < to_dt:
        week_end = min(current + timedelta(days=7), to_dt)
        since = current.strftime("%Y-%m-%dT00:00:00Z")
        until = week_end.strftime("%Y-%m-%dT00:00:00Z")

        print(f"[{current.strftime('%Y-%m-%d')} → {week_end.strftime('%Y-%m-%d')}]", end=" ", flush=True)

        messages = await fetch_sent_range(mailbox, since, until, max_items=500)
        total_stats["fetched"] += len(messages)

        if messages:
            batch_stats = await ingest_batch(mailbox, messages)
            total_stats["ingested"] += batch_stats["ingested"]
            total_stats["chunks"] += batch_stats["chunks"]
            total_stats["skipped"] += batch_stats["skipped"]
            print(f"{len(messages)} fetched → {batch_stats['ingested']} ingested, {batch_stats['chunks']} chunks")
        else:
            print("0 emails")

        total_stats["weeks"] += 1
        current = week_end
        await asyncio.sleep(2)  # Rate limit

    print(f"\n{'='*60}")
    print(f"  BULK INGEST ÖSSZESÍTÉS")
    print(f"{'='*60}")
    print(f"  Hetek:      {total_stats['weeks']}")
    print(f"  Letöltve:   {total_stats['fetched']}")
    print(f"  Ingestálva: {total_stats['ingested']}")
    print(f"  Chunks:     {total_stats['chunks']}")
    print(f"  Kihagyva:   {total_stats['skipped']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mailbox", default="lakossagitarolo@neuzrt.hu")
    parser.add_argument("--from", dest="from_date", default="2026-02-02")
    parser.add_argument("--to", dest="to_date", default="2026-04-07")
    args = parser.parse_args()
    asyncio.run(main(args.mailbox, args.from_date, args.to_date))
