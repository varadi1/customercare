#!/usr/bin/env python3
"""Generic scraper CLI — config-driven from program.yaml.

Usage (inside cc-backend container):
    python3 /app/scripts/scrape.py                     # Scrape all, skip unchanged
    python3 /app/scripts/scrape.py --force              # Re-scrape all regardless of hash
    python3 /app/scripts/scrape.py --dry-run            # Fetch+parse only, no ingest
    python3 /app/scripts/scrape.py --page 0             # Scrape specific page by index
    python3 /app/scripts/scrape.py --no-download        # Skip file downloads
    python3 /app/scripts/scrape.py --list               # List configured pages
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

sys.path.insert(0, "/app")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("scrape")


async def main():
    parser = argparse.ArgumentParser(description="CustomerCare generic scraper")
    parser.add_argument("--force", action="store_true", help="Ignore hash, re-scrape all")
    parser.add_argument("--dry-run", action="store_true", help="Fetch+parse only, no ingest or download")
    parser.add_argument("--page", type=int, default=None, help="Scrape specific page index only")
    parser.add_argument("--no-download", action="store_true", help="Skip file downloads")
    parser.add_argument("--list", action="store_true", help="List configured pages and exit")
    args = parser.parse_args()

    from app.config import get_program_config
    pcfg = get_program_config()
    scraper_cfg = pcfg.get("scraper", {})

    if not scraper_cfg.get("enabled", False):
        print("Scraper is disabled in program.yaml")
        return

    pages = scraper_cfg.get("pages", [])
    if not pages:
        print("No pages configured in program.yaml scraper.pages")
        return

    if args.list:
        print(f"Configured pages ({len(pages)}):")
        for i, p in enumerate(pages):
            print(f"  [{i}] {p.get('label', p['url'])} → doc_type={p.get('doc_type', '?')}")
            print(f"      {p['url']}")
        return

    from app.rag.scraper import Scraper

    scraper = Scraper(scraper_cfg)

    print(f"{'='*60}")
    print(f"  CustomerCare Scraper {'(DRY RUN)' if args.dry_run else ''}")
    print(f"  Pages: {len(pages) if args.page is None else 1}")
    print(f"  Method: {scraper_cfg.get('method', 'auto')}")
    print(f"{'='*60}\n")

    # Scrape
    results = await scraper.scrape_all(force=args.force, page_index=args.page)

    # Summary
    changed = [r for r in results if r.changed and not r.error]
    errors = [r for r in results if r.error]
    unchanged = [r for r in results if not r.changed]

    for r in results:
        status = "ERROR" if r.error else ("CHANGED" if r.changed else "unchanged")
        print(f"  [{status}] {r.label}")
        if r.error:
            print(f"         Error: {r.error}")
        else:
            print(f"         Method: {r.method}, hash: {r.content_hash}")
            print(f"         Announcements: {len(r.announcements)}, Downloads: {len(r.download_links)}")

    if not changed:
        print(f"\nNo changes detected. {len(unchanged)} pages unchanged, {len(errors)} errors.")
        return

    # Download files
    downloaded = []
    if not args.dry_run and not args.no_download:
        print(f"\nDownloading files...")
        downloaded = await scraper.download_all(results)
        print(f"  Downloaded: {len(downloaded)} files")

    # Ingest
    if not args.dry_run:
        print(f"\nIngesting content...")
        total_chunks = await scraper.ingest_results(results, downloaded)
        print(f"  Total chunks ingested: {total_chunks}")

    print(f"\n{'='*60}")
    print(f"  SUMMARY {'(DRY RUN)' if args.dry_run else ''}")
    print(f"{'='*60}")
    print(f"  Pages scraped: {len(results)}")
    print(f"  Changed: {len(changed)}")
    print(f"  Unchanged: {len(unchanged)}")
    print(f"  Errors: {len(errors)}")
    if not args.dry_run:
        print(f"  Files downloaded: {len(downloaded)}")


if __name__ == "__main__":
    asyncio.run(main())
