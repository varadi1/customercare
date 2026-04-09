#!/usr/bin/env python3
"""Monitor nffku.hu OETP page for changes.

Scrapes the OETP page, compares with last snapshot, and if changed:
1. Saves new snapshot
2. Shows diff
3. Optionally triggers re-ingest into Hanna

Usage:
    python3 monitor_nffku.py              # Check for changes
    python3 monitor_nffku.py --ingest     # Check + auto-ingest if changed
    python3 monitor_nffku.py --force      # Force re-ingest even if no change
"""

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

URL = "https://nffku.hu/index.php/tile-detail/otthonienergiatarolo"
SNAPSHOT_DIR = Path(os.path.expanduser("~/.openclaw/hanna/data/nffku_snapshots"))
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
LATEST_FILE = SNAPSHOT_DIR / "latest.json"
HANNA_API = "http://localhost:8101"


def scrape_page() -> dict:
    """Scrape the OETP page and extract structured content."""
    resp = requests.get(URL, timeout=30, headers={
        "User-Agent": "Mozilla/5.0 (OETP-monitor; NEU internal)"
    })
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Extract main content area
    # Remove nav, footer, scripts
    for tag in soup.find_all(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    body = soup.find("body")
    text = body.get_text(separator="\n", strip=True) if body else ""

    # Extract links to PDFs and documents
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        label = a.get_text(strip=True)
        if any(ext in href.lower() for ext in [".pdf", ".doc", ".xlsx"]):
            if not href.startswith("http"):
                href = "https://nffku.hu" + href
            links.append({"url": href, "label": label})

    # Extract közlemények (announcements) by date pattern
    announcements = []
    date_pattern = re.compile(r"(\d{4}\.\d{2}\.\d{2}\.?)\s*[-–]?\s*(.*?)(?=\d{4}\.\d{2}\.\d{2}\.?|\Z)", re.DOTALL)
    for m in date_pattern.finditer(text):
        date_str = m.group(1).strip().rstrip(".")
        title = m.group(2).strip()[:200]
        if title and len(title) > 10:
            announcements.append({"date": date_str, "title": title.split("\n")[0]})

    content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

    return {
        "url": URL,
        "scraped_at": datetime.now().isoformat(timespec="seconds"),
        "content_hash": content_hash,
        "text_length": len(text),
        "text": text,
        "links": links,
        "announcements": announcements[:20],
    }


def load_latest() :
    """Load the last saved snapshot."""
    if LATEST_FILE.exists():
        with open(LATEST_FILE) as f:
            return json.load(f)
    return None


def save_snapshot(data: dict):
    """Save snapshot with timestamp and as latest."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped = SNAPSHOT_DIR / f"snapshot_{ts}.json"

    with open(timestamped, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    with open(LATEST_FILE, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Snapshot saved: {timestamped}")


def diff_snapshots(old: dict, new: dict) -> dict:
    """Compare two snapshots and return differences."""
    changes = {
        "hash_changed": old["content_hash"] != new["content_hash"],
        "text_length_delta": new["text_length"] - old["text_length"],
        "new_announcements": [],
        "new_links": [],
    }

    old_dates = {a["date"] for a in old.get("announcements", [])}
    for a in new.get("announcements", []):
        if a["date"] not in old_dates:
            changes["new_announcements"].append(a)

    old_urls = {l["url"] for l in old.get("links", [])}
    for l in new.get("links", []):
        if l["url"] not in old_urls:
            changes["new_links"].append(l)

    return changes


def trigger_ingest():
    """Trigger Hanna scraper+ingest via Docker container."""
    import subprocess

    try:
        # Check if hanna-backend container is running
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", "hanna-backend"],
            capture_output=True, text=True, timeout=10,
        )
        if result.stdout.strip() != "true":
            print("WARNING: hanna-backend container is not running, skipping ingest")
            return False

        # Run the scraper inside the container (downloads + ingests)
        print("Running scraper+ingest in hanna-backend container...")
        result = subprocess.run(
            ["docker", "exec", "hanna-backend", "python3", "/app/scripts/scrape_nffku_oetp.py"],
            capture_output=True, text=True, timeout=600,
        )
        print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}")

        if result.returncode != 0:
            print(f"WARNING: Scraper exited with code {result.returncode}")
            return False

        print("Ingest completed successfully.")
        return True

    except Exception as e:
        print(f"WARNING: Ingest trigger failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Monitor nffku.hu OETP page")
    parser.add_argument("--ingest", action="store_true", help="Auto-ingest if changed")
    parser.add_argument("--force", action="store_true", help="Force re-ingest")
    args = parser.parse_args()

    print(f"=== nffku.hu OETP Monitor ({datetime.now().strftime('%Y-%m-%d %H:%M')}) ===\n")

    # Scrape
    print(f"Scraping {URL}...")
    try:
        current = scrape_page()
    except Exception as e:
        print(f"ERROR: Scrape failed: {e}")
        sys.exit(1)

    print(f"  Content: {current['text_length']} chars, hash: {current['content_hash']}")
    print(f"  Announcements: {len(current['announcements'])}")
    print(f"  Document links: {len(current['links'])}")

    # Compare with last snapshot
    previous = load_latest()

    if previous is None:
        print("\n  No previous snapshot — saving initial baseline.")
        save_snapshot(current)
        if args.ingest or args.force:
            trigger_ingest()
        return

    diff = diff_snapshots(previous, current)

    if not diff["hash_changed"] and not args.force:
        print(f"\n  NO CHANGES since {previous['scraped_at']}")
        return

    # Changes detected!
    print(f"\n  CHANGES DETECTED!")
    print(f"  Previous: {previous['scraped_at']}, hash: {previous['content_hash']}")
    print(f"  Text delta: {diff['text_length_delta']:+d} chars")

    if diff["new_announcements"]:
        print(f"\n  NEW ANNOUNCEMENTS ({len(diff['new_announcements'])}):")
        for a in diff["new_announcements"]:
            print(f"    {a['date']}: {a['title'][:80]}")

    if diff["new_links"]:
        print(f"\n  NEW DOCUMENT LINKS ({len(diff['new_links'])}):")
        for l in diff["new_links"]:
            print(f"    {l['label'][:60]} → {l['url'][:80]}")

    # Save new snapshot
    save_snapshot(current)

    # Ingest if requested
    if args.ingest or args.force:
        print("\nTriggering ingest...")
        trigger_ingest()


if __name__ == "__main__":
    main()
