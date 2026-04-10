#!/usr/bin/env python3
"""Monitor configured scraper pages for changes.

Lightweight check (HTTP only, no ingest) — triggers full scrape on change.
Run via LaunchAgent daily or hourly.

Usage:
    python3 monitor_scraper.py              # Check all pages for changes
    python3 monitor_scraper.py --ingest     # Check + auto-ingest if changed
    python3 monitor_scraper.py --force      # Force re-ingest even if no change
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

# Load program.yaml
import yaml

CONFIG_PATHS = [
    Path(__file__).parent.parent / "config" / "program.yaml",
    Path.home() / "DEV" / "customercare" / "config" / "program.yaml",
]


def _load_config() -> dict:
    for p in CONFIG_PATHS:
        if p.exists():
            with open(p) as f:
                return yaml.safe_load(f) or {}
    print("ERROR: program.yaml not found")
    sys.exit(1)


def _scrape_page(url: str) -> dict:
    """Lightweight scrape — just text + hash, no full extraction."""
    resp = httpx.get(url, timeout=30, follow_redirects=True, headers={
        "User-Agent": "Mozilla/5.0 (CustomerCare monitor)"
    })
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup.find_all(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

    # Extract download links
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if any(ext in href.lower() for ext in [".pdf", ".doc", ".xlsx"]):
            if not href.startswith("http"):
                base = url.rsplit("/", 1)[0] + "/"
                href = base + href
            links.append({"url": href, "label": a.get_text(strip=True)})

    # Extract announcements by date pattern
    announcements = []
    date_pattern = re.compile(r"(\d{4}\.\d{2}\.\d{2}\.?)")
    for m in date_pattern.finditer(text):
        date_str = m.group(1).strip().rstrip(".")
        # Get surrounding context
        start = max(0, m.start() - 10)
        end = min(len(text), m.end() + 200)
        title = text[m.start():end].split("\n")[0][:100]
        if len(title) > 10:
            announcements.append({"date": date_str, "title": title})

    return {
        "url": url,
        "scraped_at": datetime.now().isoformat(timespec="seconds"),
        "content_hash": content_hash,
        "text_length": len(text),
        "links": links[:20],
        "announcements": announcements[:20],
    }


def _url_slug(url: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "_", url)[:80]


def _load_latest(url: str, snapshot_dir: Path) -> dict | None:
    latest = snapshot_dir / f"{_url_slug(url)}_latest.json"
    if latest.exists():
        return json.loads(latest.read_text())
    return None


def _save_snapshot(url: str, data: dict, snapshot_dir: Path):
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    slug = _url_slug(url)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    (snapshot_dir / f"{slug}_{ts}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))
    (snapshot_dir / f"{slug}_latest.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))


def _trigger_scrape(force: bool = False):
    """Trigger full scrape inside the Docker container."""
    cmd = ["docker", "exec", "cc-backend", "python3", "/app/scripts/scrape.py"]
    if force:
        cmd.append("--force")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"Scrape trigger failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Monitor scraper pages for changes")
    parser.add_argument("--ingest", action="store_true", help="Auto-scrape+ingest if changed")
    parser.add_argument("--force", action="store_true", help="Force re-scrape")
    args = parser.parse_args()

    config = _load_config()
    scraper_cfg = config.get("scraper", {})
    pages = scraper_cfg.get("pages", [])
    snapshot_dir = Path(os.path.expanduser(
        scraper_cfg.get("snapshot_dir", "~/.openclaw/customercare/data/scraper_snapshots")
    ))

    if not pages:
        print("No pages configured")
        return

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"=== CustomerCare Scraper Monitor ({now}) ===\n")

    any_changed = False

    for page_cfg in pages:
        url = page_cfg["url"]
        label = page_cfg.get("label", url)
        print(f"Checking: {label}")
        print(f"  URL: {url}")

        try:
            current = _scrape_page(url)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        print(f"  Content: {current['text_length']} chars, hash: {current['content_hash']}")
        print(f"  Announcements: {len(current['announcements'])}, Links: {len(current['links'])}")

        previous = _load_latest(url, snapshot_dir)

        if previous is None:
            print(f"  First run — saving baseline")
            _save_snapshot(url, current, snapshot_dir)
            any_changed = True
            continue

        if current["content_hash"] == previous.get("content_hash") and not args.force:
            print(f"  No changes since {previous.get('scraped_at', '?')}")
            continue

        # Changes detected
        print(f"  CHANGED! (was: {previous.get('content_hash', '?')} @ {previous.get('scraped_at', '?')})")
        delta = current["text_length"] - previous.get("text_length", 0)
        print(f"  Text delta: {delta:+d} chars")

        # Check for new announcements
        old_dates = {a["date"] for a in previous.get("announcements", [])}
        new_anns = [a for a in current["announcements"] if a["date"] not in old_dates]
        if new_anns:
            print(f"  New announcements ({len(new_anns)}):")
            for a in new_anns[:5]:
                print(f"    {a['date']}: {a['title'][:60]}")

        _save_snapshot(url, current, snapshot_dir)
        any_changed = True

    if any_changed and (args.ingest or args.force):
        print("\nTriggering full scrape+ingest...")
        _trigger_scrape(force=args.force)


if __name__ == "__main__":
    main()
