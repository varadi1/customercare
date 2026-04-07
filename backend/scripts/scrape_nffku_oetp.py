#!/usr/bin/env python3
"""Scrape OETP közlemények + letölthető dokumentumok from nffku.hu.

Downloads PDFs and ingests them into hanna_oetp RAG.
Run inside hanna-backend container:
    python3 /app/scripts/scrape_nffku_oetp.py [--dry-run] [--download-dir /app/data/pdfs]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

sys.path.insert(0, "/app")

BASE_URL = "https://nffku.hu"
PAGE_URL = f"{BASE_URL}/index.php/tile-detail/otthonienergiatarolo"


def fetch_page(client: httpx.Client) -> BeautifulSoup:
    """Fetch the OETP page."""
    r = client.get(PAGE_URL, follow_redirects=True)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def extract_kozlemenyek(soup: BeautifulSoup) -> list[dict]:
    """Extract közlemények from accordion buttons + their content."""
    results = []

    # Find all accordion buttons (h3 inside button)
    buttons = soup.find_all("button")
    for btn in buttons:
        h3 = btn.find("h3")
        if not h3:
            continue

        title = h3.get_text(strip=True)
        if not title or len(title) < 5:
            continue

        # Extract date
        date_match = re.search(r"(\d{4})[\.\s]+(\d{2})[\.\s]+(\d{2})", title)
        date = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}" if date_match else None

        # Get the content panel (sibling div)
        panel_id = btn.get("aria-controls", "")
        panel = None
        if panel_id:
            panel = soup.find(id=panel_id)
        if not panel:
            # Try next sibling
            panel = btn.find_next_sibling("div")

        content = ""
        links = []
        if panel:
            content = panel.get_text(separator="\n", strip=True)
            # Find PDF/doc links in panel
            for a in panel.find_all("a", href=True):
                href = a["href"]
                if any(href.lower().endswith(ext) for ext in [".pdf", ".doc", ".docx", ".xls", ".xlsx"]):
                    link_text = a.get_text(strip=True) or os.path.basename(href)
                    full_url = href if href.startswith("http") else BASE_URL + href
                    links.append({"name": link_text, "url": full_url})

        results.append({
            "title": title,
            "date": date,
            "content": content,
            "links": links,
        })

    return results


def extract_downloads(soup: BeautifulSoup) -> list[dict]:
    """Extract letölthető dokumentumok from the downloads table."""
    downloads = []

    table = soup.find("table")
    if not table:
        return downloads

    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 2:
            continue

        # Get the document name and link
        link = cells[1].find("a", href=True)
        if not link:
            continue

        name = link.get_text(strip=True)
        href = link["href"]
        full_url = href if href.startswith("http") else BASE_URL + href

        downloads.append({"name": name, "url": full_url})

    return downloads


def download_file(client: httpx.Client, url: str, download_dir: Path) -> Path | None:
    """Download a file and return local path."""
    try:
        filename = os.path.basename(url.split("?")[0])
        # URL decode
        from urllib.parse import unquote
        filename = unquote(filename)

        local_path = download_dir / filename
        if local_path.exists():
            print(f"  [skip] Already exists: {filename}")
            return local_path

        r = client.get(url, follow_redirects=True)
        r.raise_for_status()

        local_path.write_bytes(r.content)
        print(f"  [ok] Downloaded: {filename} ({len(r.content)} bytes)")
        return local_path
    except Exception as e:
        print(f"  [error] Download failed {url}: {e}")
        return None


def ingest_pdf(path: Path, chunk_type: str, valid_from: str | None = None) -> int:
    """Ingest a PDF file into hanna_oetp."""
    from app.rag.ingest import ingest_pdf as _ingest_pdf

    try:
        chunks = _ingest_pdf(
            pdf_path=str(path),
            source=path.name,
            category="general",
            chunk_type=chunk_type,
            valid_from=valid_from,
        )
        print(f"  [ingest] {path.name}: {chunks} chunks (type={chunk_type})")
        return chunks
    except Exception as e:
        print(f"  [ingest error] {path.name}: {e}")
        return 0


def ingest_text_content(text: str, source: str, chunk_type: str, valid_from: str | None = None) -> int:
    """Ingest text content into hanna_oetp."""
    from app.rag.ingest import ingest_text as _ingest_text

    if len(text.strip()) < 50:
        return 0

    try:
        chunks = _ingest_text(
            text=text,
            source=source,
            category="general",
            chunk_type=chunk_type,
            valid_from=valid_from,
        )
        print(f"  [ingest] {source}: {chunks} chunks (type={chunk_type})")
        return chunks
    except Exception as e:
        print(f"  [ingest error] {source}: {e}")
        return 0


# Mapping from document name patterns to chunk_type
DOC_TYPE_HINTS = {
    "felhívás": "palyazat_felhivas",
    "felhivas": "palyazat_felhivas",
    "módosítás": "palyazat_felhivas",
    "modositas": "palyazat_felhivas",
    "melléklet": "palyazat_melleklet",
    "melleklet": "palyazat_melleklet",
    "útmutató": "palyazat_melleklet",
    "utmutato": "palyazat_melleklet",
    "elszámolhatóság": "palyazat_melleklet",
    "ajánlás": "palyazat_melleklet",
    "meghatalmazás": "segedlet",
    "meghatalmazas": "segedlet",
    "segédlet": "segedlet",
    "segedlet": "segedlet",
    "kitöltési": "segedlet",
    "tájékoztató": "kozlemeny",
    "tajekoztato": "kozlemeny",
    "közlemény": "kozlemeny",
    "kozlemeny": "kozlemeny",
    "gyik": "gyik",
    "gyakran": "gyik",
    "kivitelező": "kozlemeny",
    "regisztráció": "kozlemeny",
}


def guess_chunk_type(name: str) -> str:
    """Guess chunk_type from document name."""
    name_lower = name.lower()
    for keyword, ctype in DOC_TYPE_HINTS.items():
        if keyword in name_lower:
            return ctype
    return "document"


def main():
    parser = argparse.ArgumentParser(description="Scrape OETP közlemények from nffku.hu")
    parser.add_argument("--dry-run", action="store_true", help="Don't download or ingest")
    parser.add_argument("--download-dir", type=str, default="/app/data/pdfs",
                        help="Directory for downloaded files")
    parser.add_argument("--no-ingest", action="store_true", help="Download only, don't ingest")
    args = parser.parse_args()

    download_dir = Path(args.download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    print(f"[scrape] Fetching {PAGE_URL}")

    with httpx.Client(timeout=30, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }) as client:
        soup = fetch_page(client)

        # 1. Extract közlemények
        kozlemenyek = extract_kozlemenyek(soup)
        print(f"\n{'='*60}")
        print(f"  KÖZLEMÉNYEK: {len(kozlemenyek)}")
        print(f"{'='*60}")

        total_chunks = 0

        for i, k in enumerate(kozlemenyek, 1):
            print(f"\n  {i}. [{k['date'] or 'n/a'}] {k['title'][:80]}")
            if k['content']:
                print(f"     Tartalom: {len(k['content'])} karakter")
            if k['links']:
                for link in k['links']:
                    print(f"     📎 {link['name']}: {link['url']}")

            if not args.dry_run and not args.no_ingest and k['content'] and len(k['content']) > 50:
                source = f"kozlemeny:{k['date'] or 'nodate'}:{k['title'][:60]}"
                text = f"Közlemény: {k['title']}\nDátum: {k['date'] or 'n/a'}\n\n{k['content']}"
                total_chunks += ingest_text_content(text, source, "kozlemeny", k['date'])
                time.sleep(2)

            # Download linked documents
            if not args.dry_run:
                for link in k['links']:
                    path = download_file(client, link['url'], download_dir)
                    if path and not args.no_ingest and path.suffix.lower() == '.pdf':
                        ctype = guess_chunk_type(link['name'])
                        time.sleep(5)
                        total_chunks += ingest_pdf(path, ctype, k['date'])
                        time.sleep(2)

        # 2. Extract letölthető dokumentumok
        downloads = extract_downloads(soup)
        print(f"\n{'='*60}")
        print(f"  LETÖLTHETŐ DOKUMENTUMOK: {len(downloads)}")
        print(f"{'='*60}")

        for i, d in enumerate(downloads, 1):
            print(f"\n  {i}. {d['name']}")
            print(f"     URL: {d['url']}")

            if not args.dry_run:
                path = download_file(client, d['url'], download_dir)
                if path and not args.no_ingest and path.suffix.lower() == '.pdf':
                    ctype = guess_chunk_type(d['name'])
                    time.sleep(5)
                    total_chunks += ingest_pdf(path, ctype)
                    time.sleep(2)

        print(f"\n{'='*60}")
        print(f"  ÖSSZESÍTÉS {'(DRY RUN)' if args.dry_run else ''}")
        print(f"{'='*60}")
        print(f"  Közlemények: {len(kozlemenyek)}")
        print(f"  Letölthető dokumentumok: {len(downloads)}")
        if not args.dry_run and not args.no_ingest:
            print(f"  Összes chunk: {total_chunks}")


if __name__ == "__main__":
    main()
