"""Generic config-driven web scraper with escalation.

Fetches pages defined in program.yaml scraper.pages, cleans HTML,
extracts structured content (text, accordions, downloads), detects
changes via hash snapshots, downloads linked files, and ingests into RAG.

Escalation levels:
  1. httpx simple GET
  2. httpx with stealth headers + retry
  3. Browser service (camoufox via HTTP API)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin

import httpx
from bs4 import BeautifulSoup

from ..config import get_program_config

logger = logging.getLogger(__name__)

# Stealth user-agent rotation
_USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36",
]


@dataclass
class ScraperResult:
    url: str
    label: str
    doc_type: str
    html: str = ""
    text: str = ""
    announcements: list[dict] = field(default_factory=list)
    download_links: list[dict] = field(default_factory=list)
    content_hash: str = ""
    method: str = ""
    changed: bool = True
    error: str | None = None


# ---------------------------------------------------------------------------
# Fetch with escalation
# ---------------------------------------------------------------------------

def _is_blocked(html: str) -> bool:
    """Detect Cloudflare / bot protection pages."""
    markers = [
        "cf-browser-verification",
        "cf_chl_opt",
        "just a moment",
        "checking your browser",
        "ray id",
        "attention required",
    ]
    lower = html[:3000].lower()
    return any(m in lower for m in markers)


async def _fetch_simple(url: str) -> tuple[str, int]:
    """Level 1: Simple httpx GET."""
    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        resp = await client.get(url, headers={
            "User-Agent": _USER_AGENTS[0],
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "hu-HU,hu;q=0.9,en;q=0.5",
        })
        return resp.text, resp.status_code


async def _fetch_stealth(url: str) -> tuple[str, int]:
    """Level 2: httpx with rotated headers + retry."""
    for i, ua in enumerate(_USER_AGENTS):
        try:
            async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
                resp = await client.get(url, headers={
                    "User-Agent": ua,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "hu-HU,hu;q=0.9,en-US;q=0.5,en;q=0.3",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Referer": url.rsplit("/", 1)[0] + "/",
                    "DNT": "1",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                })
                if resp.status_code == 200 and not _is_blocked(resp.text):
                    return resp.text, resp.status_code
        except Exception:
            pass
        if i < len(_USER_AGENTS) - 1:
            await asyncio.sleep(2)
    raise RuntimeError("All stealth attempts failed")


async def _fetch_browser(url: str, browser_url: str, timeout: int = 30,
                         wait_selector: str | None = None) -> tuple[str, int]:
    """Level 3: Browser service (camoufox) via HTTP API."""
    payload: dict[str, Any] = {"url": url}
    if wait_selector:
        payload["wait_selector"] = wait_selector
    async with httpx.AsyncClient(timeout=timeout + 10) as client:
        resp = await client.post(f"{browser_url}/scrape", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("html", ""), 200


async def fetch_with_escalation(url: str, config: dict,
                                selectors: dict | None = None) -> tuple[str, str]:
    """Fetch URL with escalation: simple → stealth → browser.

    Returns (html, method_used).
    """
    method_pref = config.get("method", "auto")

    # Level 1: Simple
    if method_pref in ("auto", "simple"):
        try:
            html, status = await _fetch_simple(url)
            if status == 200 and len(html) > 500 and not _is_blocked(html):
                return html, "httpx"
        except Exception as e:
            logger.debug("Simple fetch failed for %s: %s", url, e)

    if method_pref == "simple":
        raise RuntimeError(f"Simple fetch failed for {url}")

    # Level 2: Stealth
    if method_pref in ("auto",):
        try:
            html, status = await _fetch_stealth(url)
            if status == 200 and len(html) > 500:
                return html, "httpx_stealth"
        except Exception as e:
            logger.debug("Stealth fetch failed for %s: %s", url, e)

    # Level 3: Browser service
    browser_url = config.get("browser_service_url", "")
    if browser_url and method_pref in ("auto", "browser"):
        try:
            wait_sel = (selectors or {}).get("content")
            html, _ = await _fetch_browser(url, browser_url,
                                           config.get("browser_timeout", 30), wait_sel)
            if html and len(html) > 200:
                return html, "browser"
        except Exception as e:
            logger.warning("Browser fetch failed for %s: %s", url, e)

    raise RuntimeError(f"All fetch methods failed for {url}")


# ---------------------------------------------------------------------------
# HTML cleaning pipeline
# ---------------------------------------------------------------------------

def clean_html(raw_html: str, selectors: dict | None = None) -> tuple[str, str]:
    """Clean raw HTML → (cleaned_html, plain_text).

    1. Remove boilerplate (nav, footer, script, style, etc.)
    2. Scope to content selector if specified
    3. Expand accordions inline
    4. Extract plain text
    """
    selectors = selectors or {}
    soup = BeautifulSoup(raw_html, "html.parser")

    # 1. Remove unwanted elements
    for tag_name in selectors.get("remove", ["script", "style", "nav", "footer", "header"]):
        for el in soup.find_all(tag_name):
            el.decompose()
    # Also remove cookie banners, overlays
    for el in soup.find_all(attrs={"class": re.compile(r"cookie|consent|overlay|popup", re.I)}):
        el.decompose()

    # 2. Scope to content area
    content_sel = selectors.get("content")
    if content_sel:
        main = soup.select_one(content_sel)
        if main:
            soup = BeautifulSoup(str(main), "html.parser")

    # 3. Expand accordions
    acc_btn = selectors.get("accordion_button")
    acc_panel = selectors.get("accordion_panel")
    if acc_btn and acc_panel:
        _expand_accordions(soup, acc_btn, acc_panel)

    # 4. Extract text
    text = soup.get_text(separator="\n", strip=True)
    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)

    return str(soup), text


def _expand_accordions(soup: BeautifulSoup, btn_sel: str, panel_sel: str):
    """Make accordion content visible by removing hidden attributes."""
    for panel in soup.select(panel_sel):
        # Remove display:none, hidden attr, collapsed class
        if panel.get("style"):
            panel["style"] = re.sub(r"display\s*:\s*none", "", panel["style"])
        panel.attrs.pop("hidden", None)
        if "aria-hidden" in panel.attrs:
            panel["aria-hidden"] = "false"


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------

def extract_announcements(soup: BeautifulSoup, selectors: dict | None = None) -> list[dict]:
    """Extract date-titled announcements (inline + accordion)."""
    selectors = selectors or {}
    results = []
    date_pattern = re.compile(r"(\d{4})[\.\s]+(\d{2})[\.\s]+(\d{2})")

    # Inline: <p><strong>DATE – Title</strong> blocks
    for p in soup.find_all("p"):
        strong = p.find("strong")
        if not strong:
            continue
        text = strong.get_text(strip=True)
        m = date_pattern.search(text)
        if not m:
            continue
        date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        content_parts = [p.get_text(separator="\n", strip=True)]
        for sib in p.find_next_siblings():
            if sib.name == "hr" or sib.get("data-rlta-element"):
                break
            content_parts.append(sib.get_text(separator="\n", strip=True))
        results.append({"title": text, "date": date, "content": "\n".join(content_parts)})

    # Accordion headings
    acc_btn = selectors.get("accordion_button")
    acc_heading = selectors.get("accordion_heading", "h3")
    if acc_btn:
        for btn in soup.select(acc_btn):
            h = btn.find(acc_heading.split("[")[0]) if acc_heading else btn.find("h3")
            if not h:
                continue
            title = h.get_text(strip=True)
            m = date_pattern.search(title)
            date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None
            # Find panel content
            panel_id = btn.get("aria-controls", "")
            panel = soup.find(id=panel_id) if panel_id else btn.find_next_sibling("div")
            content = panel.get_text(separator="\n", strip=True) if panel else ""
            results.append({"title": title, "date": date, "content": content})

    return results


def extract_download_links(soup: BeautifulSoup, base_url: str,
                           selectors: dict | None = None,
                           extensions: list[str] | None = None) -> list[dict]:
    """Extract file download links from the page."""
    extensions = extensions or [".pdf", ".docx", ".xlsx"]
    selectors = selectors or {}
    links = []
    seen_urls = set()

    # From downloads table (if configured)
    table_sel = selectors.get("downloads_table")
    if table_sel:
        for table in soup.select(table_sel):
            for a in table.find_all("a", href=True):
                href = a["href"]
                if any(href.lower().endswith(ext) for ext in extensions):
                    full_url = href if href.startswith("http") else urljoin(base_url, href)
                    if full_url not in seen_urls:
                        seen_urls.add(full_url)
                        filename = unquote(os.path.basename(href.split("?")[0]))
                        links.append({"url": full_url, "filename": filename,
                                      "label": a.get_text(strip=True) or filename})

    # From all links on page (catch downloads outside table)
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if any(href.lower().endswith(ext) for ext in extensions):
            full_url = href if href.startswith("http") else urljoin(base_url, href)
            if full_url not in seen_urls:
                seen_urls.add(full_url)
                filename = unquote(os.path.basename(href.split("?")[0]))
                links.append({"url": full_url, "filename": filename,
                              "label": a.get_text(strip=True) or filename})

    return links


# ---------------------------------------------------------------------------
# Snapshot + change detection
# ---------------------------------------------------------------------------

def _url_slug(url: str) -> str:
    """Convert URL to filesystem-safe slug."""
    return re.sub(r"[^a-zA-Z0-9]", "_", url)[:80]


def check_changed(url: str, new_hash: str, snapshot_dir: Path) -> bool:
    """Check if content hash differs from last snapshot."""
    latest = snapshot_dir / f"{_url_slug(url)}_latest.json"
    if not latest.exists():
        return True
    try:
        old = json.loads(latest.read_text())
        return old.get("content_hash") != new_hash
    except Exception:
        return True


def save_snapshot(url: str, result: ScraperResult, snapshot_dir: Path):
    """Save snapshot with timestamp and as latest."""
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    slug = _url_slug(url)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    data = {
        "url": url,
        "label": result.label,
        "scraped_at": datetime.now().isoformat(timespec="seconds"),
        "content_hash": result.content_hash,
        "text_length": len(result.text),
        "method": result.method,
        "announcement_count": len(result.announcements),
        "download_count": len(result.download_links),
    }

    (snapshot_dir / f"{slug}_{ts}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))
    (snapshot_dir / f"{slug}_latest.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Download pipeline
# ---------------------------------------------------------------------------

async def download_files(links: list[dict], download_dir: Path,
                         extensions: list[str] | None = None,
                         rate_limit: float = 3.0) -> list[Path]:
    """Download files with dedup and rate limiting."""
    extensions = extensions or [".pdf", ".docx", ".xlsx"]
    download_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []

    async with httpx.AsyncClient(timeout=60, follow_redirects=True, headers={
        "User-Agent": _USER_AGENTS[0],
    }) as client:
        for link in links:
            ext = Path(link["filename"]).suffix.lower()
            if ext not in extensions:
                continue
            dest = download_dir / link["filename"]
            if dest.exists():
                logger.info("  [skip] Already exists: %s", link["filename"])
                downloaded.append(dest)
                continue
            try:
                resp = await client.get(link["url"])
                resp.raise_for_status()
                dest.write_bytes(resp.content)
                logger.info("  [ok] Downloaded: %s (%d bytes)", link["filename"], len(resp.content))
                downloaded.append(dest)
            except Exception as e:
                logger.warning("  [error] Download failed %s: %s", link["url"], e)
            await asyncio.sleep(rate_limit)

    return downloaded


# ---------------------------------------------------------------------------
# Doc type guessing from filename
# ---------------------------------------------------------------------------

def guess_doc_type(filename: str) -> str:
    """Guess doc_type from filename using program.yaml doc_types keywords."""
    pcfg = get_program_config()
    doc_types = pcfg.get("doc_types", {})
    name_lower = filename.lower()

    for dt_name, dt_cfg in doc_types.items():
        if not isinstance(dt_cfg, dict):
            continue
        for kw in dt_cfg.get("keywords", []):
            if kw.lower() in name_lower:
                return dt_name

    return "dokumentum"


# ---------------------------------------------------------------------------
# Main scraper class
# ---------------------------------------------------------------------------

class Scraper:
    """Config-driven scraper with escalation, cleaning, and ingest."""

    def __init__(self, config: dict | None = None):
        if config is None:
            config = get_program_config().get("scraper", {})
        self.config = config
        self.pages = config.get("pages", [])
        self.extensions = config.get("download_extensions", [".pdf", ".docx", ".xlsx"])
        self.download_dir = Path(config.get("download_dir", "data/downloads"))
        self.snapshot_dir = Path(config.get("snapshot_dir", "data/scraper_snapshots"))
        self.rate_limit = config.get("rate_limit_seconds", 3)

    async def scrape_page(self, page_cfg: dict) -> ScraperResult:
        """Fetch, clean, and extract content from a single page."""
        url = page_cfg["url"]
        label = page_cfg.get("label", url)
        doc_type = page_cfg.get("doc_type", "dokumentum")
        selectors = page_cfg.get("selectors", {})
        base_url = url.rsplit("/", 1)[0] + "/"

        result = ScraperResult(url=url, label=label, doc_type=doc_type)

        try:
            raw_html, method = await fetch_with_escalation(url, self.config, selectors)
            result.method = method

            # Clean
            result.html, result.text = clean_html(raw_html, selectors)
            result.content_hash = hashlib.sha256(result.text.encode()).hexdigest()[:16]

            # Extract structured content
            soup = BeautifulSoup(result.html, "html.parser")
            result.announcements = extract_announcements(soup, selectors)
            result.download_links = extract_download_links(soup, base_url, selectors, self.extensions)

        except Exception as e:
            result.error = str(e)
            logger.error("Scrape failed for %s: %s", url, e)

        return result

    async def scrape_all(self, force: bool = False, page_index: int | None = None) -> list[ScraperResult]:
        """Scrape all configured pages. Skip unchanged unless force=True."""
        pages = self.pages
        if page_index is not None:
            pages = [pages[page_index]]

        results = []
        for i, page_cfg in enumerate(pages):
            logger.info("Scraping page %d/%d: %s", i + 1, len(pages), page_cfg.get("label", page_cfg["url"]))
            result = await self.scrape_page(page_cfg)

            if result.error:
                results.append(result)
                continue

            # Change detection
            if not force and not check_changed(result.url, result.content_hash, self.snapshot_dir):
                result.changed = False
                logger.info("  No changes (hash=%s)", result.content_hash)
            else:
                result.changed = True
                save_snapshot(result.url, result, self.snapshot_dir)
                logger.info("  Changed! hash=%s, method=%s, %d announcements, %d downloads",
                            result.content_hash, result.method,
                            len(result.announcements), len(result.download_links))

            results.append(result)

            if i < len(pages) - 1:
                await asyncio.sleep(self.rate_limit)

        return results

    async def download_all(self, results: list[ScraperResult]) -> list[Path]:
        """Download all files from scrape results."""
        all_links = []
        for r in results:
            if r.changed and not r.error:
                all_links.extend(r.download_links)
        if not all_links:
            return []
        return await download_files(all_links, self.download_dir, self.extensions, self.rate_limit)

    async def ingest_results(self, results: list[ScraperResult], downloaded_files: list[Path] | None = None):
        """Ingest scraped content + downloaded files into RAG."""
        from .ingest import ingest_text, ingest_pdf

        total_chunks = 0

        for r in results:
            if not r.changed or r.error:
                continue

            # Ingest announcements as individual text chunks
            for ann in r.announcements:
                if len(ann.get("content", "")) < 50:
                    continue
                source = f"scrape:{r.label}:{ann.get('date', 'nodate')}:{ann['title'][:50]}"
                text = f"{ann['title']}\nDátum: {ann.get('date', 'n/a')}\n\n{ann['content']}"
                try:
                    n = ingest_text(
                        text=text, source=source,
                        category="general", chunk_type=r.doc_type,
                        valid_from=ann.get("date"),
                    )
                    total_chunks += n or 0
                except Exception as e:
                    logger.warning("Ingest text failed for %s: %s", source, e)

            # If no announcements, ingest full page text
            if not r.announcements and len(r.text) > 100:
                source = f"scrape:{r.label}"
                try:
                    n = ingest_text(
                        text=r.text, source=source,
                        category="general", chunk_type=r.doc_type,
                    )
                    total_chunks += n or 0
                except Exception as e:
                    logger.warning("Ingest page text failed for %s: %s", source, e)

        # Ingest downloaded PDFs
        for path in (downloaded_files or []):
            if path.suffix.lower() != ".pdf":
                continue
            doc_type = guess_doc_type(path.name)
            try:
                n = ingest_pdf(
                    pdf_path=str(path), source=path.name,
                    category="general", chunk_type=doc_type,
                )
                total_chunks += n or 0
            except Exception as e:
                logger.warning("Ingest PDF failed for %s: %s", path.name, e)

        logger.info("Ingest complete: %d total chunks", total_chunks)
        return total_chunks
