#!/usr/bin/env python3
"""Live email evaluation — test CustomerCare against real colleague answers.

Fetches sent emails from Outlook, extracts question/answer pairs,
generates CC drafts for each question, and compares against
the actual colleague answer using semantic + text + style metrics.

This is the production counterpart to eval_golden_set.py:
- Golden set: fixed questions, deterministic RAG precision
- Live eval:  real emails, end-to-end draft quality

Usage (inside cc-backend container):
    python3 /app/scripts/eval_live.py --limit 50
    python3 /app/scripts/eval_live.py --limit 250 --report
    python3 /app/scripts/eval_live.py --limit 20 --dry-run

Or from host:
    docker exec cc-backend python3 /app/scripts/eval_live.py --limit 50
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent))

CC_URL = "http://localhost:8000"
GRAPH_BASE = "https://graph.microsoft.com/v1.0"
DATA_DIR = Path(__file__).parent.parent / "data"
OBSIDIAN_REPORTS = Path.home() / "Library/Mobile Documents/iCloud~md~obsidian/Documents/PARA/!inbox/!reports"

INTERNAL_DOMAINS = {"neuzrt.hu", "nffku.hu", "nffku.onmicrosoft.com", "norvegalap.hu"}

AUTO_REPLY_KEYWORDS = [
    "automatic reply", "automatikus válasz", "out of office",
    "delivery status", "undeliverable", "mailer-daemon",
    "nem kézbesíthető", "szabadságom",
]

NEU_SIG_MARKERS = [
    "Nemzeti Energetikai Ügynökség",
    "1037- Budapest", "1037 Budapest",
    "Montevideo u.",
]

# ── Helpers ──

def _html_to_text(html: str) -> str:
    if not html:
        return ""
    return BeautifulSoup(html, "html.parser").get_text(separator="\n", strip=True)


def _text_sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower()[:2000], b.lower()[:2000]).ratio()


async def _semantic_sim(a: str, b: str) -> float:
    """Embedding-based cosine similarity via BGE-M3."""
    if not a or not b:
        return 0.0
    try:
        from app.rag.embeddings import embed_texts
        embs = embed_texts([a[:1000], b[:1000]])
        if len(embs) != 2:
            return _text_sim(a, b)
        dot = sum(x * y for x, y in zip(embs[0], embs[1]))
        na = math.sqrt(sum(x * x for x in embs[0]))
        nb = math.sqrt(sum(x * x for x in embs[1]))
        return dot / (na * nb) if na and nb else 0.0
    except Exception:
        return _text_sim(a, b)


def _is_auto_reply(subject: str, body: str) -> bool:
    subj_lower = (subject or "").lower()
    body_lower = (body or "").lower()[:200]
    return any(kw in subj_lower or kw in body_lower for kw in AUTO_REPLY_KEYWORDS)


def _is_internal(email: str) -> bool:
    return (email or "").lower().split("@")[-1] in INTERNAL_DOMAINS


def _extract_app_ids(text: str) -> set[str]:
    return set(re.findall(r"OETP-\d{4}-\d{4,8}", text, re.IGNORECASE))


def _extract_key_terms(text: str) -> set[str]:
    terms = set()
    terms.update(re.findall(r"OETP-\d{4}-\d+", text))
    terms.update(re.findall(r"\d+/\d{4}", text))
    terms.update(re.findall(r"(\d+\.\d+\.?\s*pont)", text))
    domain = [
        "szaldó", "bruttó", "inverter", "akkumulátor", "tároló",
        "meghatalmazás", "meghatalmazott", "igazolási", "fenntartási",
        "kivitelező", "vállalkozási", "felhívás", "pályázat",
        "tulajdon", "társasház", "végrehajtás", "elutasít",
        "hiánypótl", "támogatás", "2,5 millió", "POD",
        "gazdasági tevékenység", "szaldó elszámolás",
    ]
    text_lower = text.lower()
    for w in domain:
        if w.lower() in text_lower:
            terms.add(w.lower())
    return terms


# ── Email parsing ──

_QUOTE_MARKERS = [
    re.compile(r"^From:\s*$", re.MULTILINE),
    re.compile(r"^From:\s+\S", re.MULTILINE),
    re.compile(r"^Feladó:\s*$", re.MULTILINE),
    re.compile(r"^Feladó:\s+\S", re.MULTILINE),
    re.compile(r"^-{3,}\s*(Eredeti üzenet|Original Message|Továbbított üzenet|Forwarded message)", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^[_=]{10,}$", re.MULTILINE),
]

_HEADER_LINE = re.compile(
    r"^(From|Feladó|Sent|Elküldve|To|Címzett|Subject|Tárgy|Date|Dátum|Cc|Másolat|Küldés ideje)\s*:",
    re.IGNORECASE,
)


def split_answer_question(body: str) -> tuple[str, str]:
    """Split sent email body into (NEÜ answer, customer question).

    Returns (answer, question). Either can be empty.
    """
    lines = body.split("\n")

    # Find first quote boundary
    split_idx = None
    for i, line in enumerate(lines):
        for pattern in _QUOTE_MARKERS:
            if pattern.search(line):
                split_idx = i
                break
        if split_idx is not None:
            break

    if split_idx is None:
        return _strip_signature(body), ""

    answer_part = "\n".join(lines[:split_idx])
    quoted_lines = lines[split_idx:]

    # Skip header block in quoted section to get to actual question text
    question_lines = []
    in_headers = True
    for line in quoted_lines:
        stripped = line.strip()
        if in_headers:
            if not stripped:
                continue
            if re.match(r"^[-_=]{3,}", stripped):
                continue
            if _HEADER_LINE.match(stripped):
                continue
            if re.match(r"^<?[\w.+-]+@[\w.-]+>?\s*$", stripped):
                continue
            # Day names (English/Hungarian) or date-only lines
            if re.match(r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)", stripped, re.IGNORECASE):
                continue
            if re.match(r"^\d{4}\.\s*(január|február|március|április|május|június|július|augusztus|szeptember|október|november|december)", stripped, re.IGNORECASE):
                continue
            if "forwarded message" in stripped.lower():
                continue
            # Name-only line right after From: (e.g. "Fendt János <email>")
            if re.match(r"^[A-ZÁÉÍÓÖŐÚÜŰa-záéíóöőúüű\s.,-]+<[\w.+-]+@[\w.-]+>$", stripped):
                continue
            in_headers = False
        question_lines.append(line)

    return _strip_signature(answer_part), "\n".join(question_lines).strip()


def _strip_signature(text: str) -> str:
    """Remove NEÜ signature block from answer."""
    lines = text.split("\n")
    sig_start = None
    for i, line in enumerate(lines):
        if any(p in line for p in NEU_SIG_MARKERS):
            sig_start = i
            break
    if sig_start is not None and sig_start > 2:
        # Also remove "Üdvözlettel" line before
        if sig_start > 0 and "dvözlettel" in lines[sig_start - 1]:
            sig_start -= 1
        text = "\n".join(lines[:sig_start])
    return text.strip()


# ── Graph API ──

async def fetch_sent_emails(mailbox: str, max_items: int = 250) -> list[dict]:
    """Fetch sent emails with pagination."""
    from app.email.auth import get_auth_headers
    headers = get_auth_headers()

    url = f"{GRAPH_BASE}/users/{mailbox}/mailFolders/sentItems/messages"
    params = {
        "$orderby": "sentDateTime desc",
        "$top": min(max_items, 50),
        "$select": "id,subject,from,toRecipients,body,sentDateTime,hasAttachments",
    }

    all_messages = []
    async with httpx.AsyncClient(timeout=60) as client:
        while url and len(all_messages) < max_items:
            resp = await client.get(url, headers=headers, params=params)
            if resp.status_code != 200:
                print(f"[eval] Fetch error: {resp.status_code}")
                break
            data = resp.json()
            all_messages.extend(data.get("value", []))
            url = data.get("@odata.nextLink")
            params = {}
            time.sleep(0.1)

    return all_messages[:max_items]


# ── CC draft ──

async def generate_draft(question: str, subject: str, sender_name: str = "", sender_email: str = "") -> dict:
    """Call CC /draft/generate."""
    app_ids = list(_extract_app_ids(question))
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{CC_URL}/draft/generate", json={
            "email_text": question[:3000],
            "email_subject": re.sub(r"^(RE|FW|Fwd|Vá|VS|AW|SV)\s*:\s*", "", subject, flags=re.IGNORECASE).strip(),
            "sender_name": sender_name,
            "sender_email": sender_email,
            "app_ids": app_ids,
            "top_k": 5,
            "max_context_chunks": 3,
        })
        if resp.status_code == 200:
            return resp.json()
        return {"error": f"HTTP {resp.status_code}"}


# ── Eval result ──

@dataclass
class EvalResult:
    msg_id: str
    subject: str
    sent_date: str
    status: str  # MATCH / PARTIAL / MISMATCH / SKIP / ERROR
    # Scores
    semantic_sim: float = 0.0
    text_sim: float = 0.0
    combined_sim: float = 0.0
    style_score: float = 0.0
    key_term_overlap: float = 0.0
    section_match: bool = True
    app_ids_match: bool = True
    # Metadata
    confidence: str = ""
    llm_provider: str = ""
    duration_s: float = 0.0
    question_len: int = 0
    answer_len: int = 0
    draft_len: int = 0
    # Previews (for analysis)
    question_preview: str = ""
    answer_preview: str = ""
    draft_preview: str = ""
    error: str = ""
    sources: list[str] = field(default_factory=list)
    style_components: dict = field(default_factory=dict)


# ── Main pipeline ──

async def run_eval(
    mailbox: str = "lakossagitarolo@neuzrt.hu",
    max_items: int = 250,
    dry_run: bool = False,
    generate_report: bool = False,
) -> dict:
    print(f"[eval] Fetching {max_items} sent emails from {mailbox}...")
    sent_emails = await fetch_sent_emails(mailbox, max_items)
    print(f"[eval] Got {len(sent_emails)} emails")

    results: list[EvalResult] = []
    skips = Counter()

    for i, msg in enumerate(sent_emails):
        subject = msg.get("subject", "")
        to_recips = msg.get("toRecipients", [])
        to_addr = to_recips[0].get("emailAddress", {}).get("address", "") if to_recips else ""
        to_name = to_recips[0].get("emailAddress", {}).get("name", "") if to_recips else ""
        body_html = msg.get("body", {}).get("content", "")
        body_text = _html_to_text(body_html)
        sent_date = msg.get("sentDateTime", "")[:10]
        msg_id = hashlib.sha256(msg.get("id", "").encode()).hexdigest()[:12]

        # Filters
        if _is_auto_reply(subject, body_text):
            skips["auto_reply"] += 1
            continue
        if to_addr and _is_internal(to_addr):
            skips["internal"] += 1
            continue
        if len(body_text) < 50:
            skips["short"] += 1
            continue

        actual_answer, question = split_answer_question(body_text)

        if not question or len(question) < 30:
            skips["no_question"] += 1
            continue
        if len(actual_answer) < 20:
            skips["short_answer"] += 1
            continue

        if dry_run:
            n = len(results) + 1
            print(f"  [{n}] {subject[:55]} — Q:{len(question)} A:{len(actual_answer)}")
            if n <= 5:
                # Show cleaned question
                q_clean = question[:150].replace("\n", " ")
                print(f"      Q: {q_clean}...")
                print(f"      A: {actual_answer[:100]}...")
            results.append(EvalResult(
                msg_id=msg_id, subject=subject[:80], sent_date=sent_date,
                status="DRY_RUN", question_len=len(question), answer_len=len(actual_answer),
                question_preview=question[:300], answer_preview=actual_answer[:300],
            ))
            continue

        # Generate CC draft
        n = len([r for r in results if r.status not in ("DRY_RUN",)]) + 1
        print(f"  [{n}] {subject[:50]}...", end=" ", flush=True)

        t0 = time.time()
        draft_result = await generate_draft(question, subject, sender_name=to_name, sender_email=to_addr)
        duration = time.time() - t0

        if "error" in draft_result:
            results.append(EvalResult(
                msg_id=msg_id, subject=subject[:80], sent_date=sent_date,
                status="ERROR", error=str(draft_result.get("error", "")),
                question_len=len(question), answer_len=len(actual_answer),
            ))
            print(f"ERROR ({draft_result.get('error', '')[:40]})")
            continue

        if draft_result.get("skip"):
            results.append(EvalResult(
                msg_id=msg_id, subject=subject[:80], sent_date=sent_date,
                status="SKIP", question_len=len(question), answer_len=len(actual_answer),
            ))
            print("SKIP")
            continue

        draft_html = draft_result.get("body_html", "")
        draft_text = _html_to_text(draft_html)

        # ── Compute metrics ──
        sem_sim = await _semantic_sim(draft_text, actual_answer)
        txt_sim = _text_sim(draft_text, actual_answer)
        combined = sem_sim * 0.7 + txt_sim * 0.3

        # Style score
        from app.reasoning.style_score import compute_style_score
        style = compute_style_score(draft_text, actual_answer, to_name)

        # Key term overlap (Jaccard)
        answer_terms = _extract_key_terms(actual_answer)
        draft_terms = _extract_key_terms(draft_text)
        union = answer_terms | draft_terms
        term_overlap = len(answer_terms & draft_terms) / len(union) if union else 1.0

        # Section reference check
        answer_sections = set(re.findall(r"(\d+\.\d+\.?\s*pont)", actual_answer))
        draft_sections = set(re.findall(r"(\d+\.\d+\.?\s*pont)", draft_text))
        section_match = bool(answer_sections & draft_sections) if answer_sections else True

        # OETP ID match
        answer_ids = _extract_app_ids(actual_answer)
        draft_ids = _extract_app_ids(draft_text)
        app_match = bool(answer_ids & draft_ids) if answer_ids else True

        # Status
        status = "MATCH" if combined >= 0.5 else "PARTIAL" if combined >= 0.25 else "MISMATCH"

        # Sources used
        sources = []
        for chunk in draft_result.get("sources", []):
            if isinstance(chunk, dict):
                ct = chunk.get("chunk_type", chunk.get("doc_type", "?"))
                sources.append(ct)

        result = EvalResult(
            msg_id=msg_id, subject=subject[:80], sent_date=sent_date,
            status=status,
            semantic_sim=round(sem_sim, 3),
            text_sim=round(txt_sim, 3),
            combined_sim=round(combined, 3),
            style_score=style["overall"],
            key_term_overlap=round(term_overlap, 3),
            section_match=section_match,
            app_ids_match=app_match,
            confidence=str(draft_result.get("confidence", "?")),
            llm_provider=str(draft_result.get("llm_provider", "?")),
            duration_s=round(duration, 1),
            question_len=len(question),
            answer_len=len(actual_answer),
            draft_len=len(draft_text),
            question_preview=question[:300],
            answer_preview=actual_answer[:300],
            draft_preview=draft_text[:300],
            sources=sources[:5],
            style_components=style.get("components", {}),
        )
        results.append(result)

        print(f"{status} (sem={sem_sim:.2f} style={style['overall']:.2f} terms={term_overlap:.2f} {duration:.1f}s)")
        await asyncio.sleep(0.5)

    # ── Summary ──
    tested = [r for r in results if r.status in ("MATCH", "PARTIAL", "MISMATCH")]
    stats = _compute_stats(results, tested, skips)
    _print_summary(stats, dry_run)

    # Save JSON
    output_path = DATA_DIR / "eval_live_results.json"
    report_data = {
        "eval_date": datetime.now(timezone.utc).isoformat(),
        "mailbox": mailbox,
        "stats": stats,
        "results": [asdict(r) for r in results],
    }
    output_path.write_text(json.dumps(report_data, ensure_ascii=False, indent=2))
    print(f"\n[eval] JSON → {output_path}")

    # Obsidian report
    if generate_report and not dry_run and tested:
        _write_obsidian_report(stats, results, tested)

    return stats


def _compute_stats(results: list[EvalResult], tested: list[EvalResult], skips: Counter) -> dict:
    stats = {
        "total_fetched": len(results) + sum(skips.values()),
        "usable": len(results),
        "tested": len(tested),
        "skips": dict(skips),
    }

    if tested:
        stats["match"] = sum(1 for r in tested if r.status == "MATCH")
        stats["partial"] = sum(1 for r in tested if r.status == "PARTIAL")
        stats["mismatch"] = sum(1 for r in tested if r.status == "MISMATCH")
        stats["errors"] = sum(1 for r in results if r.status == "ERROR")
        stats["match_rate"] = round(stats["match"] / len(tested), 3)
        stats["avg_semantic"] = round(sum(r.semantic_sim for r in tested) / len(tested), 3)
        stats["avg_combined"] = round(sum(r.combined_sim for r in tested) / len(tested), 3)
        stats["avg_style"] = round(sum(r.style_score for r in tested) / len(tested), 3)
        stats["avg_term_overlap"] = round(sum(r.key_term_overlap for r in tested) / len(tested), 3)
        stats["section_match_rate"] = round(sum(1 for r in tested if r.section_match) / len(tested), 3)
        stats["app_id_match_rate"] = round(sum(1 for r in tested if r.app_ids_match) / len(tested), 3)
        stats["confidence_dist"] = dict(Counter(r.confidence for r in tested))
        # Style components avg
        components = {}
        for key in ("greeting", "length", "closing", "formality", "brevity"):
            vals = [r.style_components.get(key, 0) for r in tested if r.style_components]
            components[key] = round(sum(vals) / len(vals), 3) if vals else 0
        stats["style_components"] = components

    return stats


def _print_summary(stats: dict, dry_run: bool):
    print(f"\n{'='*60}")
    print(f"  LIVE EVAL {'(DRY RUN)' if dry_run else 'ÖSSZESÍTÉS'}")
    print(f"{'='*60}")
    print(f"  Letöltve:           {stats['total_fetched']}")
    print(f"  Használható:        {stats['usable']}")
    print(f"  Tesztelve:          {stats.get('tested', 0)}")
    if stats.get("skips"):
        for k, v in stats["skips"].items():
            print(f"  Skip ({k}):  {v}")

    if stats.get("tested"):
        print(f"\n  MATCH (≥0.5):       {stats['match']} ({stats['match_rate']:.0%})")
        print(f"  PARTIAL (0.25-0.5): {stats['partial']}")
        print(f"  MISMATCH (<0.25):   {stats['mismatch']}")
        print(f"  Hibák:              {stats['errors']}")
        print(f"\n  Átlag semantic sim: {stats['avg_semantic']:.3f}")
        print(f"  Átlag combined:     {stats['avg_combined']:.3f}")
        print(f"  Átlag style:        {stats['avg_style']:.3f}")
        print(f"  Átlag term overlap: {stats['avg_term_overlap']:.3f}")
        print(f"  Section ref match:  {stats['section_match_rate']:.0%}")
        print(f"  OETP ID match:      {stats['app_id_match_rate']:.0%}")
        print(f"  Confidence:         {stats['confidence_dist']}")
        print(f"\n  Style komponensek:")
        for k, v in stats.get("style_components", {}).items():
            print(f"    {k}: {v:.3f}")


def _write_obsidian_report(stats: dict, results: list[EvalResult], tested: list[EvalResult]):
    """Write evaluation report to Obsidian vault."""
    now = datetime.now()
    date_str = now.strftime("%y%m%d")
    filename = f"{date_str}-cc-live-eval.md"
    filepath = OBSIDIAN_REPORTS / filename

    mismatches = [r for r in tested if r.status == "MISMATCH"][:5]

    md = f"""# CustomerCare Live Eval — {now.strftime('%Y-%m-%d %H:%M')}

## Összefoglaló

| Metrika | Érték |
|---------|-------|
| Tesztelve | {stats['tested']} email |
| MATCH (≥0.5) | {stats['match']} ({stats['match_rate']:.0%}) |
| PARTIAL | {stats['partial']} |
| MISMATCH | {stats['mismatch']} |
| Semantic sim (átlag) | {stats['avg_semantic']:.3f} |
| Combined sim (átlag) | {stats['avg_combined']:.3f} |
| Style score (átlag) | {stats['avg_style']:.3f} |
| Term overlap (átlag) | {stats['avg_term_overlap']:.3f} |
| Section ref match | {stats['section_match_rate']:.0%} |
| OETP ID match | {stats['app_id_match_rate']:.0%} |
| Confidence | {stats['confidence_dist']} |

## Style komponensek

| Komponens | Score |
|-----------|-------|
"""
    for k, v in stats.get("style_components", {}).items():
        md += f"| {k} | {v:.3f} |\n"

    if mismatches:
        md += "\n## Top 5 MISMATCH (elemzendő)\n\n"
        for r in mismatches:
            md += f"### {r.subject}\n"
            md += f"- Semantic: {r.semantic_sim:.3f}, Style: {r.style_score:.3f}\n"
            md += f"- Confidence: {r.confidence}\n"
            md += f"- **Kérdés**: {r.question_preview[:200]}...\n"
            md += f"- **Kolléga**: {r.answer_preview[:200]}...\n"
            md += f"- **CC**: {r.draft_preview[:200]}...\n\n"

    try:
        OBSIDIAN_REPORTS.mkdir(parents=True, exist_ok=True)
        filepath.write_text(md, encoding="utf-8")
        print(f"[eval] Obsidian report → {filepath}")
    except Exception as e:
        print(f"[eval] Report write failed: {e}")


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(description="CustomerCare live email evaluation")
    parser.add_argument("--limit", type=int, default=50, help="Max emails to fetch (default 50)")
    parser.add_argument("--mailbox", default="lakossagitarolo@neuzrt.hu")
    parser.add_argument("--dry-run", action="store_true", help="Only parse, don't call CC API")
    parser.add_argument("--report", action="store_true", help="Generate Obsidian report")
    args = parser.parse_args()

    asyncio.run(run_eval(
        mailbox=args.mailbox,
        max_items=args.limit,
        dry_run=args.dry_run,
        generate_report=args.report,
    ))


if __name__ == "__main__":
    main()
