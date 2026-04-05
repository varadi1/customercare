#!/usr/bin/env python3
"""
Golden set evaluation — compare Hanna's answers against known-good responses.

Uses golden_set_eval_partials.json + any additional golden set files.
For each question: generate Hanna draft, compare with expected answer.

Usage: python3 scripts/eval_golden_set.py
"""
from __future__ import annotations

import asyncio
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

HANNA_URL = "http://localhost:8000"
DATA_DIR = Path(__file__).parent.parent / "data"
OBSIDIAN_REPORTS = Path.home() / "Library/Mobile Documents/iCloud~md~obsidian/Documents/PARA/!inbox/!reports"


def _html_to_text(html: str) -> str:
    from bs4 import BeautifulSoup
    return BeautifulSoup(html, "html.parser").get_text(separator="\n", strip=True)


def _text_sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


async def _semantic_sim(a: str, b: str) -> float:
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


def load_golden_sets() -> list[dict]:
    """Load all golden set files from data directory."""
    entries = []
    for path in sorted(DATA_DIR.glob("golden_set*.json")):
        try:
            data = json.loads(path.read_text())
            if isinstance(data, list):
                entries.extend(data)
            print(f"Loaded {len(data)} entries from {path.name}")
        except Exception as e:
            print(f"Error loading {path}: {e}")
    return entries


async def generate_draft(question: str, subject: str = "") -> dict:
    payload = {
        "email_text": question[:3000],
        "email_subject": subject,
        "top_k": 5,
        "max_context_chunks": 3,
    }
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{HANNA_URL}/draft/generate", json=payload)
        if resp.status_code == 200:
            return resp.json()
        return {"error": f"HTTP {resp.status_code}", "body_html": ""}


async def run_golden_eval():
    entries = load_golden_sets()
    if not entries:
        print("No golden set entries found in data/golden_set*.json")
        return

    print(f"\n{'='*60}")
    print(f"Running golden set evaluation: {len(entries)} entries")
    print(f"{'='*60}\n")

    results = []
    for i, entry in enumerate(entries):
        question = entry.get("question", "")
        expected = entry.get("expected_answer", "")
        subject = entry.get("subject", "")

        if not question or not expected:
            continue

        print(f"[{i+1}/{len(entries)}] {subject[:50]}...", end=" ", flush=True)

        t0 = time.time()
        hanna_result = await generate_draft(question, subject)
        duration = time.time() - t0

        if hanna_result.get("skip"):
            print(f"SKIP ({hanna_result.get('skip_reason', '')})")
            results.append({"subject": subject, "status": "SKIP", "semantic_sim": 0})
            continue

        hanna_text = _html_to_text(hanna_result.get("body_html", ""))
        sem_sim = await _semantic_sim(hanna_text, expected)
        txt_sim = _text_sim(hanna_text, expected)
        combined = sem_sim * 0.7 + txt_sim * 0.3

        # Style score
        from app.reasoning.style_score import compute_style_score
        style = compute_style_score(hanna_text, expected)

        status = "PASS" if combined >= 0.5 else "PARTIAL" if combined >= 0.3 else "FAIL"
        print(f"{status} (sem={sem_sim:.2f}, style={style['overall']:.2f}, {duration:.1f}s)")

        # Check for section references
        import re
        expected_sections = set(re.findall(r'(\d+\.\d+\.?\s*pont)', expected))
        hanna_sections = set(re.findall(r'(\d+\.\d+\.?\s*pont)', hanna_text))
        section_match = bool(expected_sections & hanna_sections) if expected_sections else True

        results.append({
            "subject": subject[:80],
            "question": question[:200],
            "expected": expected[:300],
            "hanna": hanna_text[:300],
            "semantic_sim": round(sem_sim, 3),
            "text_sim": round(txt_sim, 3),
            "combined": round(combined, 3),
            "style_score": style["overall"],
            "status": status,
            "confidence": hanna_result.get("confidence", "?"),
            "section_match": section_match,
            "expected_sections": list(expected_sections),
            "hanna_sections": list(hanna_sections),
            "duration_s": round(duration, 1),
        })

        await asyncio.sleep(0.3)

    # Summary
    print(f"\n{'='*60}")
    tested = [r for r in results if r["status"] != "SKIP"]
    print(f"Results: {len(tested)} tested, {len(results) - len(tested)} skipped")

    if tested:
        pass_count = sum(1 for r in tested if r["status"] == "PASS")
        partial_count = sum(1 for r in tested if r["status"] == "PARTIAL")
        fail_count = sum(1 for r in tested if r["status"] == "FAIL")
        avg_sem = sum(r["semantic_sim"] for r in tested) / len(tested)
        avg_style = sum(r["style_score"] for r in tested) / len(tested)
        section_match_rate = sum(1 for r in tested if r["section_match"]) / len(tested)

        print(f"PASS: {pass_count} ({pass_count/len(tested):.0%})")
        print(f"PARTIAL: {partial_count} ({partial_count/len(tested):.0%})")
        print(f"FAIL: {fail_count} ({fail_count/len(tested):.0%})")
        print(f"Avg semantic: {avg_sem:.3f}")
        print(f"Avg style: {avg_style:.3f}")
        print(f"Section reference match: {section_match_rate:.0%}")

        # Show FAILs
        fails = [r for r in tested if r["status"] == "FAIL"]
        if fails:
            print(f"\nFAILs:")
            for f in fails:
                print(f"  {f['subject'][:50]} (sem={f['semantic_sim']:.2f})")
                print(f"    Expected: {f['expected'][:100]}...")
                print(f"    Hanna: {f['hanna'][:100]}...")

    # Save
    report = {
        "eval_date": datetime.now(timezone.utc).isoformat(),
        "total": len(entries),
        "tested": len(tested),
        "pass_rate": round(pass_count / len(tested), 3) if tested else 0,
        "avg_semantic": round(avg_sem, 3) if tested else 0,
        "avg_style": round(avg_style, 3) if tested else 0,
        "section_match_rate": round(section_match_rate, 3) if tested else 0,
        "details": results,
    }

    json_path = DATA_DIR / "eval_golden_set_results.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nJSON: {json_path}")


if __name__ == "__main__":
    asyncio.run(run_golden_eval())
