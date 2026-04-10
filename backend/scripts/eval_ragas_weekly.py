#!/usr/bin/env python3
"""RAGAS weekly evaluation — batch quality check on production logs.

Runs Faithfulness + Context Precision + Answer Relevancy metrics
on the last N drafts from the draft store. Generates Obsidian report.

Usage:
    cd /Users/varadiimre/DEV/customercare/backend
    ../.venv-eval/bin/python scripts/eval_ragas_weekly.py [--limit 30] [--report]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

CC_URL = os.getenv("CC_URL", "http://localhost:8101")
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
OBSIDIAN_REPORTS = Path.home() / "Library/Mobile Documents/iCloud~md~obsidian/Documents/PARA/!inbox/!reports"


def _html_to_text(html: str) -> str:
    if not html:
        return ""
    return BeautifulSoup(html, "html.parser").get_text(separator="\n", strip=True)


def _search(query: str, top_k: int = 5) -> list[str]:
    """Get retrieval context for a query."""
    r = httpx.post(f"{CC_URL}/search", json={"query": query, "top_k": top_k}, timeout=30)
    if r.status_code != 200:
        return []
    return [res.get("text", "")[:500] for res in r.json().get("results", [])[:5]]


def _generate_and_eval(question: str, subject: str = "") -> dict | None:
    """Generate draft and collect eval data."""
    try:
        r = httpx.post(f"{CC_URL}/draft/generate", json={
            "email_text": question[:3000],
            "email_subject": subject,
            "sender_name": "Eval Pályázó",
            "sender_email": "eval@test.hu",
            "top_k": 5,
        }, timeout=120)
        if r.status_code != 200:
            return None
        result = r.json()
        if result.get("skip"):
            return None
        return result
    except Exception:
        return None


def load_recent_questions(limit: int = 30) -> list[dict]:
    """Load recent questions from eval results or draft store."""
    # Try eval_live_results first
    eval_path = DATA_DIR / "eval_live_results.json"
    if eval_path.exists():
        data = json.loads(eval_path.read_text())
        entries = []
        for r in data.get("results", []):
            if r.get("status") in ("MATCH", "PARTIAL") and r.get("question_preview"):
                entries.append({
                    "question": r["question_preview"],
                    "subject": r.get("subject", ""),
                    "expected": r.get("answer_preview", ""),
                })
        if entries:
            return entries[:limit]

    # Fallback: golden set
    gs_path = DATA_DIR / "golden_set_eval_partials.json"
    if gs_path.exists():
        gs = json.loads(gs_path.read_text())
        return [{"question": e["question"], "subject": e.get("subject", ""), "expected": e.get("expected_answer", "")} for e in gs[:limit]]

    return []


def compute_faithfulness_manual(draft_text: str, context: list[str]) -> float:
    """Simple faithfulness score: what fraction of draft sentences are supported by context."""
    import re
    if not draft_text or not context:
        return 0.0

    context_text = " ".join(context).lower()
    sentences = [s.strip() for s in re.split(r"[.!?]\n", draft_text) if len(s.strip()) > 20]
    # Skip greeting and closing
    content = [s for s in sentences if "tisztelt" not in s.lower() and "üdvözlettel" not in s.lower() and "kollégánk" not in s.lower()]

    if not content:
        return 1.0  # Only greeting/closing → technically faithful

    supported = 0
    for sent in content:
        words = set(sent.lower().split())
        # Check word overlap with context (rough approximation)
        context_words = set(context_text.split())
        overlap = len(words & context_words) / len(words) if words else 0
        if overlap >= 0.3:  # At least 30% word overlap → likely supported
            supported += 1

    return supported / len(content) if content else 1.0


def compute_semantic_sim(text_a: str, text_b: str) -> float:
    """Embedding cosine similarity via CC BGE-M3."""
    if not text_a or not text_b:
        return 0.0
    try:
        r = httpx.post("http://localhost:8104/embed", json={"texts": [text_a[:500], text_b[:500]]}, timeout=10)
        if r.status_code != 200:
            return 0.0
        embs = r.json().get("embeddings", [])
        if len(embs) != 2:
            return 0.0
        dot = sum(a * b for a, b in zip(embs[0], embs[1]))
        na = math.sqrt(sum(x * x for x in embs[0]))
        nb = math.sqrt(sum(x * x for x in embs[1]))
        return dot / (na * nb) if na and nb else 0.0
    except Exception:
        return 0.0


def run_ragas_eval(limit: int = 30, generate_report: bool = False):
    """Run RAGAS-style batch evaluation."""
    questions = load_recent_questions(limit)
    if not questions:
        print("[ragas] No questions found for evaluation")
        return

    print(f"[ragas] Evaluating {len(questions)} questions...")

    results = []
    for i, q in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] {q['subject'][:40]}...", end=" ", flush=True)

        draft_result = _generate_and_eval(q["question"], q["subject"])
        if not draft_result:
            print("SKIP")
            continue

        draft_text = _html_to_text(draft_result.get("body_html", ""))
        context = _search(q["question"])

        # Metrics
        faithfulness = compute_faithfulness_manual(draft_text, context)
        relevancy = compute_semantic_sim(draft_text, q["question"]) if q["question"] else 0
        answer_sim = compute_semantic_sim(draft_text, q["expected"]) if q.get("expected") else 0

        # Context precision: how many retrieved chunks are actually used
        citations = draft_result.get("citations", {})
        ctx_precision = len(citations) / len(context) if context else 0

        confidence = draft_result.get("confidence", "?")
        guardrails = draft_result.get("guardrails")
        guardrail_pass = guardrails is None  # None = no warnings

        results.append({
            "subject": q["subject"][:60],
            "faithfulness": round(faithfulness, 3),
            "relevancy": round(relevancy, 3),
            "answer_similarity": round(answer_sim, 3),
            "context_precision": round(ctx_precision, 3),
            "confidence": confidence,
            "guardrail_pass": guardrail_pass,
        })

        status = "OK" if faithfulness >= 0.5 else "LOW"
        print(f"{status} (faith={faithfulness:.2f} rel={relevancy:.2f} conf={confidence})")

        import time
        time.sleep(1)

    if not results:
        print("[ragas] No results")
        return

    # Aggregate
    avg = lambda key: round(sum(r[key] for r in results) / len(results), 3)
    stats = {
        "eval_date": datetime.now(timezone.utc).isoformat(),
        "total": len(results),
        "avg_faithfulness": avg("faithfulness"),
        "avg_relevancy": avg("relevancy"),
        "avg_answer_similarity": avg("answer_similarity"),
        "avg_context_precision": avg("context_precision"),
        "guardrail_pass_rate": round(sum(1 for r in results if r["guardrail_pass"]) / len(results), 3),
        "confidence_dist": {},
    }
    for r in results:
        stats["confidence_dist"][r["confidence"]] = stats["confidence_dist"].get(r["confidence"], 0) + 1

    print(f"\n{'='*60}")
    print(f"  RAGAS WEEKLY EVAL")
    print(f"{'='*60}")
    print(f"  Tesztelve:           {stats['total']}")
    print(f"  Avg faithfulness:    {stats['avg_faithfulness']}")
    print(f"  Avg relevancy:       {stats['avg_relevancy']}")
    print(f"  Avg answer sim:      {stats['avg_answer_similarity']}")
    print(f"  Avg context prec:    {stats['avg_context_precision']}")
    print(f"  Guardrail pass:      {stats['guardrail_pass_rate']:.0%}")
    print(f"  Confidence:          {stats['confidence_dist']}")

    # Save
    out_path = DATA_DIR / "eval_ragas_weekly.json"
    out_path.write_text(json.dumps({"stats": stats, "results": results}, ensure_ascii=False, indent=2))
    print(f"\n[ragas] JSON → {out_path}")

    # Obsidian report
    if generate_report:
        _write_report(stats, results)


def _write_report(stats: dict, results: list[dict]):
    now = datetime.now()
    filename = f"{now.strftime('%y%m%d')}-cc-ragas-eval.md"

    low_faith = [r for r in results if r["faithfulness"] < 0.5]

    md = f"""# CustomerCare RAGAS Eval — {now.strftime('%Y-%m-%d %H:%M')}

## Összefoglaló

| Metrika | Érték |
|---------|-------|
| Tesztelve | {stats['total']} |
| Faithfulness (átlag) | {stats['avg_faithfulness']} |
| Relevancy (átlag) | {stats['avg_relevancy']} |
| Answer similarity | {stats['avg_answer_similarity']} |
| Context precision | {stats['avg_context_precision']} |
| Guardrail pass | {stats['guardrail_pass_rate']:.0%} |
| Confidence | {stats['confidence_dist']} |
"""

    if low_faith:
        md += "\n## Alacsony faithfulness (< 0.5)\n\n"
        for r in low_faith:
            md += f"- **{r['subject']}** — faith={r['faithfulness']}, conf={r['confidence']}\n"

    try:
        OBSIDIAN_REPORTS.mkdir(parents=True, exist_ok=True)
        (OBSIDIAN_REPORTS / filename).write_text(md, encoding="utf-8")
        print(f"[ragas] Report → {OBSIDIAN_REPORTS / filename}")
    except Exception as e:
        print(f"[ragas] Report write failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    run_ragas_eval(limit=args.limit, generate_report=args.report)
