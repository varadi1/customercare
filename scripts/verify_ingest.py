#!/usr/bin/env python3
"""CustomerCare — Unified Ingest Verification & Status Tracking.

CustomerCare manages 2 collections:
  1. OETP/email docs — emails, PDFs, segédletek (/stats, /search endpoints)
  2. Cross-RAG — entity linking across systems (/cross-rag/* endpoints)

Note: Obsidian RAG moved to standalone service (obsidian-rag :8115).

Usage:
    python3 scripts/verify_ingest.py              # Report only
    python3 scripts/verify_ingest.py --report     # Generate Obsidian report
    python3 scripts/verify_ingest.py --json       # JSON output
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    import urllib.request

    class _Requests:
        """Minimal requests-like wrapper using urllib."""
        class _Resp:
            def __init__(self, data, status):
                self._data = data
                self.status_code = status
            def json(self):
                return json.loads(self._data)
        def get(self, url, timeout=10):
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return self._Resp(r.read(), r.status)
    requests = _Requests()

API = "http://localhost:8101"
REPORT_DIR = os.path.expanduser(
    "~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Para/!inbox/!reports"
)


def ts():
    return datetime.now().strftime("%H:%M:%S")


def get_api_health():
    """Check API health endpoint."""
    try:
        resp = requests.get(f"{API}/health", timeout=10)
        return resp.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_oetp_stats():
    """Get OETP/email collection stats."""
    try:
        resp = requests.get(f"{API}/stats", timeout=10)
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def get_crossrag_stats():
    """Get cross-RAG stats."""
    try:
        resp = requests.get(f"{API}/cross-rag/stats", timeout=10)
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def verify_all():
    """Run full verification."""
    issues = []

    # 1. API health
    health = get_api_health()
    api_ok = health.get("status") == "ok"
    if not api_ok:
        issues.append("API health check failed")

    # 2. OETP/email collection
    oetp_stats = get_oetp_stats()
    oetp_chunks = oetp_stats.get("total_chunks", 0)
    oetp_doc_types = oetp_stats.get("doc_types", {})
    oetp_programs = oetp_stats.get("programs", {})
    oetp_kg = oetp_stats.get("knowledge_graph", {})
    if "error" in oetp_stats:
        issues.append(f"OETP stats error: {oetp_stats['error']}")
    elif oetp_chunks == 0:
        issues.append("OETP collection has 0 chunks")

    # 3. Cross-RAG
    crossrag_stats = get_crossrag_stats()
    crossrag_entities = 0
    crossrag_links = 0
    crossrag_multi_db = 0
    if "error" not in crossrag_stats:
        crossrag_entities = crossrag_stats.get("canonical_entities", crossrag_stats.get("total_entities", 0))
        crossrag_links = crossrag_stats.get("total_links", 0)
        crossrag_multi_db = crossrag_stats.get("multi_db_entities", 0)
    else:
        issues.append(f"Cross-RAG stats error: {crossrag_stats.get('error','')}")

    # Determine overall status
    critical = any("CRITICAL" in i for i in issues)
    if critical:
        status = "error"
    elif len(issues) > 0:
        status = "warning"
    else:
        status = "ok"

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        # API
        "api_ok": api_ok,
        # OETP/email collection
        "oetp_chunks": oetp_chunks,
        "oetp_doc_types": oetp_doc_types,
        "oetp_programs": oetp_programs,
        "oetp_kg_entities": oetp_kg.get("entities", 0),
        "oetp_kg_relations": oetp_kg.get("relations", 0),
        # Cross-RAG
        "crossrag_entities": crossrag_entities,
        "crossrag_links": crossrag_links,
        "crossrag_multi_db": crossrag_multi_db,
        # Issues
        "issues": issues,
    }

    return summary


def generate_report(summary, output_path=None):
    """Generate Obsidian markdown report."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")

    lines = [
        "---",
        "tags:",
        "  - customercare",
        "  - oetp",
        "  - rag",
        "  - verify",
        f"date: {date_str}",
        "---",
        "",
        f"# CustomerCare — Ingest verifikáció {date_str} {time_str}",
        "",
        "> [!info] Összefoglaló",
        f"> Státusz: **{summary['status'].upper()}**",
        f"> OETP/Email: {summary['oetp_chunks']:,} chunks, {summary['oetp_kg_entities']} KG entities",
        f"> Cross-RAG: {summary['crossrag_entities']} entities, {summary['crossrag_links']} links",
        "",
    ]

    if summary["issues"]:
        lines.append("## Problémák")
        lines.append("")
        for issue in summary["issues"]:
            if "CRITICAL" in issue:
                lines.append(f"- **{issue}**")
            else:
                lines.append(f"- {issue}")
        lines.append("")

    # OETP section
    lines.extend([
        "## OETP/Email gyűjtemény",
        "",
        f"- Chunks: {summary['oetp_chunks']:,}",
    ])
    if summary["oetp_doc_types"]:
        lines.append("- Típusok:")
        for typ, cnt in sorted(summary["oetp_doc_types"].items(), key=lambda x: -x[1]):
            lines.append(f"  - {typ}: {cnt:,}")
    if summary["oetp_programs"]:
        lines.append("- Programok:")
        for prog, cnt in sorted(summary["oetp_programs"].items(), key=lambda x: -x[1]):
            lines.append(f"  - {prog}: {cnt:,}")
    lines.append(f"- KG: {summary['oetp_kg_entities']} entities, {summary['oetp_kg_relations']} relations")
    lines.append("")

    # Cross-RAG section
    lines.extend([
        "## Cross-RAG",
        "",
        f"- Canonical entities: {summary['crossrag_entities']}",
        f"- Links: {summary['crossrag_links']}",
        f"- Multi-DB entities: {summary['crossrag_multi_db']}",
        "",
    ])

    lines.extend([
        "## Rendszer állapot",
        "",
        f"- API: {'OK' if summary['api_ok'] else 'DOWN'}",
        "",
    ])

    report_text = "\n".join(lines)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report_text)
        print(f"[{ts()}] Report: {output_path}")

    return report_text


def main():
    parser = argparse.ArgumentParser(description="Verify CustomerCare ingest (OETP + Cross-RAG)")
    parser.add_argument("--report", action="store_true", help="Generate Obsidian report")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    if not args.json:
        print(f"[{ts()}] CustomerCare — Ingest Verification")
        print(f"{'=' * 60}")

    summary = verify_all()

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"\n  API: {'OK' if summary['api_ok'] else 'DOWN'}")
        print(f"\n  === OETP/Email ===")
        print(f"  Chunks: {summary['oetp_chunks']:,}")
        if summary['oetp_doc_types']:
            top_types = sorted(summary['oetp_doc_types'].items(), key=lambda x: -x[1])[:4]
            print(f"  Types: {', '.join(f'{t}={c}' for t,c in top_types)}")
        print(f"  KG: {summary['oetp_kg_entities']} entities, {summary['oetp_kg_relations']} relations")
        print(f"\n  === Cross-RAG ===")
        print(f"  Entities: {summary['crossrag_entities']}, Links: {summary['crossrag_links']}, Multi-DB: {summary['crossrag_multi_db']}")

        if summary["issues"]:
            print(f"\n  Issues ({len(summary['issues'])}):")
            for issue in summary["issues"]:
                print(f"    - {issue}")

        print(f"\n  Status: {summary['status'].upper()}")

    if args.report:
        date_str = datetime.now().strftime("%Y-%m-%d")
        report_path = os.path.join(REPORT_DIR, f"cc-ingest-verify-{date_str}.md")
        generate_report(summary, report_path)

    return 0 if summary["status"] != "error" else 1


if __name__ == "__main__":
    sys.exit(main())
