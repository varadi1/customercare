#!/usr/bin/env python3
"""Hanna — Unified Ingest Verification & Status Tracking.

Hanna manages 3 collections:
  1. Obsidian RAG — vault markdown files (/obsidian/* endpoints)
  2. OETP/email docs — emails, PDFs, segédletek (/stats, /search endpoints)
  3. Cross-RAG — entity linking across systems (/cross-rag/* endpoints)

This script verifies all 3 collections.

Usage:
    python3 scripts/verify_ingest.py              # Report only
    python3 scripts/verify_ingest.py --fix        # Trigger re-sync if stale
    python3 scripts/verify_ingest.py --report     # Generate Obsidian report
    python3 scripts/verify_ingest.py --json       # JSON output
    python3 scripts/verify_ingest.py --fix --report  # Full run
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone, timedelta
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
        def post(self, url, timeout=10, json=None):
            import urllib.request as ur
            data = None
            headers = {}
            if json is not None:
                import json as j
                data = j.dumps(json).encode()
                headers["Content-Type"] = "application/json"
            req = ur.Request(url, data=data, headers=headers, method="POST")
            with ur.urlopen(req, timeout=timeout) as r:
                return self._Resp(r.read(), r.status)
    requests = _Requests()

API = "http://localhost:8101"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HASHES_FILE = DATA_DIR / "obsidian_hashes.json"
VAULT_DIR = DATA_DIR / "obsidian-vault"
LAST_SYNC_FILE = DATA_DIR / "obsidian_last_sync.json"
REPORT_DIR = os.path.expanduser(
    "~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Para/!inbox/!reports"
)

STALE_HOURS = 24


def ts():
    return datetime.now().strftime("%H:%M:%S")


def count_vault_files(vault_dir):
    """Count .md files in Obsidian vault."""
    if not vault_dir.exists():
        return 0, {}
    counts_by_folder = {}
    total = 0
    for md_file in vault_dir.rglob("*.md"):
        total += 1
        # Get top-level folder
        rel = md_file.relative_to(vault_dir)
        top = rel.parts[0] if len(rel.parts) > 1 else "(root)"
        counts_by_folder[top] = counts_by_folder.get(top, 0) + 1
    return total, counts_by_folder


def get_api_health():
    """Check API health endpoint."""
    try:
        resp = requests.get(f"{API}/health", timeout=10)
        return resp.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_obsidian_stats():
    """Get stats from obsidian API."""
    try:
        resp = requests.get(f"{API}/obsidian/stats", timeout=10)
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


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


def get_last_sync():
    """Get last sync info from API and local file."""
    api_sync = None
    try:
        resp = requests.get(f"{API}/obsidian/last-sync", timeout=5)
        api_sync = resp.json()
    except Exception:
        pass

    file_sync = None
    if LAST_SYNC_FILE.exists():
        try:
            with open(LAST_SYNC_FILE) as f:
                file_sync = json.load(f)
        except Exception:
            pass

    return api_sync, file_sync


def get_sync_hashes():
    """Get hash tracking state."""
    if not HASHES_FILE.exists():
        return {}
    try:
        with open(HASHES_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def test_search():
    """Test search functionality."""
    try:
        resp = requests.get(f"{API}/obsidian/search/hybrid?q=test&top_k=1", timeout=10)
        data = resp.json()
        results = data.get("results", [])
        return len(results) > 0, len(results)
    except Exception as e:
        return False, str(e)


def verify_all():
    """Run full verification."""
    issues = []

    # 1. API health
    health = get_api_health()
    api_ok = health.get("status") == "ok"
    chromadb_ok = health.get("chromadb") == "connected"
    chromadb_count = health.get("collection_count", 0)
    if not api_ok:
        issues.append("API health check failed")
    if not chromadb_ok:
        issues.append("ChromaDB not connected")

    # 2. Obsidian stats (pgvector)
    stats = get_obsidian_stats()
    total_chunks_pg = stats.get("total_chunks", 0)
    total_files_pg = stats.get("total_files", 0)
    folders_pg = stats.get("folders", {})
    if "error" in stats:
        issues.append(f"Obsidian stats error: {stats['error']}")

    # 3. Vault reachability
    vault_exists = VAULT_DIR.exists()
    vault_files, vault_folders = count_vault_files(VAULT_DIR)
    if not vault_exists:
        issues.append(f"Vault not found at {VAULT_DIR}")
    elif vault_files == 0:
        issues.append("Vault exists but contains 0 .md files")

    # 4. Last sync
    api_sync, file_sync = get_last_sync()
    sync_timestamp = None
    sync_stale = False
    if api_sync and api_sync.get("timestamp"):
        try:
            sync_ts_str = api_sync["timestamp"]
            # Parse ISO format
            if sync_ts_str.endswith("Z"):
                sync_ts_str = sync_ts_str[:-1] + "+00:00"
            sync_timestamp = datetime.fromisoformat(sync_ts_str)
            age = datetime.now(timezone.utc) - sync_timestamp
            sync_stale = age > timedelta(hours=STALE_HOURS)
            if sync_stale:
                issues.append(f"Last sync is stale: {age.days}d {age.seconds // 3600}h ago")
        except Exception:
            pass

    sync_errors = []
    if api_sync:
        sync_errors = api_sync.get("errors", [])
        if sync_errors:
            issues.append(f"Sync had {len(sync_errors)} errors")

    # 5. Hash tracking
    hashes = get_sync_hashes()
    tracked_files = len(hashes)

    # 6. File coverage check
    coverage_pct = 0.0
    if vault_files > 0 and total_files_pg > 0:
        coverage_pct = total_files_pg / vault_files * 100
        if coverage_pct < 80:
            issues.append(f"Low coverage: {coverage_pct:.1f}% ({total_files_pg}/{vault_files})")

    # 7. Chunk ratio check
    chunks_per_file = total_chunks_pg / total_files_pg if total_files_pg > 0 else 0
    if total_files_pg > 0 and chunks_per_file < 1:
        issues.append(f"Low chunk ratio: {chunks_per_file:.1f} chunks/file")

    # 8. Zero-chunk critical check
    if total_chunks_pg == 0 and total_files_pg > 0:
        issues.append("CRITICAL: Files in DB but 0 chunks - embedding likely failed")

    if total_files_pg == 0 and vault_files > 0:
        issues.append("CRITICAL: Vault has files but DB has 0 - ingest not run?")

    # 9. Search test
    search_ok, search_detail = test_search()
    if not search_ok:
        issues.append(f"Search test failed: {search_detail}")

    # 10. OETP/email collection
    oetp_stats = get_oetp_stats()
    oetp_chunks = oetp_stats.get("total_chunks", 0)
    oetp_doc_types = oetp_stats.get("doc_types", {})
    oetp_programs = oetp_stats.get("programs", {})
    oetp_kg = oetp_stats.get("knowledge_graph", {})
    if "error" in oetp_stats:
        issues.append(f"OETP stats error: {oetp_stats['error']}")
    elif oetp_chunks == 0:
        issues.append("OETP collection has 0 chunks")

    # 11. Cross-RAG
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
        "chromadb_ok": chromadb_ok,
        "chromadb_collection_count": chromadb_count,
        # Obsidian collection
        "vault_exists": vault_exists,
        "vault_files": vault_files,
        "vault_folders": vault_folders,
        "db_files": total_files_pg,
        "db_chunks": total_chunks_pg,
        "db_folders": folders_pg,
        "chunks_per_file": round(chunks_per_file, 1),
        "coverage_pct": round(coverage_pct, 1),
        "tracked_hashes": tracked_files,
        "last_sync": sync_timestamp.isoformat() if sync_timestamp else None,
        "sync_stale": sync_stale,
        "sync_errors": sync_errors,
        "search_ok": search_ok,
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
        "  - hanna",
        "  - obsidian",
        "  - oetp",
        "  - rag",
        "  - verify",
        f"date: {date_str}",
        "---",
        "",
        f"# Hanna — Unified Ingest verifikáció {date_str} {time_str}",
        "",
        "> [!info] Összefoglaló",
        f"> Státusz: **{summary['status'].upper()}**",
        f"> Obsidian: {summary['vault_files']} vault files → {summary['db_files']} DB files, {summary['db_chunks']:,} chunks",
        f"> OETP/Email: {summary['oetp_chunks']:,} chunks, {summary['oetp_kg_entities']} KG entities",
        f"> Cross-RAG: {summary['crossrag_entities']} entities, {summary['crossrag_links']} links",
        f"> Last sync: {summary['last_sync'] or 'unknown'} {'(STALE)' if summary['sync_stale'] else ''}",
        "",
    ]

    if summary["issues"]:
        lines.append("## Problemas")
        lines.append("")
        for issue in summary["issues"]:
            if "CRITICAL" in issue:
                lines.append(f"- **{issue}**")
            else:
                lines.append(f"- {issue}")
        lines.append("")

    # Folder breakdown
    lines.append("## Folder reszletezese")
    lines.append("")
    lines.append("### DB (pgvector)")
    lines.append("")
    lines.append("| Folder | Chunks |")
    lines.append("|--------|--------|")
    for folder, count in sorted(summary["db_folders"].items(), key=lambda x: -x[1]):
        lines.append(f"| {folder} | {count:,} |")
    lines.append("")

    if summary["vault_folders"]:
        lines.append("### Vault (disk)")
        lines.append("")
        lines.append("| Folder | Files |")
        lines.append("|--------|-------|")
        for folder, count in sorted(summary["vault_folders"].items(), key=lambda x: -x[1]):
            lines.append(f"| {folder} | {count:,} |")
        lines.append("")

    # Sync errors
    if summary["sync_errors"]:
        lines.append("## Sync hibak")
        lines.append("")
        for err in summary["sync_errors"][:20]:
            lines.append(f"- `{err}`")
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
        f"- Search: {'OK' if summary['search_ok'] else 'Failed'}",
        f"- Tracked hashes: {summary['tracked_hashes']}",
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
    parser = argparse.ArgumentParser(description="Verify Hanna (Obsidian RAG) ingest")
    parser.add_argument("--fix", action="store_true", help="Trigger re-sync if stale")
    parser.add_argument("--report", action="store_true", help="Generate Obsidian report")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    if not args.json:
        print(f"[{ts()}] Hanna — Unified Ingest Verification")
        print(f"{'=' * 60}")

    summary = verify_all()

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"\n  API: {'OK' if summary['api_ok'] else 'DOWN'}")
        print(f"\n  === Obsidian RAG ===")
        print(f"  Vault: {summary['vault_files']} .md files")
        print(f"  DB: {summary['db_files']} files, {summary['db_chunks']:,} chunks")
        print(f"  Coverage: {summary['coverage_pct']}% | Chunks/file: {summary['chunks_per_file']}")
        print(f"  Last sync: {summary['last_sync'] or 'unknown'} {'(STALE!)' if summary['sync_stale'] else ''}")
        print(f"\n  === OETP/Email ===")
        print(f"  Chunks: {summary['oetp_chunks']:,}")
        if summary['oetp_doc_types']:
            top_types = sorted(summary['oetp_doc_types'].items(), key=lambda x: -x[1])[:4]
            print(f"  Types: {', '.join(f'{t}={c}' for t,c in top_types)}")
        print(f"  KG: {summary['oetp_kg_entities']} entities, {summary['oetp_kg_relations']} relations")
        print(f"\n  === Cross-RAG ===")
        print(f"  Entities: {summary['crossrag_entities']}, Links: {summary['crossrag_links']}, Multi-DB: {summary['crossrag_multi_db']}")
        print(f"\n  Search: {'OK' if summary['search_ok'] else 'FAILED'}")

        if summary["issues"]:
            print(f"\n  Issues ({len(summary['issues'])}):")
            for issue in summary["issues"]:
                print(f"    - {issue}")

        print(f"\n  Status: {summary['status'].upper()}")

    if args.fix and summary["sync_stale"]:
        print(f"\n[{ts()}] Triggering re-sync (last sync is stale)...")
        try:
            resp = requests.post(f"{API}/obsidian/ingest", timeout=10)
            print(f"  Sync triggered: {resp.json()}")
        except Exception as e:
            print(f"  Sync failed: {e}")

    if args.report:
        date_str = datetime.now().strftime("%Y-%m-%d")
        report_path = os.path.join(REPORT_DIR, f"hanna-ingest-verify-{date_str}.md")
        generate_report(summary, report_path)

    return 0 if summary["status"] != "error" else 1


if __name__ == "__main__":
    sys.exit(main())
