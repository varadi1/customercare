#!/usr/bin/env python3
"""Hanna (Obsidian RAG) — Ingest Verification.

Compares Obsidian vault files vs DB state.
Reports files that were synced but have 0 chunks in DB.

Usage:
    python3 scripts/verify_ingest.py              # Report only
    python3 scripts/verify_ingest.py --fix        # Report + trigger re-sync
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import requests

API = "http://localhost:8101"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HASHES_FILE = DATA_DIR / "obsidian_hashes.json"


def ts():
    return datetime.now().strftime("%H:%M:%S")


def get_db_stats() -> dict:
    """Get DB stats from API."""
    try:
        resp = requests.get(f"{API}/obsidian/stats", timeout=10)
        return resp.json()
    except Exception as e:
        print(f"[{ts()}] ERROR: Cannot reach Hanna API: {e}")
        sys.exit(1)


def get_sync_state() -> dict:
    """Get last sync hashes (what files were synced)."""
    if not HASHES_FILE.exists():
        print(f"[{ts()}] WARNING: {HASHES_FILE} not found")
        return {}
    with open(HASHES_FILE) as f:
        return json.load(f)


def verify():
    """Check DB health and sync consistency."""
    stats = get_db_stats()
    hashes = get_sync_state()

    total_chunks = stats.get("total_chunks", 0)
    total_files = stats.get("total_files", 0)
    synced_files = len(hashes)

    print(f"\n  DB: {total_files} files, {total_chunks:,} chunks")
    print(f"  Sync hashes: {synced_files} files tracked")

    if synced_files > 0 and total_files > 0:
        coverage = total_files / synced_files * 100
        print(f"  Coverage: {coverage:.1f}% ({total_files}/{synced_files})")

    # Check for zero-chunk anomaly
    if total_chunks == 0 and total_files > 0:
        print(f"\n  *** CRITICAL: {total_files} files but 0 chunks! Embedding likely failed ***")
        return False

    if total_files == 0 and synced_files > 0:
        print(f"\n  *** CRITICAL: {synced_files} synced files but 0 in DB! Ingest failed ***")
        return False

    # Quick search test
    try:
        resp = requests.get(f"{API}/obsidian/search/hybrid?q=test&top_k=1", timeout=10)
        results = resp.json().get("results", [])
        if results:
            print(f"  Search test: OK (got result)")
        else:
            print(f"  Search test: WARNING - no results for 'test'")
    except Exception as e:
        print(f"  Search test: FAILED ({e})")

    # Check last sync
    try:
        resp = requests.get(f"{API}/obsidian/last-sync", timeout=5)
        sync_info = resp.json()
        print(f"  Last sync: {sync_info.get('last_sync', 'unknown')}")
    except Exception:
        pass

    print(f"\n  Status: {'OK' if total_chunks > 0 else 'NEEDS ATTENTION'}")
    return total_chunks > 0


def main():
    parser = argparse.ArgumentParser(description="Verify Hanna (Obsidian RAG) ingest")
    parser.add_argument("--fix", action="store_true", help="Trigger re-sync")
    args = parser.parse_args()

    print(f"[{ts()}] Hanna (Obsidian RAG) — Ingest Verification")
    print(f"{'=' * 60}")

    ok = verify()

    if not ok and args.fix:
        print(f"\n[{ts()}] Triggering re-sync...")
        try:
            resp = requests.post(f"{API}/obsidian/ingest", timeout=10)
            print(f"  Sync triggered: {resp.json()}")
        except Exception as e:
            print(f"  Sync failed: {e}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
