"""Cross-RAG Entity Enrichment: NEÜ Docs ↔ Obsidian bidirectional sync.

Uses the cross_rag database to find entities that exist in both NEÜ Docs
and Obsidian, then enriches Obsidian People/Companies notes with NEÜ Docs
relationship data (contracts, partners, legal references).

Two modes:
1. enrich_from_crossrag() — Uses cross_rag DB matches (high precision)
2. discover_new_matches() — Finds NEÜ Docs entities that SHOULD be in Obsidian
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import asyncpg

# Cross-RAG module
sys.path.insert(0, os.environ.get("CROSSRAG_SCRIPTS", "/app/crossrag_scripts"))
from cross_rag_sync import CROSSRAG_DSN, get_crossrag_pool

# NEÜ Docs DB
KLARA_DSN = "postgresql://klara:klara_docs_2026@host.docker.internal:5433/neu_docs"

# Obsidian vault path (inside Docker container)
VAULT_PATH = Path("/app/obsidian-vault")
PEOPLE_DIR = VAULT_PATH / "3_resources" / "People"
COMPANIES_DIR = VAULT_PATH / "3_resources" / "Companies"

AUTO_SECTION = "## Dokumentum alapú adatok (automatikus)"

INTERNAL_KEYWORDS = [
    "nffkü", "neü", "vital management", "vital intermed", "snomed",
    "promenis", "ghs", "ierp", "orient river", "zebra consulting", "vasc",
    "nffku", "neu zrt",
]

_klara_pool: asyncpg.Pool | None = None


async def _get_klara_pool() -> asyncpg.Pool:
    global _klara_pool
    if _klara_pool is None:
        _klara_pool = await asyncpg.create_pool(KLARA_DSN, min_size=1, max_size=3)
    return _klara_pool


def _is_internal(name: str) -> bool:
    nl = name.lower()
    return any(kw in nl for kw in INTERNAL_KEYWORDS)


def _is_valid_person_name(name: str) -> bool:
    parts = name.strip().split()
    if len(parts) < 2 or len(parts) > 5:
        return False
    if any(kw in name.lower() for kw in ['kft', 'zrt', 'bt.', 'ltd', 'iroda', 'étterem', '@']):
        return False
    if name.startswith('HR_') or name.startswith('OCR-'):
        return False
    return all(p[0].isupper() for p in parts if len(p) > 1)


def _is_valid_org_name(name: str) -> bool:
    if len(name) < 3 or len(name) > 100:
        return False
    # Reject Google Drive IDs, file hashes, OCR noise
    if re.match(r'^[a-zA-Z0-9_-]{20,}$', name):
        return False
    if name.startswith('HR_') or name.startswith('OCR-') or '@' in name:
        return False
    # Reject if it looks like a document title (contains "szerződés", "megbízás", etc.)
    if re.search(r'(szerződés|megbízás|megrendelés|módosítás|tervezet|melléklet|számla|jegyzőkönyv)', name.lower()):
        return False
    # Must have at least some alphabetic chars
    alnum = sum(1 for c in name if c.isalpha())
    return alnum >= max(3, len(name) * 0.4)


def _build_auto_section(
    canonical_id: int,
    canonical_name: str,
    neu_entity_id: int,
    neu_name: str,
    relations: list[dict],
    entity_type: str = "person",
    internal: bool = False,
    chunk_count: int = 0,
) -> str:
    """Build the automatic data section for an Obsidian file."""
    lines = [f"\n\n{AUTO_SECTION}\n"]
    lines.append(
        f"*Cross-RAG ID: {canonical_id} | NEÜ entity: {neu_entity_id} ({neu_name}) | "
        f"Chunks: {chunk_count} | Frissítve: {datetime.now().strftime('%Y-%m-%d')}*\n"
    )
    if internal:
        lines.append("**Belső** (NEÜ/NFFKÜ csoport)\n")

    if relations:
        rel_groups: dict[str, list[str]] = {}
        for r in relations:
            rt = r["relation_type"]
            other = r["other_name"]
            rel_groups.setdefault(rt, []).append(other)

        lines.append("\n### Kapcsolatok (NEÜ Docs KG)\n")
        for rt, targets in sorted(rel_groups.items()):
            lines.append(f"**{rt}:** {len(targets)}")
            for t in targets[:10]:
                lines.append(f"  - {t}")
            if len(targets) > 10:
                lines.append(f"  - ... és {len(targets) - 10} további")
            lines.append("")

    return "\n".join(lines)


async def enrich_from_klara(
    since_hours: int = 0,
    min_relations: int = 2,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Enrich Obsidian People/Companies from NEÜ Docs via cross_rag DB.

    Uses cross_rag entity_links to find precise matches between
    Obsidian and NEÜ Docs entities, then writes relationship data
    into the Obsidian vault files.
    """
    crossrag_pool = await get_crossrag_pool()
    klara_pool = await _get_klara_pool()

    # Load existing Obsidian files
    people_files = {f.stem: f for f in PEOPLE_DIR.glob("*.md")} if PEOPLE_DIR.exists() else {}
    company_files = {f.stem: f for f in COMPANIES_DIR.glob("*.md")} if COMPANIES_DIR.exists() else {}

    stats = {
        "entities_checked": 0,
        "people_enriched": 0,
        "people_created": 0,
        "companies_enriched": 0,
        "companies_created": 0,
        "skipped": 0,
        "dry_run": dry_run,
        "details": [],
    }

    # ── Step 1: Get all obsidian↔neu_docs cross-links ──────────────
    async with crossrag_pool.acquire() as conn:
        cross_links = await conn.fetch("""
            SELECT c.id as canonical_id, c.canonical_name, c.canonical_type,
                   obs.source_entity_id as obs_entity_id,
                   obs.source_entity_name as obs_name,
                   obs.source_entity_type as obs_type,
                   neu.source_entity_id as neu_entity_id,
                   neu.source_entity_name as neu_name,
                   neu.source_entity_type as neu_type,
                   neu.confidence as confidence
            FROM canonical_entities c
            JOIN entity_links obs ON obs.canonical_id = c.id AND obs.source_db = 'obsidian_rag'
            JOIN entity_links neu ON neu.canonical_id = c.id AND neu.source_db = 'neu_docs'
            WHERE c.canonical_type IN ('person', 'org')
            ORDER BY c.canonical_type, c.canonical_name
        """)

    stats["entities_checked"] = len(cross_links)

    # Deduplicate: one Obsidian entity can match multiple NEÜ entities
    # Group by canonical_id, pick the NEÜ entity with most relations
    canonical_groups: dict[int, list] = {}
    for link in cross_links:
        cid = link["canonical_id"]
        canonical_groups.setdefault(cid, []).append(link)

    for cid, links in canonical_groups.items():
        canonical_name = links[0]["canonical_name"]
        canonical_type = links[0]["canonical_type"]
        obs_name = links[0]["obs_name"]

        # Pick the best NEÜ entity (most relations)
        best_neu = None
        best_rel_count = -1
        for link in links:
            neu_eid = int(link["neu_entity_id"])
            async with klara_pool.acquire() as kconn:
                rc = await kconn.fetchval(
                    "SELECT count(*) FROM kg_relations WHERE source_id=$1 OR target_id=$1",
                    neu_eid,
                )
            if rc > best_rel_count:
                best_rel_count = rc
                best_neu = link

        if not best_neu or best_rel_count < min_relations:
            stats["skipped"] += 1
            continue

        neu_eid = int(best_neu["neu_entity_id"])
        neu_name = best_neu["neu_name"]

        # Get NEÜ Docs relations for this entity
        async with klara_pool.acquire() as kconn:
            rels = await kconn.fetch("""
                SELECT r.relation_type, e2.name as other_name
                FROM kg_relations r
                JOIN kg_entities e2 ON CASE WHEN r.source_id=$1 THEN r.target_id ELSE r.source_id END = e2.id
                WHERE r.source_id=$1 OR r.target_id=$1
            """, neu_eid)

            chunk_count = await kconn.fetchval(
                "SELECT count(*) FROM kg_entity_chunks WHERE entity_id = $1", neu_eid
            )

        if canonical_type == "person":
            if not _is_valid_person_name(obs_name):
                stats["skipped"] += 1
                continue

            # Find matching Obsidian file
            match_file = _find_obsidian_file(obs_name, people_files)
            if match_file:
                fpath = people_files[match_file]
                content = fpath.read_text(encoding="utf-8")
                if AUTO_SECTION in content:
                    # Already enriched — update
                    content = _remove_auto_section(content)

                section = _build_auto_section(
                    cid, canonical_name, neu_eid, neu_name,
                    [dict(r) for r in rels], "person",
                    chunk_count=chunk_count,
                )
                if not dry_run:
                    fpath.write_text(content.rstrip() + "\n" + section + "\n", encoding="utf-8")
                stats["people_enriched"] += 1
                stats["details"].append({
                    "action": "enrich", "type": "person",
                    "name": obs_name, "file": match_file,
                    "neu_name": neu_name, "relations": best_rel_count,
                    "chunks": chunk_count,
                })
            else:
                stats["skipped"] += 1

        elif canonical_type == "org":
            if not _is_valid_org_name(obs_name):
                stats["skipped"] += 1
                continue

            match_file = _find_obsidian_file(obs_name, company_files)
            if match_file:
                fpath = company_files[match_file]
                content = fpath.read_text(encoding="utf-8")
                if AUTO_SECTION in content:
                    content = _remove_auto_section(content)

                internal = _is_internal(obs_name)
                section = _build_auto_section(
                    cid, canonical_name, neu_eid, neu_name,
                    [dict(r) for r in rels], "org", internal=internal,
                    chunk_count=chunk_count,
                )
                if not dry_run:
                    fpath.write_text(content.rstrip() + "\n" + section + "\n", encoding="utf-8")
                stats["companies_enriched"] += 1
                stats["details"].append({
                    "action": "enrich", "type": "org",
                    "name": obs_name, "file": match_file,
                    "neu_name": neu_name, "relations": best_rel_count,
                    "chunks": chunk_count,
                })
            else:
                stats["skipped"] += 1

    return stats


def _find_obsidian_file(entity_name: str, files: dict[str, Path]) -> str | None:
    """Match entity name to an Obsidian file name."""
    # Exact match first
    for fname in files:
        if fname == entity_name or fname.lower() == entity_name.lower():
            return fname

    # Normalize: strip company suffixes, titles
    norm = _normalize_for_match(entity_name)
    for fname in files:
        fnorm = _normalize_for_match(fname)
        if fnorm == norm:
            return fname

    return None


def _normalize_for_match(name: str) -> str:
    n = re.sub(r'\s+', ' ', name.strip()).lower()
    n = re.sub(r'^(dr\.?\s*|prof\.?\s*)', '', n)
    for s in [' zrt.', ' zrt', ' kft.', ' kft', ' bt.', ' bt',
              ' ltd.', ' ltd', ' gmbh', ' nyrt.', ' nyrt']:
        if n.endswith(s):
            n = n[:-len(s)].strip()
    return n


def _remove_auto_section(content: str) -> str:
    """Remove existing auto-generated section from content."""
    idx = content.find(AUTO_SECTION)
    if idx < 0:
        return content
    return content[:idx].rstrip()
