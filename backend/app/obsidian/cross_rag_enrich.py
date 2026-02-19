"""Cross-RAG Entity Enrichment: Klára NEÜ docs KG → Obsidian People/Companies.

When new documents are ingested into Klára's KG, this module:
1. Queries newly extracted person/org entities from Klára KG
2. Matches them against Obsidian People/Companies files
3. Updates existing files or creates new ones as needed

Designed to be called as a hook after Klára ingest, or via API endpoint.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import asyncpg

# Klára NEÜ docs DB
KLARA_DSN = "postgresql://klara:klara_docs_2026@host.docker.internal:5433/neu_docs"

# Obsidian vault path (inside Docker container)
VAULT_PATH = Path("/app/obsidian-vault")
PEOPLE_DIR = VAULT_PATH / "3_resources" / "People"
COMPANIES_DIR = VAULT_PATH / "3_resources" / "Companies"

INTERNAL_KEYWORDS = [
    "nffkü", "neü", "vital management", "vital intermed", "snomed",
    "promenis", "ghs", "ierp", "orient river", "zebra consulting", "vasc",
]

AUTO_SECTION = "## Dokumentum alapú adatok (automatikus)"

_klara_pool: asyncpg.Pool | None = None


async def _get_klara_pool() -> asyncpg.Pool:
    global _klara_pool
    if _klara_pool is None:
        _klara_pool = await asyncpg.create_pool(KLARA_DSN, min_size=1, max_size=3)
    return _klara_pool


def _is_internal(name: str) -> bool:
    nl = name.lower()
    return any(kw in nl for kw in INTERNAL_KEYWORDS)


def _normalize_name(name: str) -> str:
    n = re.sub(r'\s+', ' ', name.strip())
    n = re.sub(r'^(dr\.|Dr\.|DR\.|Mr\.?|Mrs\.?|Prof\.?)\s*', '', n)
    for s in [' Zrt.', ' Zrt', ' Kft.', ' Kft', ' Bt.', ' Bt',
              ' Ltd.', ' Ltd', ' GmbH', ' Nyrt.', ' Nyrt']:
        if n.endswith(s):
            n = n[:-len(s)].strip()
    return n


def _is_valid_person_name(name: str) -> bool:
    """Check if name looks like a real person name."""
    parts = name.strip().split()
    if len(parts) < 2 or len(parts) > 5:
        return False
    if any(kw in name.lower() for kw in ['kft', 'zrt', 'bt.', 'ltd', 'iroda', 'étterem']):
        return False
    if name.startswith('HR_') or name.startswith('OCR-') or '@' in name:
        return False
    return all(p[0].isupper() for p in parts if len(p) > 1)


def _is_valid_org_name(name: str) -> bool:
    """Check if name looks like a real org/company name."""
    if len(name) < 3 or len(name) > 100:
        return False
    alnum = sum(1 for c in name if c.isalnum())
    if alnum < len(name) * 0.4:
        return False
    if name.startswith('HR_') or name.startswith('OCR-') or '@' in name:
        return False
    return True


def _match_file(name: str, files: dict[str, Path]) -> str | None:
    """Try to match entity name to existing file."""
    norm = _normalize_name(name).lower()
    for fname in files:
        if fname.lower() == norm or fname.lower() == name.lower():
            return fname
        if SequenceMatcher(None, norm, fname.lower()).ratio() >= 0.85:
            return fname
    return None


def _build_auto_section(entity_id: int, name: str, relations: list[dict],
                         entity_type: str = "person", internal: bool = False) -> str:
    """Build the automatic data section for an Obsidian file."""
    lines = [f"\n\n{AUTO_SECTION}\n"]
    lines.append(f"*KG entity ID: {entity_id} | Klára név: {name} | Frissítve: {datetime.now().strftime('%Y-%m-%d')}*\n")
    if internal:
        lines.append("**Belső** (NEÜ/NFFKÜ csoport)\n")

    if relations:
        rel_groups: dict[str, list[str]] = {}
        for r in relations:
            rt = r["relation_type"]
            other = r["other_name"]
            rel_groups.setdefault(rt, []).append(other)

        lines.append("\n### Kapcsolatok (KG)\n")
        for rt, targets in sorted(rel_groups.items()):
            lines.append(f"**{rt}:** {len(targets)}")
            for t in targets[:5]:
                lines.append(f"  - {t}")
            if len(targets) > 5:
                lines.append(f"  - ... és {len(targets)-5} további")
            lines.append("")

    return "\n".join(lines)


async def enrich_from_klara(
    since_hours: int = 24,
    min_relations: int = 2,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Cross-RAG enrichment: check Klára KG for entities and sync to Obsidian vault.

    Args:
        since_hours: Look at entities updated in the last N hours (0 = all)
        min_relations: Minimum relation count to create new files
        dry_run: If True, don't write files, just return what would happen
    """
    pool = await _get_klara_pool()

    # Get entities (optionally filtered by recent activity)
    if since_hours > 0:
        entities = await pool.fetch("""
            SELECT DISTINCT e.id, e.name, e.entity_type, e.properties
            FROM kg_entities e
            JOIN kg_relations r ON r.source_id = e.id OR r.target_id = e.id
            WHERE e.entity_type IN ('person', 'organization', 'Partner')
            ORDER BY e.name
        """)
        # Note: Klára KG doesn't have updated_at on entities, so we process all
        # In future, add tracking of which entities were already synced
    else:
        entities = await pool.fetch("""
            SELECT id, name, entity_type, properties
            FROM kg_entities
            WHERE entity_type IN ('person', 'organization', 'Partner')
            ORDER BY name
        """)

    # Load existing Obsidian files
    people_files = {f.stem: f for f in PEOPLE_DIR.glob("*.md")} if PEOPLE_DIR.exists() else {}
    company_files = {f.stem: f for f in COMPANIES_DIR.glob("*.md")} if COMPANIES_DIR.exists() else {}

    stats = {
        "entities_checked": len(entities),
        "people_enriched": 0,
        "people_created": 0,
        "companies_enriched": 0,
        "companies_created": 0,
        "skipped": 0,
        "dry_run": dry_run,
        "details": [],
    }

    for entity in entities:
        eid = entity["id"]
        name = entity["name"]
        etype = entity["entity_type"]

        # Get relation count
        rel_count = await pool.fetchval(
            "SELECT count(*) FROM kg_relations WHERE source_id=$1 OR target_id=$1", eid
        )

        if etype == "person":
            if not _is_valid_person_name(name):
                stats["skipped"] += 1
                continue

            match = _match_file(name, people_files)
            if match:
                # Enrich existing file
                fpath = people_files[match]
                content = fpath.read_text(encoding="utf-8")
                if AUTO_SECTION in content:
                    continue  # Already enriched

                rels = await pool.fetch("""
                    SELECT r.relation_type, e2.name as other_name
                    FROM kg_relations r JOIN kg_entities e2
                    ON CASE WHEN r.source_id=$1 THEN r.target_id ELSE r.source_id END = e2.id
                    WHERE r.source_id=$1 OR r.target_id=$1
                """, eid)

                section = _build_auto_section(eid, name, [dict(r) for r in rels])
                if not dry_run:
                    fpath.write_text(content.rstrip() + "\n" + section + "\n", encoding="utf-8")
                stats["people_enriched"] += 1
                stats["details"].append({"action": "enrich", "type": "person", "name": name, "file": match})

            elif rel_count >= min_relations and _is_valid_person_name(name):
                # Create new person file
                if not dry_run:
                    safe_name = re.sub(r'[<>"/\\|?*:]', '', name).strip()
                    fpath = PEOPLE_DIR / f"{safe_name}.md"
                    if not fpath.exists():
                        roles = await pool.fetch("""
                            SELECT DISTINCT r.relation_type FROM kg_relations r
                            WHERE r.source_id=$1 OR r.target_id=$1
                        """, eid)
                        role_names = [r["relation_type"] for r in roles]
                        is_employee = any(r in role_names for r in ["munkavállaló", "munkáltató"])

                        content = f"""---
tags:
  - Person
type:
  - Contact
category:
  - People
status: active
Munkahely: {"NEÜ Zrt." if is_employee else ""}
pozíció:
Email:
Telefon:
projects: {"NEÜ" if is_employee else ""}
---

# {name}
"""
                        rels = await pool.fetch("""
                            SELECT r.relation_type, e2.name as other_name
                            FROM kg_relations r JOIN kg_entities e2
                            ON CASE WHEN r.source_id=$1 THEN r.target_id ELSE r.source_id END = e2.id
                            WHERE r.source_id=$1 OR r.target_id=$1
                        """, eid)
                        section = _build_auto_section(eid, name, [dict(r) for r in rels])
                        fpath.write_text(content + section + "\n", encoding="utf-8")

                stats["people_created"] += 1
                stats["details"].append({"action": "create", "type": "person", "name": name})

        elif etype in ("organization", "Partner"):
            if not _is_valid_org_name(name):
                stats["skipped"] += 1
                continue

            match = _match_file(name, company_files)
            if match:
                fpath = company_files[match]
                content = fpath.read_text(encoding="utf-8")
                if AUTO_SECTION in content:
                    continue

                rels = await pool.fetch("""
                    SELECT r.relation_type, e2.name as other_name
                    FROM kg_relations r JOIN kg_entities e2
                    ON CASE WHEN r.source_id=$1 THEN r.target_id ELSE r.source_id END = e2.id
                    WHERE r.source_id=$1 OR r.target_id=$1
                """, eid)

                internal = _is_internal(name)
                section = _build_auto_section(eid, name, [dict(r) for r in rels],
                                               entity_type="organization", internal=internal)
                if not dry_run:
                    fpath.write_text(content.rstrip() + "\n" + section + "\n", encoding="utf-8")
                stats["companies_enriched"] += 1
                stats["details"].append({"action": "enrich", "type": "org", "name": name, "file": match})

            elif rel_count >= min_relations:
                if not dry_run:
                    safe_name = re.sub(r'[<>"/\\|?*:]', '', name).strip()
                    if len(safe_name) < 3 or len(safe_name) > 100:
                        stats["skipped"] += 1
                        continue
                    fpath = COMPANIES_DIR / f"{safe_name}.md"
                    if not fpath.exists():
                        internal = _is_internal(name)
                        content = f"""---
tags:
  - Company
{"  - Internal" if internal else ""}
status: active
category: Company
type: {"internal" if internal else "private"}
website:
---

# {name}

## Alapadatok

- **Típus:** {"Belső cég (NEÜ/NFFKÜ csoport)" if internal else ""}
- **Székhely:**
- **Szektor:**

## Kapcsolódó személyek

## Jegyzetek
"""
                        rels = await pool.fetch("""
                            SELECT r.relation_type, e2.name as other_name
                            FROM kg_relations r JOIN kg_entities e2
                            ON CASE WHEN r.source_id=$1 THEN r.target_id ELSE r.source_id END = e2.id
                            WHERE r.source_id=$1 OR r.target_id=$1
                        """, eid)
                        section = _build_auto_section(eid, name, [dict(r) for r in rels],
                                                       entity_type="organization", internal=internal)
                        fpath.write_text(content + section + "\n", encoding="utf-8")

                stats["companies_created"] += 1
                stats["details"].append({"action": "create", "type": "org", "name": name})
        else:
            stats["skipped"] += 1

    return stats
