"""Knowledge Graph entity & relation extraction for Obsidian vault.

Three extraction methods (layered):
1. Wikilink parsing — deterministic, fast, high confidence
2. YAML frontmatter — structured metadata, high confidence
3. LLM-based NER — free-text extraction, lower confidence

All methods produce normalized Entity/Relation dicts that are
upserted into kg_entities / kg_relations / kg_entity_chunks.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

import asyncpg

from .enrichment import _get_client

PG_DSN = "postgresql://klara:klara_docs_2026@host.docker.internal:5433/obsidian_rag"

KG_MODEL = os.environ.get("KG_MODEL", "gpt-4o-mini")
KG_BATCH_SIZE = int(os.environ.get("KG_BATCH_SIZE", "5"))

_pool: asyncpg.Pool | None = None


async def _get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(PG_DSN, min_size=2, max_size=5)
    return _pool


# ── Wikilink parser ──────────────────────────────────────────────────

WIKILINK_RE = re.compile(r"\[\[([^\]|]+?)(?:\|[^\]]+?)?\]\]")

# Heuristic type detection based on naming conventions
ENTITY_TYPE_PATTERNS = [
    (re.compile(r"^§"), "project"),         # §ENAIRGY → project
    (re.compile(r"^\d{4}[./]\d"), "law"),   # 2012/C, 272/2014 → law
    (re.compile(r"^\d+/\d+"), "law"),       # 651/2014 → law
    (re.compile(r"(?:Kft|Ltd|Zrt|Bt|Nyrt|GmbH|Inc)\b", re.I), "org"),
    (re.compile(r"(?:NEÜ|SZTNH|ÉMI|MTA|NKFI|EU|EC)\b"), "org"),
]


def _guess_entity_type(name: str) -> str:
    """Guess entity type from naming conventions."""
    for pattern, etype in ENTITY_TYPE_PATTERNS:
        if pattern.search(name):
            return etype
    # If it looks like a person name (2-3 capitalized words)
    parts = name.strip().split()
    if 2 <= len(parts) <= 4 and all(p[0].isupper() for p in parts if p):
        return "person"
    return "concept"


def extract_wikilinks(content: str, file_path: str) -> list[dict]:
    """Extract wikilink-based entities from markdown content."""
    matches = WIKILINK_RE.findall(content)
    entities = []
    seen = set()
    for match in matches:
        name = match.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        etype = _guess_entity_type(name)
        entities.append({
            "name": name,
            "type": etype,
            "source_file": file_path,
            "mention_type": "wikilink",
            "confidence": 1.0,
        })
    return entities


def extract_wikilink_relations(content: str, file_path: str) -> list[dict]:
    """Extract LINKS_TO relations from wikilinks (note → note)."""
    matches = WIKILINK_RE.findall(content)
    # The source is the file itself
    source_name = Path(file_path).stem
    if source_name.startswith("§"):
        source_type = "project"
    else:
        source_type = "concept"

    relations = []
    seen = set()
    for match in matches:
        target = match.strip()
        if not target or target == source_name or target in seen:
            continue
        seen.add(target)
        relations.append({
            "source_name": source_name,
            "source_type": source_type,
            "target_name": target,
            "target_type": _guess_entity_type(target),
            "relation_type": "LINKS_TO",
            "source_file": file_path,
            "confidence": 1.0,
        })
    return relations


# ── YAML frontmatter parser ─────────────────────────────────────────

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)


def _parse_frontmatter(content: str) -> dict:
    """Simple YAML frontmatter parser (no PyYAML dependency)."""
    m = FRONTMATTER_RE.match(content)
    if not m:
        return {}
    try:
        import yaml
        return yaml.safe_load(m.group(1)) or {}
    except ImportError:
        # Fallback: basic key-value parsing
        result = {}
        for line in m.group(1).split("\n"):
            line = line.strip()
            if ":" in line and not line.startswith("-"):
                key, _, val = line.partition(":")
                result[key.strip()] = val.strip()
        return result


def extract_frontmatter_entities(content: str, file_path: str) -> tuple[list[dict], list[dict]]:
    """Extract entities and relations from YAML frontmatter.

    Returns (entities, relations).
    """
    fm = _parse_frontmatter(content)
    if not fm:
        return [], []

    entities = []
    relations = []
    file_stem = Path(file_path).stem
    file_type = "project" if file_stem.startswith("§") else "concept"

    # People field
    people = fm.get("People") or fm.get("people") or []
    if isinstance(people, str):
        people = [people]
    for p in people:
        name = WIKILINK_RE.sub(r"\1", str(p)).strip()
        if name:
            entities.append({
                "name": name,
                "type": "person",
                "source_file": file_path,
                "mention_type": "frontmatter",
                "confidence": 1.0,
            })
            relations.append({
                "source_name": name,
                "source_type": "person",
                "target_name": file_stem,
                "target_type": file_type,
                "relation_type": "WORKS_ON" if file_type == "project" else "MENTIONED_IN",
                "source_file": file_path,
                "confidence": 1.0,
            })

    # Tags
    tags = fm.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",")]
    for tag in tags:
        tag = str(tag).strip()
        if tag:
            entities.append({
                "name": tag,
                "type": "tag",
                "source_file": file_path,
                "mention_type": "frontmatter",
                "confidence": 1.0,
            })

    # Program (e.g., HORIZON Europe)
    program = fm.get("program")
    if program:
        entities.append({
            "name": str(program),
            "type": "org",
            "source_file": file_path,
            "mention_type": "frontmatter",
            "confidence": 1.0,
        })
        relations.append({
            "source_name": file_stem,
            "source_type": file_type,
            "target_name": str(program),
            "target_type": "org",
            "relation_type": "BELONGS_TO",
            "source_file": file_path,
            "confidence": 1.0,
        })

    # Coordinator
    coordinator = fm.get("coordinator")
    if coordinator:
        coord_name = WIKILINK_RE.sub(r"\1", str(coordinator)).strip()
        # Try to split "Org (Person)" pattern
        coord_match = re.match(r"(.+?)\s*\((.+?)\)", coord_name)
        if coord_match:
            org_name = coord_match.group(1).strip()
            person_name = coord_match.group(2).strip()
            entities.append({"name": org_name, "type": "org", "source_file": file_path,
                             "mention_type": "frontmatter", "confidence": 1.0})
            entities.append({"name": person_name, "type": "person", "source_file": file_path,
                             "mention_type": "frontmatter", "confidence": 1.0})
            relations.append({
                "source_name": file_stem, "source_type": file_type,
                "target_name": org_name, "target_type": "org",
                "relation_type": "COORDINATED_BY",
                "source_file": file_path, "confidence": 1.0,
            })
        else:
            entities.append({"name": coord_name, "type": "org", "source_file": file_path,
                             "mention_type": "frontmatter", "confidence": 1.0})

    # Status
    status = fm.get("status")
    if status and file_type == "project":
        relations.append({
            "source_name": file_stem, "source_type": file_type,
            "target_name": str(status), "target_type": "concept",
            "relation_type": "HAS_STATUS",
            "source_file": file_path, "confidence": 1.0,
        })

    # Meeting
    meetings = fm.get("meeting") or []
    if isinstance(meetings, str):
        meetings = [meetings]
    for mtg in meetings:
        mtg = str(mtg).strip()
        if mtg:
            entities.append({"name": mtg, "type": "concept", "source_file": file_path,
                             "mention_type": "frontmatter", "confidence": 1.0})

    return entities, relations


# ── LLM-based NER ────────────────────────────────────────────────────

LLM_NER_PROMPT = """Te egy Obsidian tudásbázis Knowledge Graph kinyerő rendszer vagy.

A bemenet egy Obsidian jegyzetből származó szövegrészlet. Kinyered a benne lévő entitásokat és relációkat.

## Entitástípusok:
- person: Személy (pl. "Szuper József", "Czerovszki Noémi")
- project: Projekt (pl. "ENAIRGY", "SMART4BUILD")
- org: Szervezet (pl. "NEÜ", "Enasco Ltd.", "EU")
- law: Jogszabály (pl. "272/2014 EU rendelet", "Ptk.")
- concept: Fogalom/téma (pl. "energiahatékonyság", "AI-driven forecasting")

## Relációtípusok:
- WORKS_ON(person, project)
- BELONGS_TO(project, org/program)
- COORDINATED_BY(project, org)
- REFERENCES(note, law)
- MET_WITH(person, person)
- MANAGES(person, project)
- MEMBER_OF(person, org)
- PARTNER_IN(org, project)
- FUNDED_BY(project, org)

## Válaszformátum (STRICT JSON):
{
  "entities": [
    {"name": "...", "type": "person|project|org|law|concept", "aliases": []}
  ],
  "relations": [
    {"source": "entity_name", "target": "entity_name", "type": "RELATION_TYPE"}
  ]
}

Szabályok:
- CSAK a szövegben TÉNYLEGESEN megjelenő entitásokat add vissza
- Legyél specifikus, kerüld a túl általános fogalmakat
- Magyar nevek magyar formában (Vezetéknév Keresztnév)
- CSAK VALID JSON-t adj vissza!"""


def extract_llm_entities(
    chunks: list[dict],
    file_name: str,
    file_path: str,
) -> tuple[list[dict], list[dict]]:
    """Use LLM to extract entities and relations from chunks.

    Returns (entities, relations).
    """
    client = _get_client()
    all_entities = []
    all_relations = []

    # Process in batches
    for i in range(0, len(chunks), KG_BATCH_SIZE):
        batch = chunks[i:i + KG_BATCH_SIZE]
        combined = "\n\n---\n\n".join(
            c.get("content", c.get("text", ""))[:2000] for c in batch
        )
        user_msg = f"Fájl: {file_name}\n\nSzövegrészletek:\n\n{combined}"

        try:
            resp = client.chat.completions.create(
                model=KG_MODEL,
                messages=[
                    {"role": "system", "content": LLM_NER_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=2000,
            )
            result = json.loads(resp.choices[0].message.content)

            for ent in result.get("entities", []):
                all_entities.append({
                    "name": ent["name"],
                    "type": ent.get("type", "concept"),
                    "aliases": ent.get("aliases", []),
                    "source_file": file_path,
                    "mention_type": "llm_ner",
                    "confidence": 0.8,
                })

            for rel in result.get("relations", []):
                all_relations.append({
                    "source_name": rel["source"],
                    "source_type": "concept",  # will be resolved on upsert
                    "target_name": rel["target"],
                    "target_type": "concept",
                    "relation_type": rel.get("type", "RELATED_TO"),
                    "source_file": file_path,
                    "confidence": 0.8,
                })

            time.sleep(0.2)  # Rate limiting

        except Exception as e:
            print(f"[kg-extract] LLM NER failed for batch {i}: {e}")

    return all_entities, all_relations


# ── DB upsert ────────────────────────────────────────────────────────

async def upsert_entity(
    pool: asyncpg.Pool,
    name: str,
    etype: str,
    aliases: list[str] | None = None,
    metadata: dict | None = None,
    source_file: str | None = None,
) -> int:
    """Upsert entity, return entity ID."""
    row = await pool.fetchrow(
        """INSERT INTO kg_entities (name, type, aliases, metadata, source_file)
           VALUES ($1, $2, $3, $4, $5)
           ON CONFLICT (name, type) DO UPDATE
           SET aliases = (
               SELECT array_agg(DISTINCT a)
               FROM unnest(kg_entities.aliases || EXCLUDED.aliases) a
           ),
           metadata = kg_entities.metadata || EXCLUDED.metadata,
           updated_at = now()
           RETURNING id""",
        name, etype,
        aliases or [],
        json.dumps(metadata or {}),
        source_file,
    )
    return row["id"]


async def upsert_relation(
    pool: asyncpg.Pool,
    source_id: int,
    target_id: int,
    relation_type: str,
    metadata: dict | None = None,
    source_chunk_id: str | None = None,
    source_file: str | None = None,
    confidence: float = 1.0,
) -> int:
    """Upsert relation, return relation ID."""
    row = await pool.fetchrow(
        """INSERT INTO kg_relations (source_id, target_id, relation_type, metadata,
                                     source_chunk_id, source_file, confidence)
           VALUES ($1, $2, $3, $4, $5, $6, $7)
           ON CONFLICT (source_id, target_id, relation_type) DO UPDATE
           SET metadata = kg_relations.metadata || EXCLUDED.metadata,
               confidence = GREATEST(kg_relations.confidence, EXCLUDED.confidence),
               source_file = COALESCE(EXCLUDED.source_file, kg_relations.source_file)
           RETURNING id""",
        source_id, target_id, relation_type,
        json.dumps(metadata or {}),
        source_chunk_id, source_file, confidence,
    )
    return row["id"]


async def link_entity_chunk(
    pool: asyncpg.Pool,
    entity_id: int,
    chunk_id: str,
    file_path: str,
    mention_type: str = "wikilink",
):
    """Link an entity to a chunk."""
    await pool.execute(
        """INSERT INTO kg_entity_chunks (entity_id, chunk_id, file_path, mention_type)
           VALUES ($1, $2, $3, $4)
           ON CONFLICT (entity_id, chunk_id) DO NOTHING""",
        entity_id, chunk_id, file_path, mention_type,
    )


# ── Main extraction pipeline ────────────────────────────────────────

async def extract_file_kg(
    content: str,
    file_path: str,
    chunks: list[dict] | None = None,
    use_llm: bool = False,
) -> dict[str, Any]:
    """Full KG extraction pipeline for a single file.

    1. Wikilink parsing (deterministic)
    2. YAML frontmatter parsing (deterministic)
    3. LLM NER (optional, for rich extraction)

    Returns stats dict.
    """
    pool = await _get_pool()
    file_name = Path(file_path).name
    stats = {"entities": 0, "relations": 0, "file": file_path}

    all_entities = []
    all_relations = []

    # Layer 1: Wikilinks
    wl_entities = extract_wikilinks(content, file_path)
    wl_relations = extract_wikilink_relations(content, file_path)
    all_entities.extend(wl_entities)
    all_relations.extend(wl_relations)

    # Layer 2: Frontmatter
    fm_entities, fm_relations = extract_frontmatter_entities(content, file_path)
    all_entities.extend(fm_entities)
    all_relations.extend(fm_relations)

    # Layer 3: LLM NER (optional)
    if use_llm and chunks:
        llm_entities, llm_relations = extract_llm_entities(chunks, file_name, file_path)
        all_entities.extend(llm_entities)
        all_relations.extend(llm_relations)

    # Deduplicate entities by (name, type)
    entity_map: dict[tuple[str, str], dict] = {}
    for e in all_entities:
        key = (e["name"], e["type"])
        if key not in entity_map:
            entity_map[key] = e
        else:
            # Merge: keep higher confidence
            existing = entity_map[key]
            if e.get("confidence", 1.0) > existing.get("confidence", 1.0):
                entity_map[key] = e

    # Upsert entities
    entity_ids: dict[tuple[str, str], int] = {}
    for key, e in entity_map.items():
        eid = await upsert_entity(
            pool, e["name"], e["type"],
            aliases=e.get("aliases", []),
            source_file=e.get("source_file"),
        )
        entity_ids[key] = eid
        stats["entities"] += 1

    # Upsert relations
    for r in all_relations:
        src_key = (r["source_name"], r["source_type"])
        tgt_key = (r["target_name"], r["target_type"])

        # Ensure both entities exist
        if src_key not in entity_ids:
            eid = await upsert_entity(pool, r["source_name"], r["source_type"],
                                       source_file=r.get("source_file"))
            entity_ids[src_key] = eid
        if tgt_key not in entity_ids:
            eid = await upsert_entity(pool, r["target_name"], r["target_type"],
                                       source_file=r.get("source_file"))
            entity_ids[tgt_key] = eid

        try:
            await upsert_relation(
                pool,
                entity_ids[src_key],
                entity_ids[tgt_key],
                r["relation_type"],
                source_file=r.get("source_file"),
                confidence=r.get("confidence", 1.0),
            )
            stats["relations"] += 1
        except Exception as e:
            print(f"[kg-extract] Relation upsert failed: {e}")

    # Link entities to chunks (if chunks provided)
    if chunks:
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", chunk.get("id", ""))
            chunk_content = chunk.get("content", chunk.get("text", ""))
            if not chunk_id:
                continue
            # Find which entities appear in this chunk
            for (name, etype), eid in entity_ids.items():
                if name in chunk_content:
                    await link_entity_chunk(
                        pool, eid, chunk_id, file_path,
                        mention_type="chunk_match",
                    )

    return stats


async def extract_vault_kg(
    vault_path: str,
    use_llm: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run KG extraction across the entire vault.

    Reads files and extracts entities/relations.
    """
    from .pg_ingest import scan_vault_files

    vault = Path(vault_path)
    files = scan_vault_files(vault_path)
    if limit:
        files = files[:limit]

    total_stats = {"files": 0, "entities": 0, "relations": 0, "errors": []}

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            total_stats["errors"].append(f"Read failed: {file_path}: {e}")
            continue

        if not content.strip():
            continue

        relative_path = str(file_path.relative_to(vault))
        try:
            stats = await extract_file_kg(content, relative_path, use_llm=use_llm)
            total_stats["files"] += 1
            total_stats["entities"] += stats["entities"]
            total_stats["relations"] += stats["relations"]

            if total_stats["files"] % 100 == 0:
                print(f"[kg-extract] Progress: {total_stats['files']} files, "
                      f"{total_stats['entities']} entities, {total_stats['relations']} relations")
        except Exception as e:
            total_stats["errors"].append(f"Extract failed: {relative_path}: {e}")

    print(f"[kg-extract] Done: {total_stats['files']} files, "
          f"{total_stats['entities']} entities, {total_stats['relations']} relations, "
          f"{len(total_stats['errors'])} errors")

    return total_stats
