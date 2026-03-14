#!/usr/bin/env python3
"""F1.1 — Obsidian KG Cleanup Script

Cleans up the obsidian_rag Knowledge Graph:
1. Remove dummy/test entities (Teszt Elek, Bruce Wayne, etc.)
2. Remove file-extension entities (.jpg, .pdf, .docx, etc.)
3. Remove numeric-only and hashtag-only entities
4. Remove path-like concept entities (containing /)
5. Fix person → org misclassifications (Minisztérium, Alapítvány, etc.)
6. Fix org → law misclassifications (EU irányelvek)
7. Remove very long org/person names (>60 chars, likely noise)
8. Merge case-insensitive duplicates (keep lower ID)

Usage:
    python obsidian_kg_cleanup.py --dry-run    # Preview changes
    python obsidian_kg_cleanup.py              # Execute cleanup
"""

import argparse
import asyncio
import sys

import asyncpg

PG_DSN = "postgresql://klara:klara_docs_2026@localhost:5433/obsidian_rag"

# ── Known dummy/test entities ────────────────────────────────────────
DUMMY_NAMES = {
    "Teszt Elek",
    "Bruce Wayne",
    "Clark Kent",
    "Vezetéknév Keresztnév",
    "John Doe",
    "Jane Doe",
    "Test User",
    "Lorem Ipsum",
}

# ── Org indicator patterns (for person → org reclassification) ───────
ORG_INDICATORS = (
    "minisztérium", "hivatal", "bizottság", "alapítvány",
    "zrt.", "kft.", "bt.", "nyrt.", "kht.",
    "intézet", "hatóság", "ügynökség", "szövetség",
    "kamara", "társaság", "szolgálat", "központ",
    "bíróság", "ügyészség", "rendőrség", "igazgatóság",
    "nonprofit", "szervezet", "tanács", "testület",
)

# ── File extension pattern ───────────────────────────────────────────
FILE_EXT_SUFFIXES = (
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".bmp",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".pptx", ".ppt",
    ".csv", ".zip", ".mp4", ".mp3", ".wav", ".mov",
)


async def run_cleanup(dry_run: bool = True):
    pool = await asyncpg.create_pool(PG_DSN, min_size=1, max_size=3)

    stats = {
        "dummy_deleted": 0,
        "file_ext_deleted": 0,
        "numeric_hash_deleted": 0,
        "path_concept_deleted": 0,
        "long_name_deleted": 0,
        "person_to_org": 0,
        "org_to_law": 0,
        "duplicates_merged": 0,
    }

    async with pool.acquire() as conn:
        # Get totals before
        total_before = await conn.fetchval("SELECT count(*) FROM kg_entities")
        relations_before = await conn.fetchval("SELECT count(*) FROM kg_relations")
        echunks_before = await conn.fetchval("SELECT count(*) FROM kg_entity_chunks")

        print(f"=== Obsidian KG Cleanup {'(DRY RUN)' if dry_run else '(LIVE)'} ===")
        print(f"Before: {total_before} entities, {relations_before} relations, {echunks_before} entity-chunks\n")

        # ── 1. Dummy/test entities ───────────────────────────────────
        dummy_ids = await conn.fetch(
            "SELECT id, name, type FROM kg_entities WHERE name = ANY($1::text[])",
            list(DUMMY_NAMES),
        )
        stats["dummy_deleted"] = len(dummy_ids)
        if dummy_ids:
            print(f"[1] Dummy entities to delete: {len(dummy_ids)}")
            for r in dummy_ids:
                print(f"    - {r['name']} ({r['type']}, id={r['id']})")
            if not dry_run:
                ids = [r["id"] for r in dummy_ids]
                await _delete_entities(conn, ids)

        # ── 2. File extension entities ───────────────────────────────
        file_ext_rows = await conn.fetch(
            """SELECT id, name, type FROM kg_entities
               WHERE lower(name) ~ '\\.(jpg|jpeg|png|gif|svg|webp|bmp|pdf|doc|docx|xls|xlsx|pptx|ppt|csv|zip|mp4|mp3|wav|mov)$'"""
        )
        stats["file_ext_deleted"] = len(file_ext_rows)
        if file_ext_rows:
            print(f"\n[2] File-extension entities to delete: {len(file_ext_rows)}")
            for r in file_ext_rows[:10]:
                print(f"    - {r['name'][:80]} ({r['type']})")
            if len(file_ext_rows) > 10:
                print(f"    ... and {len(file_ext_rows) - 10} more")
            if not dry_run:
                ids = [r["id"] for r in file_ext_rows]
                await _delete_entities(conn, ids)

        # ── 3. Numeric-only, hashtag-only, very short entities ───────
        noise_rows = await conn.fetch(
            """SELECT id, name, type FROM kg_entities
               WHERE name ~ '^[0-9_/\\-\\.\\s]+$'
                  OR (name ~ '^#' AND length(name) <= 3)
                  OR length(name) <= 1"""
        )
        stats["numeric_hash_deleted"] = len(noise_rows)
        if noise_rows:
            print(f"\n[3] Numeric/hashtag/tiny entities to delete: {len(noise_rows)}")
            for r in noise_rows[:15]:
                print(f"    - '{r['name']}' ({r['type']})")
            if len(noise_rows) > 15:
                print(f"    ... and {len(noise_rows) - 15} more")
            if not dry_run:
                ids = [r["id"] for r in noise_rows]
                await _delete_entities(conn, ids)

        # ── 4. Path-like concept entities (file paths as names) ──────
        path_rows = await conn.fetch(
            """SELECT id, name, type FROM kg_entities
               WHERE type = 'concept'
                 AND (name ~ '/' AND name ~ '\\.')
                 AND name !~ '^\\d+/\\d+'"""  # don't match law references like 272/2014
        )
        stats["path_concept_deleted"] = len(path_rows)
        if path_rows:
            print(f"\n[4] Path-like concept entities to delete: {len(path_rows)}")
            for r in path_rows[:10]:
                print(f"    - {r['name'][:80]} ({r['type']})")
            if len(path_rows) > 10:
                print(f"    ... and {len(path_rows) - 10} more")
            if not dry_run:
                ids = [r["id"] for r in path_rows]
                await _delete_entities(conn, ids)

        # ── 5. Long noisy names (org/person > 60 chars) ─────────────
        #    Exclude entities that step 7 would reclassify to law
        long_rows = await conn.fetch(
            """SELECT id, name, type FROM kg_entities
               WHERE type IN ('org', 'person')
                 AND length(name) > 60
                 AND NOT (lower(name) ~ '(irányelv|rendelet|szabályoz|törvény|határozat|directive|regulation)'
                          OR name ~ '^\\(EU\\)')"""
        )
        stats["long_name_deleted"] = len(long_rows)
        if long_rows:
            print(f"\n[5] Long noisy org/person names to delete: {len(long_rows)}")
            for r in long_rows:
                print(f"    - {r['name'][:80]}... ({r['type']})")
            if not dry_run:
                ids = [r["id"] for r in long_rows]
                await _delete_entities(conn, ids)

        # ── 6. Person → Org reclassification ────────────────────────
        person_to_org_rows = await conn.fetch(
            """SELECT id, name FROM kg_entities
               WHERE type = 'person'
                 AND (""" +
            " OR ".join(f"lower(name) LIKE '%{ind}%'" for ind in ORG_INDICATORS) +
            ")"
        )
        stats["person_to_org"] = len(person_to_org_rows)
        if person_to_org_rows:
            print(f"\n[6] Person → Org reclassification: {len(person_to_org_rows)}")
            for r in person_to_org_rows:
                print(f"    - {r['name']} → org")
            if not dry_run:
                await _reclassify_entities(conn, person_to_org_rows, "org")

        # ── 7. Org → Law reclassification (EU directives/regulations) ─
        org_to_law_rows = await conn.fetch(
            """SELECT id, name FROM kg_entities
               WHERE type = 'org'
                 AND (lower(name) ~ '(irányelv|rendelet|szabályoz|törvény|határozat|directive|regulation)'
                      OR name ~ '^\\(EU\\)')"""
        )
        stats["org_to_law"] = len(org_to_law_rows)
        if org_to_law_rows:
            print(f"\n[7] Org → Law reclassification: {len(org_to_law_rows)}")
            for r in org_to_law_rows:
                print(f"    - {r['name']} → law")
            if not dry_run:
                await _reclassify_entities(conn, org_to_law_rows, "law")

        # ── 8. Case-insensitive duplicate merge ─────────────────────
        dupes = await conn.fetch(
            """SELECT lower(name) as lname, type, count(*) as cnt,
                      array_agg(id ORDER BY id) as ids
               FROM kg_entities
               GROUP BY lower(name), type
               HAVING count(*) > 1"""
        )
        merge_count = 0
        if dupes:
            print(f"\n[8] Case-insensitive duplicates to merge: {len(dupes)} groups")
            for d in dupes:
                keep_id = d["ids"][0]  # keep lowest ID
                remove_ids = d["ids"][1:]
                merge_count += len(remove_ids)
                print(f"    - '{d['lname']}' ({d['type']}): keep id={keep_id}, merge {remove_ids}")
                if not dry_run:
                    for rid in remove_ids:
                        await _merge_entity(conn, keep_id, rid)
                    # Delete the merged entities
                    await conn.execute(
                        "DELETE FROM kg_entities WHERE id = ANY($1::int[])", remove_ids
                    )
        stats["duplicates_merged"] = merge_count

        # ── Summary ──────────────────────────────────────────────────
        print("\n" + "=" * 50)
        print("SUMMARY:")
        print(f"  Dummy entities deleted:      {stats['dummy_deleted']}")
        print(f"  File-extension deleted:       {stats['file_ext_deleted']}")
        print(f"  Numeric/hash/tiny deleted:    {stats['numeric_hash_deleted']}")
        print(f"  Path-concept deleted:          {stats['path_concept_deleted']}")
        print(f"  Long noisy names deleted:      {stats['long_name_deleted']}")
        print(f"  Person → Org reclassified:     {stats['person_to_org']}")
        print(f"  Org → Law reclassified:        {stats['org_to_law']}")
        print(f"  Duplicates merged (removed):   {stats['duplicates_merged']}")
        total_removed = (
            stats["dummy_deleted"]
            + stats["file_ext_deleted"]
            + stats["numeric_hash_deleted"]
            + stats["path_concept_deleted"]
            + stats["long_name_deleted"]
            + stats["duplicates_merged"]
        )
        print(f"  TOTAL entities removed:        {total_removed}")
        print(f"  Type reclassifications:        {stats['person_to_org'] + stats['org_to_law']}")

        if not dry_run:
            total_after = await conn.fetchval("SELECT count(*) FROM kg_entities")
            relations_after = await conn.fetchval("SELECT count(*) FROM kg_relations")
            echunks_after = await conn.fetchval("SELECT count(*) FROM kg_entity_chunks")
            print(f"\nAfter: {total_after} entities, {relations_after} relations, {echunks_after} entity-chunks")
            print(f"Delta: {total_after - total_before} entities, {relations_after - relations_before} relations")
        else:
            print(f"\n(DRY RUN — no changes made. Run without --dry-run to execute.)")

    await pool.close()


async def _delete_entities(conn: asyncpg.Connection, ids: list[int]):
    """Delete entities and cascade to relations and entity-chunks."""
    # CASCADE handles this via FK constraints, but let's be explicit
    await conn.execute("DELETE FROM kg_entity_chunks WHERE entity_id = ANY($1::int[])", ids)
    await conn.execute(
        "DELETE FROM kg_relations WHERE source_id = ANY($1::int[]) OR target_id = ANY($1::int[])", ids
    )
    await conn.execute("DELETE FROM kg_entities WHERE id = ANY($1::int[])", ids)


async def _merge_entity(conn: asyncpg.Connection, keep_id: int, remove_id: int):
    """Merge remove_id entity into keep_id: reassign relations and chunks safely."""
    # Delete relations that would become duplicates after reassignment
    await conn.execute(
        """DELETE FROM kg_relations
           WHERE source_id = $1
             AND (target_id, relation_type) IN (
                 SELECT target_id, relation_type FROM kg_relations WHERE source_id = $2
             )""",
        remove_id, keep_id,
    )
    await conn.execute(
        """DELETE FROM kg_relations
           WHERE target_id = $1
             AND (source_id, relation_type) IN (
                 SELECT source_id, relation_type FROM kg_relations WHERE target_id = $2
             )""",
        remove_id, keep_id,
    )
    # Also delete self-referential relations that would result from merge
    await conn.execute(
        "DELETE FROM kg_relations WHERE source_id = $1 AND target_id = $2", remove_id, keep_id
    )
    await conn.execute(
        "DELETE FROM kg_relations WHERE source_id = $2 AND target_id = $1", keep_id, remove_id
    )
    # Now safely reassign remaining relations
    await conn.execute(
        "UPDATE kg_relations SET source_id = $1 WHERE source_id = $2", keep_id, remove_id
    )
    await conn.execute(
        "UPDATE kg_relations SET target_id = $1 WHERE target_id = $2", keep_id, remove_id
    )
    # Reassign entity-chunks (ignore conflicts)
    await conn.execute(
        """INSERT INTO kg_entity_chunks (entity_id, chunk_id)
           SELECT $1, chunk_id FROM kg_entity_chunks WHERE entity_id = $2
           ON CONFLICT (entity_id, chunk_id) DO NOTHING""",
        keep_id, remove_id,
    )
    await conn.execute("DELETE FROM kg_entity_chunks WHERE entity_id = $1", remove_id)


async def _reclassify_entities(conn: asyncpg.Connection, rows: list, new_type: str):
    """Reclassify entities to a new type; merge if target (name, type) already exists."""
    for r in rows:
        existing = await conn.fetchrow(
            "SELECT id FROM kg_entities WHERE name = $1 AND type = $2",
            r["name"], new_type,
        )
        if existing:
            keep_id = existing["id"]
            src_id = r["id"]
            await _merge_entity(conn, keep_id, src_id)
            await conn.execute("DELETE FROM kg_entities WHERE id = $1", src_id)
            print(f"      (merged id={src_id} into existing id={keep_id})")
        else:
            await conn.execute(
                "UPDATE kg_entities SET type = $1, updated_at = now() WHERE id = $2",
                new_type, r["id"],
            )


def main():
    parser = argparse.ArgumentParser(description="Obsidian KG Cleanup")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without executing")
    args = parser.parse_args()
    asyncio.run(run_cleanup(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
