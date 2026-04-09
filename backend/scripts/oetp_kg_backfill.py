#!/usr/bin/env python3
"""F1.3 — OETP Entity-Chunk Coverage Backfill

Increases kg_entity_chunks coverage from ~3% to 60%+.

Strategy (SQL-based for performance):
1. Text matching via PostgreSQL ILIKE — fast, done in DB
2. Document propagation: if entity found in any chunk of a doc, link all doc chunks
3. Skip very short/generic entity names to avoid false positives

Current: 932 entities, 9,972 chunks, 2,052 links (306 unique chunks = 3%)
Target: 60%+ chunk coverage

Usage:
    python oetp_kg_backfill.py --dry-run    # Preview
    python oetp_kg_backfill.py              # Execute
"""

import argparse
import asyncio
import time
import sys

import asyncpg

PG_DSN = "postgresql://klara:klara_docs_2026@localhost:5433/customercare"

MIN_NAME_LENGTH = 4

SKIP_NAMES = (
    # Too generic — appear in almost every chunk
    "adó", "adók", "állam", "állami", "befektető", "eladó",
    "vevő", "szállító", "díj", "piac", "támogatás",
    "forrás", "cél", "alap", "terv", "költség", "összeg",
    "+36 1 755 5938", "barátja",
    "pályázat", "pályázó", "email", "tároló", "felhívás",
    "értesítés", "ingatlan", "melléklet", "meghatalmazás",
    "lakossagitarolo@neuzrt.hu",
)

CONF_TEXT_MATCH = 0.75
CONF_DOC_PROPAGATION = 0.5


async def run_backfill(dry_run=True):
    pool = await asyncpg.create_pool(PG_DSN, min_size=1, max_size=3)

    async with pool.acquire() as conn:
        total_entities = await conn.fetchval("SELECT count(*) FROM kg_entities")
        total_chunks = await conn.fetchval("SELECT count(*) FROM chunks")
        existing_links = await conn.fetchval("SELECT count(*) FROM kg_entity_chunks")
        existing_linked_chunks = await conn.fetchval(
            "SELECT count(DISTINCT chunk_id) FROM kg_entity_chunks"
        )

        print(f"=== OETP KG Entity-Chunk Backfill {'(DRY RUN)' if dry_run else '(LIVE)'} ===")
        print(f"Before: {total_entities} entities, {total_chunks} chunks")
        print(f"  Existing links: {existing_links} ({existing_linked_chunks} unique chunks = {existing_linked_chunks*100//total_chunks}%)\n")

        # ── Phase 1: SQL-based text matching ─────────────────────────
        print("[Phase 1] SQL text matching (entity name ILIKE in chunk content)...")
        t0 = time.time()

        # Create temp table with eligible entities (skip short/generic)
        await conn.execute("DROP TABLE IF EXISTS _backfill_entities")
        await conn.execute("""
            CREATE TEMP TABLE _backfill_entities AS
            SELECT id, name, type
            FROM kg_entities
            WHERE length(name) >= $1
              AND lower(name) NOT IN (SELECT unnest($2::text[]))
        """, MIN_NAME_LENGTH, list(SKIP_NAMES))

        eligible = await conn.fetchval("SELECT count(*) FROM _backfill_entities")
        print(f"  Eligible entities: {eligible} (of {total_entities})")

        # Phase 1: text match — find chunks containing entity name
        # Use ILIKE with %name% for substring match
        await conn.execute("DROP TABLE IF EXISTS _backfill_text_matches")
        await conn.execute("""
            CREATE TEMP TABLE _backfill_text_matches AS
            SELECT DISTINCT e.id as entity_id, c.id as chunk_id, c.doc_id
            FROM _backfill_entities e
            JOIN chunks c ON c.content ILIKE '%' || e.name || '%'
            WHERE NOT EXISTS (
                SELECT 1 FROM kg_entity_chunks ec
                WHERE ec.entity_id = e.id AND ec.chunk_id = c.id
            )
        """)

        text_matches = await conn.fetchval("SELECT count(*) FROM _backfill_text_matches")
        text_entities = await conn.fetchval("SELECT count(DISTINCT entity_id) FROM _backfill_text_matches")
        text_chunks = await conn.fetchval("SELECT count(DISTINCT chunk_id) FROM _backfill_text_matches")
        text_docs = await conn.fetchval("SELECT count(DISTINCT doc_id) FROM _backfill_text_matches")

        elapsed1 = time.time() - t0
        print(f"  Text matches: {text_matches} links ({text_entities} entities × {text_chunks} chunks in {text_docs} docs)")
        print(f"  Phase 1 time: {elapsed1:.1f}s")

        # Top entities by hit count
        top_entities = await conn.fetch("""
            SELECT e.name, e.type, count(*) as hits
            FROM _backfill_text_matches tm
            JOIN kg_entities e ON e.id = tm.entity_id
            GROUP BY e.name, e.type
            ORDER BY hits DESC
            LIMIT 15
        """)
        print("\n  Top entities by hits:")
        for r in top_entities:
            print(f"    {r['name'][:50]:50s} ({r['type']:15s}) → {r['hits']} chunks")

        # ── Phase 2: Document propagation ────────────────────────────
        print("\n[Phase 2] Document propagation...")
        t1 = time.time()

        # For each (entity, doc) pair from text matches,
        # link entity to other chunks in same document
        # Only for non-email docs (emails are standalone, propagation adds noise)
        await conn.execute("DROP TABLE IF EXISTS _backfill_propagation")
        await conn.execute("""
            CREATE TEMP TABLE _backfill_propagation AS
            SELECT DISTINCT tm.entity_id, c.id as chunk_id
            FROM (SELECT DISTINCT entity_id, doc_id FROM _backfill_text_matches) tm
            JOIN chunks c ON c.doc_id = tm.doc_id
            WHERE c.doc_type NOT IN ('email_reply', 'email_question')
              AND NOT EXISTS (
                SELECT 1 FROM kg_entity_chunks ec
                WHERE ec.entity_id = tm.entity_id AND ec.chunk_id = c.id
              )
              AND NOT EXISTS (
                SELECT 1 FROM _backfill_text_matches btm
                WHERE btm.entity_id = tm.entity_id AND btm.chunk_id = c.id
              )
        """)

        prop_links = await conn.fetchval("SELECT count(*) FROM _backfill_propagation")
        prop_chunks = await conn.fetchval("SELECT count(DISTINCT chunk_id) FROM _backfill_propagation")
        elapsed2 = time.time() - t1
        print(f"  Propagation links: {prop_links} ({prop_chunks} additional chunks)")
        print(f"  Phase 2 time: {elapsed2:.1f}s")

        # ── Summary & Insert ─────────────────────────────────────────
        total_new = text_matches + prop_links
        all_new_chunks = await conn.fetchval("""
            SELECT count(DISTINCT chunk_id) FROM (
                SELECT chunk_id FROM _backfill_text_matches
                UNION
                SELECT chunk_id FROM _backfill_propagation
            ) t
        """)
        est_coverage = (existing_linked_chunks + all_new_chunks) * 100 // total_chunks

        print(f"\n{'='*50}")
        print(f"SUMMARY:")
        print(f"  Text-match links:        {text_matches}")
        print(f"  Propagation links:        {prop_links}")
        print(f"  Total new links:          {total_new}")
        print(f"  New unique chunks:        {all_new_chunks}")
        print(f"  Estimated coverage:       {existing_linked_chunks*100//total_chunks}% → {est_coverage}%")

        if not dry_run:
            print("\nInserting...")
            t2 = time.time()

            await conn.execute("""
                INSERT INTO kg_entity_chunks (entity_id, chunk_id, confidence, extraction_method)
                SELECT entity_id, chunk_id, $1, 'text_match'
                FROM _backfill_text_matches
                ON CONFLICT (entity_id, chunk_id) DO NOTHING
            """, CONF_TEXT_MATCH)

            await conn.execute("""
                INSERT INTO kg_entity_chunks (entity_id, chunk_id, confidence, extraction_method)
                SELECT entity_id, chunk_id, $1, 'doc_propagation'
                FROM _backfill_propagation
                ON CONFLICT (entity_id, chunk_id) DO NOTHING
            """, CONF_DOC_PROPAGATION)

            elapsed3 = time.time() - t2
            print(f"  Insert time: {elapsed3:.1f}s")

            final_links = await conn.fetchval("SELECT count(*) FROM kg_entity_chunks")
            final_linked = await conn.fetchval("SELECT count(DISTINCT chunk_id) FROM kg_entity_chunks")
            coverage = final_linked * 100 // total_chunks
            print(f"\nAfter: {final_links} links ({final_linked} unique chunks = {coverage}%)")
            print(f"Delta: +{final_links - existing_links} links, coverage {existing_linked_chunks*100//total_chunks}% → {coverage}%")

            methods = await conn.fetch(
                "SELECT extraction_method, count(*) FROM kg_entity_chunks GROUP BY extraction_method ORDER BY count(*) DESC"
            )
            print("\nBreakdown by method:")
            for m in methods:
                print(f"  {m['extraction_method'] or 'unknown':20s} {m['count']}")
        else:
            print(f"\n(DRY RUN — no changes made.)")

        # Cleanup temp tables
        await conn.execute("DROP TABLE IF EXISTS _backfill_text_matches")
        await conn.execute("DROP TABLE IF EXISTS _backfill_propagation")
        await conn.execute("DROP TABLE IF EXISTS _backfill_entities")

    await pool.close()


def main():
    parser = argparse.ArgumentParser(description="OETP KG Entity-Chunk Backfill")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(run_backfill(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
