"""Post-ingest KG extraction for OETP chunks — deterministic, zero LLM cost.

Called automatically after /ingest/text, /ingest/pdf, /ingest/email-pair
as a BackgroundTask. Also usable as a CLI backfill tool.

Strategy:
1. Document entity from doc_type + source metadata
2. Text-match existing fogalom/szereplő/program entities in chunk content
3. Regex-based jogszabály reference extraction
4. Entity-chunk linking
5. Cross-RAG sync

Usage (backfill):
    docker exec -it cc-backend python -m app.rag.post_ingest_kg_oetp --backfill
    docker exec -it cc-backend python -m app.rag.post_ingest_kg_oetp --backfill --doc-type felhívás
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from typing import Optional

import asyncpg

import os
PG_DSN = os.environ.get("CC_PG_DSN", "postgresql://klara:klara_docs_2026@cc-db:5432/customercare")

# Jogszabály regex patterns (Hungarian legal references)
_LAW_PATTERNS = [
    # "55/2025. (II.28.) Korm. rendelet" or "55/2025"
    re.compile(
        r"(\d{1,4})/(\d{4})\.?\s*"
        r"(?:\([IVXLCDM]+\.?\s*\d+\.?\)\s*)?"
        r"(Korm\.\s*rendelet|törvény|miniszteri\s*rendelet|"
        r"önkormányzati\s*rendelet|BM\s*rendelet|NGM\s*rendelet|"
        r"PM\s*rendelet|IM\s*rendelet|KvVM\s*rendelet|"
        r"EMMI\s*rendelet|MNB\s*rendelet|NFM\s*rendelet|"
        r"EM\s*rendelet|FM\s*rendelet)?",
        re.IGNORECASE,
    ),
    # "2011. évi CXCV. törvény"
    re.compile(
        r"(\d{4})\.\s*évi\s+([IVXLCDM]+)\.\s*(törvény)",
        re.IGNORECASE,
    ),
    # EU regulations: "2014/24/EU", "1435/2003/EK"
    re.compile(
        r"(\d{1,4})/(\d{4})/(?:EU|EK|EGK)\b",
        re.IGNORECASE,
    ),
    # "Étv.", "Ptk.", "Kbt." style abbreviations
    re.compile(
        r"\b(Étv|Ptk|Kbt|Áht|Áfa\s*tv|Szja\s*tv|Art|Avt|Mt|Evt)\.",
        re.IGNORECASE,
    ),
]

# Minimum entity name length for text matching (skip short/noisy names)
MIN_ENTITY_NAME_LEN = 4

# Entity types to text-match in chunk content
MATCH_TYPES = ("fogalom", "szereplő", "program", "munkálat_tipus")

# Cache for entity names (loaded once per process)
_entity_cache: Optional[list[dict]] = None
_entity_cache_lock = asyncio.Lock()


async def _load_entity_cache(pool: asyncpg.Pool) -> list[dict]:
    """Load all matchable entities from kg_entities, sorted longest-first."""
    global _entity_cache
    async with _entity_cache_lock:
        if _entity_cache is not None:
            return _entity_cache

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT id, name, type, aliases
                   FROM kg_entities
                   WHERE type = ANY($1)
                     AND char_length(name) >= $2
                   ORDER BY char_length(name) DESC""",
                list(MATCH_TYPES),
                MIN_ENTITY_NAME_LEN,
            )

        _entity_cache = []
        for r in rows:
            name = r["name"]
            # Pre-compile case-insensitive word-boundary pattern
            escaped = re.escape(name)
            try:
                pattern = re.compile(r"\b" + escaped + r"\b", re.IGNORECASE)
            except re.error:
                pattern = None

            _entity_cache.append({
                "id": r["id"],
                "name": name,
                "type": r["type"],
                "pattern": pattern,
            })

        print(f"[kg-oetp] Entity cache loaded: {len(_entity_cache)} entities")
        return _entity_cache


def invalidate_entity_cache():
    """Clear the entity cache (call after new entities are created)."""
    global _entity_cache
    _entity_cache = None


def _extract_law_refs(text: str) -> list[str]:
    """Extract unique law reference strings from text."""
    refs = set()
    for pattern in _LAW_PATTERNS:
        for m in pattern.finditer(text):
            ref = m.group(0).strip().rstrip(".")
            if len(ref) >= 4:
                refs.add(ref)
    return sorted(refs)


async def _upsert_entity(
    conn: asyncpg.Connection,
    name: str,
    entity_type: str,
) -> int:
    """Upsert an entity and return its ID."""
    # Try insert; on conflict just return existing id
    entity_id = await conn.fetchval(
        """INSERT INTO kg_entities (name, type)
           VALUES ($1, $2)
           ON CONFLICT (name, type) DO UPDATE SET metadata = kg_entities.metadata
           RETURNING id""",
        name,
        entity_type,
    )
    return entity_id


async def _link_entity_chunk(
    conn: asyncpg.Connection,
    entity_id: int,
    chunk_id: str,
    extraction_method: str = "deterministic",
    confidence: float = 0.85,
) -> bool:
    """Link entity to chunk. Returns True if new link created."""
    result = await conn.execute(
        """INSERT INTO kg_entity_chunks (entity_id, chunk_id, confidence, extraction_method)
           VALUES ($1, $2, $3, $4)
           ON CONFLICT (entity_id, chunk_id) DO NOTHING""",
        entity_id,
        chunk_id,
        confidence,
        extraction_method,
    )
    return result == "INSERT 0 1"


async def post_ingest_kg_oetp(
    source: str,
    pool: asyncpg.Pool = None,
) -> dict:
    """Run deterministic KG extraction on all chunks for a given source (doc_id).

    Args:
        source: The doc_id of the ingested document
        pool: asyncpg pool (will create one if None)

    Returns:
        dict with stats: entities_created, links_created, laws_found, crossrag_stats
    """
    own_pool = pool is None
    if own_pool:
        pool = await asyncpg.create_pool(PG_DSN, min_size=2, max_size=5)

    stats = {
        "source": source,
        "entities_created": 0,
        "links_created": 0,
        "laws_found": 0,
        "chunks_processed": 0,
    }

    try:
        # Load entity cache
        entities = await _load_entity_cache(pool)

        # Get chunks for this source
        async with pool.acquire() as conn:
            chunks = await conn.fetch(
                """SELECT id, doc_id, doc_type, program, content, title
                   FROM chunks
                   WHERE doc_id = $1""",
                source,
            )

        if not chunks:
            print(f"[kg-oetp] No chunks found for source={source}")
            return stats

        async with pool.acquire() as conn:
            # 1. Document entity
            doc_type = chunks[0]["doc_type"]
            doc_entity_id = await _upsert_entity(conn, source, "dokumentum")

            # Link doc entity to all chunks
            for chunk in chunks:
                new = await _link_entity_chunk(
                    conn, doc_entity_id, chunk["id"],
                    extraction_method="deterministic", confidence=1.0,
                )
                if new:
                    stats["links_created"] += 1

            # 2. Program entity
            program = chunks[0]["program"] or "OETP"
            prog_entity_id = await _upsert_entity(conn, program, "program")
            for chunk in chunks:
                new = await _link_entity_chunk(
                    conn, prog_entity_id, chunk["id"],
                    extraction_method="deterministic", confidence=1.0,
                )
                if new:
                    stats["links_created"] += 1

            # 3. Process each chunk: text matching + law extraction
            for chunk in chunks:
                content = chunk["content"] or ""
                stats["chunks_processed"] += 1

                # 3a. Law reference extraction
                law_refs = _extract_law_refs(content)
                for ref in law_refs:
                    law_id = await _upsert_entity(conn, ref, "jogszabály")
                    new = await _link_entity_chunk(
                        conn, law_id, chunk["id"],
                        extraction_method="deterministic", confidence=0.90,
                    )
                    if new:
                        stats["links_created"] += 1
                        stats["laws_found"] += 1

                    # Create HIVATKOZIK relation: doc → law
                    # Check if relation already exists to avoid duplicates
                    existing = await conn.fetchval(
                        """SELECT id FROM kg_relations
                           WHERE source_id = $1 AND target_id = $2 AND relation_type = 'HIVATKOZIK'""",
                        doc_entity_id, law_id,
                    )
                    if not existing:
                        await conn.execute(
                            """INSERT INTO kg_relations
                                   (source_id, target_id, relation_type, source_chunk_id, weight)
                               VALUES ($1, $2, 'HIVATKOZIK', $3, 0.90)""",
                            doc_entity_id, law_id, chunk["id"],
                        )

                # 3b. Text-match existing entities
                for ent in entities:
                    if ent["pattern"] is None:
                        continue
                    if ent["pattern"].search(content):
                        new = await _link_entity_chunk(
                            conn, ent["id"], chunk["id"],
                            extraction_method="deterministic", confidence=0.85,
                        )
                        if new:
                            stats["links_created"] += 1

        # 4. Cross-RAG sync
        crossrag_stats = await _crossrag_sync(pool, source)
        stats["crossrag"] = crossrag_stats

    except Exception as e:
        print(f"[kg-oetp] Error processing source={source}: {e}")
        stats["error"] = str(e)
    finally:
        if own_pool:
            await pool.close()

    print(
        f"[kg-oetp] {source}: {stats['chunks_processed']} chunks, "
        f"{stats['links_created']} new links, {stats['laws_found']} law refs"
    )
    return stats


async def _crossrag_sync(pool: asyncpg.Pool, source: str) -> dict:
    """Sync entities from this source's chunks to cross_rag DB."""
    try:
        # Import cross_rag_sync — mounted at /app/crossrag_scripts/ or /app/scripts/
        import os
        crossrag_path = os.environ.get("CROSSRAG_SCRIPTS", "/app/crossrag_scripts")
        if crossrag_path not in sys.path:
            sys.path.insert(0, crossrag_path)
        # Fallback path
        if "/app/scripts" not in sys.path:
            sys.path.insert(0, "/app/scripts")
        from cross_rag_sync import sync_entities_to_crossrag, get_crossrag_pool

        # Gather all entities linked to this source's chunks
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT DISTINCT e.id::text as id, e.name, e.type
                   FROM kg_entities e
                   JOIN kg_entity_chunks ec ON e.id = ec.entity_id
                   JOIN chunks c ON ec.chunk_id = c.id
                   WHERE c.doc_id = $1""",
                source,
            )

        if not rows:
            return {"skipped": "no entities"}

        entities = [dict(r) for r in rows]
        crossrag_pool = await get_crossrag_pool()
        result = await sync_entities_to_crossrag("customercare", entities, crossrag_pool)
        return result

    except ImportError:
        print("[kg-oetp] cross_rag_sync not available — skipping cross-RAG sync")
        return {"skipped": "import_error"}
    except Exception as e:
        print(f"[kg-oetp] Cross-RAG sync error: {e}")
        return {"error": str(e)}


# --- Backfill CLI ---


async def backfill(
    doc_types: list[str] | None = None,
    limit: int | None = None,
    skip_linked: bool = True,
):
    """Run KG extraction on existing chunks that haven't been processed yet.

    Args:
        doc_types: Filter by doc_type (None = all)
        limit: Max number of sources to process
        skip_linked: Skip sources that already have entity-chunk links
    """
    pool = await asyncpg.create_pool(PG_DSN, min_size=2, max_size=10)

    # Find distinct sources
    if skip_linked:
        # Sources with no deterministic entity-chunk links
        sql = """
            SELECT DISTINCT c.doc_id, c.doc_type, count(*) as chunk_count
            FROM chunks c
            LEFT JOIN kg_entity_chunks ec ON c.id = ec.chunk_id
                AND ec.extraction_method = 'deterministic'
            WHERE ec.entity_id IS NULL
        """
    else:
        sql = """
            SELECT DISTINCT c.doc_id, c.doc_type, count(*) as chunk_count
            FROM chunks c
            WHERE 1=1
        """

    params = []
    if doc_types:
        sql += f" AND c.doc_type = ANY(${len(params) + 1})"
        params.append(doc_types)

    sql += " GROUP BY c.doc_id, c.doc_type ORDER BY chunk_count DESC"

    if limit:
        sql += f" LIMIT ${len(params) + 1}"
        params.append(limit)

    async with pool.acquire() as conn:
        sources = await conn.fetch(sql, *params)

    print(f"[kg-oetp backfill] Found {len(sources)} sources to process")

    total_stats = {
        "sources_processed": 0,
        "total_links": 0,
        "total_laws": 0,
        "total_chunks": 0,
    }

    for i, src in enumerate(sources, 1):
        doc_id = src["doc_id"]
        result = await post_ingest_kg_oetp(doc_id, pool)
        total_stats["sources_processed"] += 1
        total_stats["total_links"] += result.get("links_created", 0)
        total_stats["total_laws"] += result.get("laws_found", 0)
        total_stats["total_chunks"] += result.get("chunks_processed", 0)

        if i % 50 == 0:
            print(f"[kg-oetp backfill] Progress: {i}/{len(sources)} sources")

    await pool.close()

    print(f"\n=== Backfill Complete ===")
    print(f"Sources processed: {total_stats['sources_processed']}")
    print(f"Chunks processed:  {total_stats['total_chunks']}")
    print(f"New links created: {total_stats['total_links']}")
    print(f"Law refs found:    {total_stats['total_laws']}")
    return total_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OETP KG post-ingest extraction")
    parser.add_argument("--backfill", action="store_true", help="Process all unlinked chunks")
    parser.add_argument("--doc-type", nargs="*", help="Filter by doc_type (e.g. felhívás email_reply)")
    parser.add_argument("--limit", type=int, help="Max sources to process")
    parser.add_argument("--source", help="Process a single source doc_id")
    parser.add_argument("--no-skip", action="store_true", help="Process all sources (don't skip already linked)")
    args = parser.parse_args()

    if args.source:
        asyncio.run(post_ingest_kg_oetp(args.source))
    elif args.backfill or args.doc_type:
        asyncio.run(backfill(
            doc_types=args.doc_type,
            limit=args.limit,
            skip_linked=not args.no_skip,
        ))
    else:
        parser.print_help()
