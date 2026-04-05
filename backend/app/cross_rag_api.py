"""Cross-RAG API — query entities across all RAG databases.

Provides endpoints to search and retrieve entities from the unified
cross_rag database, which links entities across jogszabaly_rag, neu_docs,
obsidian_rag, uae_legal_rag, and hanna_oetp.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

import asyncpg

# Add cross_rag_sync module to path (may not be available in all deployments)
sys.path.insert(0, os.environ.get("CROSSRAG_SCRIPTS", "/app/crossrag_scripts"))

try:
    from cross_rag_sync import (
        CROSSRAG_DSN,
        DB_CONFIGS,
        get_crossrag_pool,
        normalize_key,
        get_canonical_type,
    )
except ImportError:
    import logging as _log
    _log.getLogger(__name__).warning("cross_rag_sync not available — cross-RAG API disabled")
    CROSSRAG_DSN = ""
    DB_CONFIGS = {}
    async def get_crossrag_pool(): raise RuntimeError("cross_rag_sync not installed")
    def normalize_key(k, t=""): return k.lower()
    def get_canonical_type(t): return t

# Source DB connection pools (lazy init)
_source_pools: dict[str, asyncpg.Pool] = {}


def _resolve_dsn(dsn: str) -> str:
    """Replace localhost with host.docker.internal when running inside Docker."""
    docker_host = os.environ.get("CROSSRAG_DSN", "")
    if "host.docker.internal" in docker_host:
        return dsn.replace("localhost", "host.docker.internal")
    return dsn


async def _get_source_pool(db_name: str) -> Optional[asyncpg.Pool]:
    """Get connection pool for a source database."""
    if db_name not in DB_CONFIGS:
        return None
    cfg = DB_CONFIGS[db_name]
    if not cfg.get("dsn"):
        return None  # UAE has no direct DSN from host

    if db_name not in _source_pools or _source_pools[db_name]._closed:
        dsn = _resolve_dsn(cfg["dsn"])
        _source_pools[db_name] = await asyncpg.create_pool(dsn, min_size=1, max_size=3)
    return _source_pools[db_name]


async def search_canonical(
    q: str,
    entity_type: str = None,
    limit: int = 20,
) -> list[dict]:
    """Search canonical entities by name, return with source DB info."""
    pool = await get_crossrag_pool()
    async with pool.acquire() as conn:
        if entity_type:
            canonical_type = get_canonical_type(entity_type)
            nkey = normalize_key(q, canonical_type)
            rows = await conn.fetch("""
                SELECT c.id, c.canonical_name, c.canonical_type, c.normalized_key,
                       c.aliases, c.attributes,
                       array_agg(DISTINCT el.source_db ORDER BY el.source_db) as present_in,
                       count(DISTINCT el.source_db) as db_count
                FROM canonical_entities c
                JOIN entity_links el ON el.canonical_id = c.id
                WHERE c.canonical_type = $1
                  AND (similarity(c.normalized_key, $2) > 0.3
                       OR c.canonical_name ILIKE '%' || $3 || '%')
                GROUP BY c.id
                ORDER BY similarity(c.normalized_key, $2) DESC
                LIMIT $4
            """, canonical_type, nkey, q, limit)
        else:
            nkey = normalize_key(q, "concept")
            rows = await conn.fetch("""
                SELECT c.id, c.canonical_name, c.canonical_type, c.normalized_key,
                       c.aliases, c.attributes,
                       array_agg(DISTINCT el.source_db ORDER BY el.source_db) as present_in,
                       count(DISTINCT el.source_db) as db_count
                FROM canonical_entities c
                JOIN entity_links el ON el.canonical_id = c.id
                WHERE similarity(c.normalized_key, $1) > 0.3
                   OR c.canonical_name ILIKE '%' || $2 || '%'
                GROUP BY c.id
                ORDER BY similarity(c.normalized_key, $1) DESC
                LIMIT $3
            """, nkey, q, limit)

    results = []
    for r in rows:
        results.append({
            "id": r["id"],
            "canonical_name": r["canonical_name"],
            "canonical_type": r["canonical_type"],
            "aliases": json.loads(r["aliases"]) if isinstance(r["aliases"], str) else r["aliases"],
            "present_in": list(r["present_in"]),
            "db_count": r["db_count"],
        })
    return results


async def get_canonical_entity(canonical_id: int) -> Optional[dict]:
    """Get a canonical entity with all its source links and optional chunk counts."""
    pool = await get_crossrag_pool()
    async with pool.acquire() as conn:
        canonical = await conn.fetchrow(
            "SELECT * FROM canonical_entities WHERE id = $1", canonical_id
        )
        if not canonical:
            return None

        links = await conn.fetch(
            "SELECT * FROM entity_links WHERE canonical_id = $1 ORDER BY source_db",
            canonical_id,
        )

    result = {
        "id": canonical["id"],
        "canonical_name": canonical["canonical_name"],
        "canonical_type": canonical["canonical_type"],
        "normalized_key": canonical["normalized_key"],
        "aliases": json.loads(canonical["aliases"]) if isinstance(canonical["aliases"], str) else canonical["aliases"],
        "sources": {},
    }

    for link in links:
        db_name = link["source_db"]
        source_info = {
            "entity_id": link["source_entity_id"],
            "entity_name": link["source_entity_name"],
            "entity_type": link["source_entity_type"],
            "confidence": float(link["confidence"]),
            "match_method": link["match_method"],
        }

        # Try to get chunk count from source DB
        src_pool = await _get_source_pool(db_name)
        if src_pool:
            cfg = DB_CONFIGS[db_name]
            try:
                async with src_pool.acquire() as src_conn:
                    chunk_count = await src_conn.fetchval(
                        "SELECT count(*) FROM kg_entity_chunks WHERE entity_id = $1",
                        int(link["source_entity_id"]) if cfg["id_type"] == "int" else link["source_entity_id"],
                    )
                    source_info["chunk_count"] = chunk_count
            except Exception:
                source_info["chunk_count"] = None
        else:
            source_info["chunk_count"] = None

        result["sources"][db_name] = source_info

    return result


async def get_stats() -> dict:
    """Get cross-rag database statistics."""
    pool = await get_crossrag_pool()
    async with pool.acquire() as conn:
        total_canonical = await conn.fetchval("SELECT count(*) FROM canonical_entities")
        total_links = await conn.fetchval("SELECT count(*) FROM entity_links")

        type_rows = await conn.fetch(
            "SELECT canonical_type, count(*) as cnt FROM canonical_entities "
            "GROUP BY canonical_type ORDER BY cnt DESC"
        )
        db_rows = await conn.fetch(
            "SELECT source_db, count(*) as cnt FROM entity_links "
            "GROUP BY source_db ORDER BY cnt DESC"
        )
        multi_db = await conn.fetchval(
            """SELECT count(*) FROM canonical_entities c
               WHERE (SELECT count(DISTINCT source_db) FROM entity_links WHERE canonical_id = c.id) >= 2"""
        )

    return {
        "canonical_entities": total_canonical,
        "entity_links": total_links,
        "multi_db_entities": multi_db,
        "by_type": {r["canonical_type"]: r["cnt"] for r in type_rows},
        "by_source_db": {r["source_db"]: r["cnt"] for r in db_rows},
    }


async def get_multi_db_entities(
    min_dbs: int = 2,
    entity_type: str = None,
    limit: int = 50,
) -> list[dict]:
    """Get entities that appear in multiple databases."""
    pool = await get_crossrag_pool()
    async with pool.acquire() as conn:
        type_filter = "AND c.canonical_type = $3" if entity_type else ""
        params = [min_dbs, limit]
        if entity_type:
            params.append(get_canonical_type(entity_type))

        rows = await conn.fetch(f"""
            SELECT c.id, c.canonical_name, c.canonical_type,
                   array_agg(DISTINCT el.source_db ORDER BY el.source_db) as present_in,
                   count(DISTINCT el.source_db) as db_count
            FROM canonical_entities c
            JOIN entity_links el ON el.canonical_id = c.id
            {type_filter}
            GROUP BY c.id, c.canonical_name, c.canonical_type
            HAVING count(DISTINCT el.source_db) >= $1
            ORDER BY count(DISTINCT el.source_db) DESC, c.canonical_name
            LIMIT $2
        """, *params)

    return [{
        "id": r["id"],
        "canonical_name": r["canonical_name"],
        "canonical_type": r["canonical_type"],
        "present_in": list(r["present_in"]),
        "db_count": r["db_count"],
    } for r in rows]
