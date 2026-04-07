"""Hanna OETP RAG healthcheck — napi ellenőrzés.

Usage:
    docker exec hanna-backend python3 /app/scripts/healthcheck.py
    
Returns JSON with all checks and overall status (ok/warning/critical).
"""

import asyncio
import json
import sys

import asyncpg

import os
PG_DSN = os.environ.get("HANNA_PG_DSN", "postgresql://klara:klara_docs_2026@hanna-db:5432/hanna_oetp")


async def run_checks():
    report = {"status": "ok", "checks": {}, "warnings": [], "errors": []}

    try:
        pool = await asyncpg.create_pool(PG_DSN, min_size=1, max_size=3)
    except Exception as e:
        report["status"] = "critical"
        report["errors"].append(f"DB connection failed: {e}")
        return report

    async with pool.acquire() as conn:
        # 1. Chunks
        total = await conn.fetchval("SELECT count(*) FROM chunks")
        report["checks"]["chunks_total"] = total
        if total == 0:
            report["status"] = "critical"
            report["errors"].append("No chunks in database!")

        # 2. Embeddings
        has_emb = await conn.fetchval("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL")
        missing_emb = total - has_emb
        report["checks"]["embeddings"] = {"present": has_emb, "missing": missing_emb}
        if missing_emb > 0:
            report["warnings"].append(f"{missing_emb} chunks missing embeddings")

        # 3. BM25 tsvector
        has_tsv = await conn.fetchval("SELECT count(*) FROM chunks WHERE content_tsvector IS NOT NULL")
        report["checks"]["bm25_tsvector"] = {"present": has_tsv, "missing": total - has_tsv}

        # 4. Authority scores
        null_auth = await conn.fetchval("SELECT count(*) FROM chunks WHERE authority_score IS NULL")
        auth_dist = await conn.fetch(
            "SELECT doc_type, count(*) as cnt, avg(authority_score)::numeric(4,2) as avg_auth "
            "FROM chunks GROUP BY doc_type ORDER BY avg_auth DESC"
        )
        report["checks"]["authority"] = {
            "null_count": null_auth,
            "distribution": {r["doc_type"]: {"count": r["cnt"], "avg": float(r["avg_auth"])} for r in auth_dist},
        }
        if null_auth > 0:
            report["warnings"].append(f"{null_auth} chunks with NULL authority")

        # 5. Enrichment
        enriched = await conn.fetchval("SELECT count(*) FROM chunks WHERE content_enriched IS NOT NULL")
        report["checks"]["enrichment"] = {"enriched": enriched, "not_enriched": total - enriched}
        if enriched == 0 and total > 0:
            report["warnings"].append("No chunks have contextual enrichment")

        # 6. Metadata
        has_date = await conn.fetchval("SELECT count(*) FROM chunks WHERE source_date IS NOT NULL")
        has_hash = await conn.fetchval("SELECT count(*) FROM chunks WHERE content_hash IS NOT NULL")
        report["checks"]["metadata"] = {
            "has_source_date": has_date,
            "has_content_hash": has_hash,
            "missing_date": total - has_date,
        }

        # 7. Knowledge Graph
        ent_count = await conn.fetchval("SELECT count(*) FROM kg_entities")
        rel_count = await conn.fetchval("SELECT count(*) FROM kg_relations")
        links = await conn.fetchval("SELECT count(*) FROM kg_entity_chunks")
        ent_types = await conn.fetch("SELECT type, count(*) as cnt FROM kg_entities GROUP BY type ORDER BY cnt DESC")
        rel_types = await conn.fetch("SELECT relation_type, count(*) as cnt FROM kg_relations GROUP BY relation_type ORDER BY cnt DESC")

        report["checks"]["knowledge_graph"] = {
            "entities": ent_count,
            "relations": rel_count,
            "entity_chunk_links": links,
            "entity_types": {r["type"]: r["cnt"] for r in ent_types},
            "relation_types": {r["relation_type"]: r["cnt"] for r in rel_types},
        }
        if ent_count == 0:
            report["warnings"].append("Knowledge graph is empty")

        # 8. Index check
        indexes = await conn.fetch(
            "SELECT indexname FROM pg_indexes WHERE tablename = 'chunks'"
        )
        idx_names = [r["indexname"] for r in indexes]
        expected = ["idx_chunks_embedding", "idx_chunks_tsvector", "idx_chunks_doc_type", "idx_chunks_authority"]
        missing_idx = [i for i in expected if i not in idx_names]
        report["checks"]["indexes"] = {"present": idx_names, "missing": missing_idx}
        if missing_idx:
            report["warnings"].append(f"Missing indexes: {missing_idx}")

        # 9. Programs
        programs = await conn.fetch("SELECT program, count(*) as cnt FROM chunks GROUP BY program ORDER BY cnt DESC")
        report["checks"]["programs"] = {r["program"]: r["cnt"] for r in programs}

    await pool.close()

    # Set overall status
    if report["errors"]:
        report["status"] = "critical"
    elif report["warnings"]:
        report["status"] = "warning"

    return report


def main():
    report = asyncio.run(run_checks())
    compact = "--compact" in sys.argv

    if compact:
        c = report["checks"]
        print(f"Status: {report['status']} | "
              f"Chunks: {c['chunks_total']} | "
              f"Emb: {c['embeddings']['present']}/{c['chunks_total']} | "
              f"BM25: {c['bm25_tsvector']['present']}/{c['chunks_total']} | "
              f"Enriched: {c['enrichment']['enriched']}/{c['chunks_total']} | "
              f"KG: {c['knowledge_graph']['entities']} ent / {c['knowledge_graph']['relations']} rel | "
              f"Idx: {len(c['indexes']['present'])} ok, {len(c['indexes']['missing'])} missing")
        if report["warnings"]:
            print(f"Warnings: {'; '.join(report['warnings'])}")
        if report["errors"]:
            print(f"ERRORS: {'; '.join(report['errors'])}")
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))

    sys.exit(0 if report["status"] != "critical" else 1)


if __name__ == "__main__":
    main()
