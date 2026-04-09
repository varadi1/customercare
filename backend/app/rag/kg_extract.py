"""Inline Knowledge Graph extraction for the ingest pipeline.

Extracts entities and relations from chunks using LLM (gpt-4o-mini)
and stores them in kg_entities / kg_relations / kg_entity_chunks.

Called automatically after chunk insertion in ingest.py.
"""

from __future__ import annotations

import json
import logging
import os

import asyncpg
import httpx
import json_repair

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = "gpt-4o-mini"

# Only extract from high-value doc types (skip emails — too noisy)
EXTRACT_DOC_TYPES = {"felhívás", "melléklet", "közlemény", "gyik", "segédlet", "dokumentum"}

KG_PROMPT = """Entitás és reláció kinyerés OETP (Otthoni Energiatároló Program) dokumentumból.

Dokumentum típus: {doc_type}
Forrás: {source}

Szöveg:
---
{content}
---

Feladat: Nyerd ki a szövegből az entitásokat és relációkat az alábbi JSON formátumban.

Entitás típusok:
- fogalom: energetikai tanúsítvány, számla, költségvetés, műszaki ellenőr, stb.
- feltétel: jogosultsági feltételek (max összeg, min besorolás, stb.)
- program: OETP, NPP2, Otthonfelújítás, stb.
- munkálat_tipus: nyílászáró csere, hőszigetelés, napelem, fűtéskorszerűsítés, stb.
- szereplő: pályázó, kivitelező, műszaki ellenőr, energetikai tanúsítvány kiállító, stb.
- jogszabály: jogszabály hivatkozások (pl. 55/2025, Étv., 176/2008)
- dokumentum: felhívás, melléklet, GYIK, stb.

Reláció típusok:
- DEFINIÁLJA: dokumentum definiál egy fogalmat
- FELTÉTELE: program feltétele valami
- VONATKOZIK: fogalom vonatkozik valamire
- HIVATKOZIK: dokumentum hivatkozik jogszabályra
- RÉSZE: munkálat_tipus része valaminek
- SZÜKSÉGES: feltételhez szükséges dokumentum/igazolás

Válasz JSON (CSAK ezt add vissza, semmi mást):
{{"entities": [{{"name": "...", "type": "...", "aliases": []}}], "relations": [{{"source": "...", "target": "...", "type": "..."}}]}}"""


async def _call_openai(content: str, doc_type: str, source: str) -> dict | None:
    """Call OpenAI API for entity/relation extraction."""
    if not OPENAI_API_KEY:
        return None

    prompt = KG_PROMPT.format(
        doc_type=doc_type,
        source=source[:100],
        content=content[:3000],
    )

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": "JSON entitás és reláció extraction. Válaszolj CSAK valid JSON-nel."},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 1000,
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"},
                },
            )
            if resp.status_code != 200:
                logger.warning("KG extraction API error: %s", resp.status_code)
                return None

            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                # LLM sometimes returns truncated JSON — try repair
                repaired = json_repair.loads(text)
                if isinstance(repaired, dict):
                    return repaired
                return None

    except Exception as e:
        logger.warning("KG extraction failed for %s: %s", source, e)
        return None


async def extract_and_store(
    conn: asyncpg.Connection,
    chunk_id: str,
    content: str,
    doc_type: str,
    source: str,
) -> tuple[int, int]:
    """Extract entities/relations from a single chunk and store in DB.

    Returns (entities_count, relations_count).
    """
    if doc_type not in EXTRACT_DOC_TYPES:
        return 0, 0

    result = await _call_openai(content, doc_type, source)
    if not result:
        return 0, 0

    entities_created = 0
    relations_created = 0

    # Insert entities
    for ent in result.get("entities", []):
        name = ent.get("name", "").strip()
        etype = ent.get("type", "fogalom").strip()
        aliases = ent.get("aliases", [])

        if not name or len(name) < 2:
            continue

        try:
            entity_id = await conn.fetchval(
                """INSERT INTO kg_entities (name, type, aliases)
                   VALUES ($1, $2, $3)
                   ON CONFLICT (name, type) DO UPDATE SET
                       aliases = ARRAY(SELECT DISTINCT unnest(kg_entities.aliases || EXCLUDED.aliases))
                   RETURNING id""",
                name, etype, aliases,
            )

            await conn.execute(
                """INSERT INTO kg_entity_chunks (entity_id, chunk_id, confidence, extraction_method)
                   VALUES ($1, $2, 0.75, 'llm-inline')
                   ON CONFLICT DO NOTHING""",
                entity_id, chunk_id,
            )
            entities_created += 1
        except Exception as e:
            logger.debug("KG entity insert error: %s", e)

    # Insert relations
    for rel in result.get("relations", []):
        src_name = rel.get("source", "").strip()
        tgt_name = rel.get("target", "").strip()
        rel_type = rel.get("type", "VONATKOZIK").strip()

        if not src_name or not tgt_name:
            continue

        try:
            src_id = await conn.fetchval(
                "SELECT id FROM kg_entities WHERE name = $1 LIMIT 1", src_name
            )
            tgt_id = await conn.fetchval(
                "SELECT id FROM kg_entities WHERE name = $1 LIMIT 1", tgt_name
            )

            if src_id and tgt_id:
                await conn.execute(
                    """INSERT INTO kg_relations (source_id, target_id, relation_type, source_chunk_id)
                       VALUES ($1, $2, $3, $4)
                       ON CONFLICT DO NOTHING""",
                    src_id, tgt_id, rel_type, chunk_id,
                )
                relations_created += 1
        except Exception as e:
            logger.debug("KG relation insert error: %s", e)

    return entities_created, relations_created


async def extract_batch(
    pool: asyncpg.Pool,
    chunk_ids: list[str],
) -> tuple[int, int]:
    """Extract KG for a list of chunk IDs (for backfill/manual runs).

    Returns (total_entities, total_relations).
    """
    total_ent = 0
    total_rel = 0

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id, content, doc_type, doc_id
               FROM chunks WHERE id = ANY($1)""",
            chunk_ids,
        )

    for row in rows:
        async with pool.acquire() as conn:
            ent, rel = await extract_and_store(
                conn, row["id"], row["content"], row["doc_type"], row["doc_id"],
            )
            total_ent += ent
            total_rel += rel

    return total_ent, total_rel
