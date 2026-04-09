"""Knowledge Graph entity & relation extraction via OpenAI Batch API.

Extracts entities (fogalom, feltétel, program, munkálat, szereplő, jogszabály)
and relations from OETP knowledge base chunks.

Usage:
    python3 scripts/kg_extraction_batch.py prepare
    python3 scripts/kg_extraction_batch.py status
    python3 scripts/kg_extraction_batch.py apply
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import asyncpg

import os
PG_DSN = os.environ.get("HANNA_PG_DSN", "postgresql://klara:klara_docs_2026@cc-db:5432/customercare")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
BATCH_DIR = Path("/app/data/kg_batch")
MODEL = "gpt-4o-mini"

# Only extract from high-value chunks (skip most emails)
EXTRACT_DOC_TYPES = ("felhívás", "melléklet", "közlemény", "gyik", "segédlet", "dokumentum")

KG_PROMPT = """Entitás és reláció kinyerés OETP (Otthonfelújítási Program) dokumentumból.

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


async def get_extractable_chunks():
    pool = await asyncpg.create_pool(PG_DSN, min_size=1, max_size=5)
    async with pool.acquire() as conn:
        # Extract from official docs (not emails - too noisy)
        rows = await conn.fetch(
            """SELECT id, doc_id, doc_type, program, content 
               FROM chunks 
               WHERE doc_type = ANY($1)
               ORDER BY authority_score DESC, id""",
            list(EXTRACT_DOC_TYPES),
        )
    await pool.close()
    return rows


async def prepare_batch():
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    
    rows = await get_extractable_chunks()
    print(f"Chunks for KG extraction: {len(rows)}")
    
    if not rows:
        print("Nothing to extract!")
        return
    
    batch_file = BATCH_DIR / "kg_input.jsonl"
    with open(batch_file, "w") as f:
        for row in rows:
            prompt = KG_PROMPT.format(
                source=row["doc_id"][:100],
                doc_type=row["doc_type"],
                content=row["content"][:3000],
            )
            
            request = {
                "custom_id": row["id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": "JSON entitás és reláció extraction. Válaszolj CSAK valid JSON-nel."},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 500,
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"},
                },
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")
    
    print(f"Batch file: {batch_file} ({len(rows)} requests)")
    
    if OPENAI_API_KEY:
        print("\nAuto-submitting batch...")
        import httpx
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        
        async with httpx.AsyncClient(timeout=60) as client:
            with open(batch_file, "rb") as bf:
                resp = await client.post(
                    "https://api.openai.com/v1/files",
                    headers=headers,
                    files={"file": ("kg_input.jsonl", bf, "application/jsonl")},
                    data={"purpose": "batch"},
                )
            
            if resp.status_code != 200:
                print(f"Upload failed: {resp.text}")
                return
            
            file_id = resp.json()["id"]
            print(f"Uploaded: {file_id}")
            
            resp = await client.post(
                "https://api.openai.com/v1/batches",
                headers=headers,
                json={
                    "input_file_id": file_id,
                    "endpoint": "/v1/chat/completions",
                    "completion_window": "24h",
                },
            )
            
            if resp.status_code != 200:
                print(f"Batch creation failed: {resp.text}")
                return
            
            batch_id = resp.json()["id"]
            print(f"Batch created: {batch_id}")
            (BATCH_DIR / "batch_id.txt").write_text(batch_id)


async def check_status():
    batch_id_file = BATCH_DIR / "batch_id.txt"
    if not batch_id_file.exists():
        print("No batch ID found.")
        return
    
    batch_id = batch_id_file.read_text().strip()
    
    if not OPENAI_API_KEY:
        print(f"Batch ID: {batch_id}")
        return
    
    import httpx
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"https://api.openai.com/v1/batches/{batch_id}", headers=headers)
        data = resp.json()
        print(f"Batch: {batch_id}")
        print(f"Status: {data['status']}")
        print(f"Total: {data.get('request_counts', {}).get('total', '?')}")
        print(f"Completed: {data.get('request_counts', {}).get('completed', '?')}")
        print(f"Failed: {data.get('request_counts', {}).get('failed', '?')}")
        
        if data["status"] == "completed":
            output_file_id = data.get("output_file_id")
            (BATCH_DIR / "output_file_id.txt").write_text(output_file_id or "")
            print(f"Output file: {output_file_id}")


async def apply_results():
    output_jsonl = BATCH_DIR / "kg_output.jsonl"
    output_file = BATCH_DIR / "output_file_id.txt"
    
    if not output_jsonl.exists() and output_file.exists() and OPENAI_API_KEY:
        file_id = output_file.read_text().strip()
        import httpx
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.get(f"https://api.openai.com/v1/files/{file_id}/content", headers=headers)
            output_jsonl.write_bytes(resp.content)
            print(f"Downloaded: {output_jsonl}")
    
    if not output_jsonl.exists():
        print("No output file. Check status first.")
        return
    
    pool = await asyncpg.create_pool(PG_DSN, min_size=2, max_size=10)
    
    entities_created = 0
    relations_created = 0
    errors = 0
    
    with open(output_jsonl) as f:
        for line in f:
            try:
                result = json.loads(line)
                chunk_id = result["custom_id"]
                
                if result.get("error"):
                    errors += 1
                    continue
                
                content = result["response"]["body"]["choices"][0]["message"]["content"]
                data = json.loads(content)
                
                async with pool.acquire() as conn:
                    # Insert entities
                    for ent in data.get("entities", []):
                        name = ent.get("name", "").strip()
                        etype = ent.get("type", "fogalom").strip()
                        aliases = ent.get("aliases", [])
                        
                        if not name or len(name) < 2:
                            continue
                        
                        # Upsert entity
                        entity_id = await conn.fetchval(
                            """INSERT INTO kg_entities (name, type, aliases)
                               VALUES ($1, $2, $3)
                               ON CONFLICT (name, type) DO UPDATE SET
                                   aliases = ARRAY(SELECT DISTINCT unnest(kg_entities.aliases || EXCLUDED.aliases))
                               RETURNING id""",
                            name, etype, aliases,
                        )
                        
                        # Link to chunk
                        await conn.execute(
                            """INSERT INTO kg_entity_chunks (entity_id, chunk_id, confidence, extraction_method)
                               VALUES ($1, $2, 0.75, 'llm')
                               ON CONFLICT DO NOTHING""",
                            entity_id, chunk_id,
                        )
                        entities_created += 1
                    
                    # Insert relations
                    for rel in data.get("relations", []):
                        src_name = rel.get("source", "").strip()
                        tgt_name = rel.get("target", "").strip()
                        rel_type = rel.get("type", "VONATKOZIK").strip()
                        
                        if not src_name or not tgt_name:
                            continue
                        
                        # Find source and target entity IDs
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
                errors += 1
                if errors <= 5:
                    print(f"Error: {e}")
    
    await pool.close()
    
    # Print stats
    pool2 = await asyncpg.create_pool(PG_DSN, min_size=1, max_size=2)
    async with pool2.acquire() as conn:
        ent_count = await conn.fetchval("SELECT count(*) FROM kg_entities")
        rel_count = await conn.fetchval("SELECT count(*) FROM kg_relations")
        types = await conn.fetch("SELECT type, count(*) as cnt FROM kg_entities GROUP BY type ORDER BY cnt DESC")
    await pool2.close()
    
    print(f"\n=== KG Extraction Results ===")
    print(f"Entities created/updated: {entities_created}")
    print(f"Relations created: {relations_created}")
    print(f"Errors: {errors}")
    print(f"\nTotal entities: {ent_count}")
    print(f"Total relations: {rel_count}")
    print(f"\nEntity types:")
    for t in types:
        print(f"  {t['type']:20s} {t['cnt']:6d}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "prepare"
    
    if cmd == "prepare":
        asyncio.run(prepare_batch())
    elif cmd == "status":
        asyncio.run(check_status())
    elif cmd == "apply":
        asyncio.run(apply_results())
    else:
        print(f"Unknown: {cmd}. Use: prepare|status|apply")
