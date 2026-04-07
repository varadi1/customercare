"""Contextual enrichment for existing chunks using OpenAI Batch API.

Generates context summaries for each chunk and stores in content_enriched.
Uses gpt-4o-mini via Batch API for cost efficiency (~50% cheaper).

Usage:
    # Prepare batch
    python3 scripts/contextual_enrichment_batch.py prepare
    
    # Check status
    python3 scripts/contextual_enrichment_batch.py status
    
    # Download and apply results
    python3 scripts/contextual_enrichment_batch.py apply
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import asyncpg

import os
PG_DSN = os.environ.get("HANNA_PG_DSN", "postgresql://klara:klara_docs_2026@hanna-db:5432/hanna_oetp")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
BATCH_DIR = Path("/app/data/enrichment_batch")
MODEL = "gpt-4o-mini"


ENRICHMENT_PROMPT = """Kontextus összefoglaló generálása egy OETP (Otthonfelújítási Program) tudásbázis chunkhoz.

A chunk a következő dokumentumból származik:
- Forrás: {source}
- Típus: {doc_type}
- Program: {program}

Chunk szöveg:
---
{content}
---

Feladat: Írj egy 1-2 mondatos kontextus összefoglalót, ami segít megérteni miről szól ez a szövegrész és hogyan kapcsolódik az OETP pályázati rendszerhez. A kontextus legyen informatív és keresés-barát — tartalmazzon kulcsszavakat amik segítenek megtalálni ezt a chunkot releváns kérdéseknél.

Kontextus összefoglaló:"""


async def get_chunks_needing_enrichment():
    pool = await asyncpg.create_pool(PG_DSN, min_size=1, max_size=5)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id, doc_id, doc_type, program, content 
               FROM chunks 
               WHERE content_enriched IS NULL
               ORDER BY authority_score DESC, id"""
        )
    await pool.close()
    return rows


async def prepare_batch():
    """Prepare JSONL file for OpenAI Batch API."""
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    
    rows = await get_chunks_needing_enrichment()
    print(f"Chunks needing enrichment: {len(rows)}")
    
    if not rows:
        print("Nothing to enrich!")
        return
    
    batch_file = BATCH_DIR / "enrichment_input.jsonl"
    with open(batch_file, "w") as f:
        for row in rows:
            prompt = ENRICHMENT_PROMPT.format(
                source=row["doc_id"][:100],
                doc_type=row["doc_type"],
                program=row["program"] or "OETP",
                content=row["content"][:2000],  # Truncate long chunks
            )
            
            request = {
                "custom_id": row["id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": "Magyar nyelvű kontextus összefoglalókat generálsz OETP pályázati dokumentumokhoz. Rövid, informatív, keresés-barát."},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 200,
                    "temperature": 0.3,
                },
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")
    
    print(f"Batch file created: {batch_file} ({len(rows)} requests)")
    print(f"\nNext steps:")
    print(f"1. Upload: curl https://api.openai.com/v1/files -F purpose=batch -F file=@{batch_file}")
    print(f"2. Create batch: curl https://api.openai.com/v1/batches -d '{{\"input_file_id\": \"FILE_ID\", \"endpoint\": \"/v1/chat/completions\", \"completion_window\": \"24h\"}}'")
    
    # Auto-submit if API key available
    if OPENAI_API_KEY:
        print("\nAuto-submitting batch...")
        import httpx
        
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        
        # Upload file
        async with httpx.AsyncClient(timeout=60) as client:
            with open(batch_file, "rb") as bf:
                resp = await client.post(
                    "https://api.openai.com/v1/files",
                    headers=headers,
                    files={"file": ("enrichment_input.jsonl", bf, "application/jsonl")},
                    data={"purpose": "batch"},
                )
            
            if resp.status_code != 200:
                print(f"Upload failed: {resp.text}")
                return
            
            file_id = resp.json()["id"]
            print(f"Uploaded: {file_id}")
            
            # Create batch
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
            
            # Save batch ID
            (BATCH_DIR / "batch_id.txt").write_text(batch_id)
            print(f"Saved batch ID to {BATCH_DIR / 'batch_id.txt'}")


async def check_status():
    """Check batch status."""
    batch_id_file = BATCH_DIR / "batch_id.txt"
    if not batch_id_file.exists():
        print("No batch ID found. Run 'prepare' first.")
        return
    
    batch_id = batch_id_file.read_text().strip()
    
    if not OPENAI_API_KEY:
        print(f"Batch ID: {batch_id}")
        print("Set OPENAI_API_KEY to check status automatically.")
        return
    
    import httpx
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"https://api.openai.com/v1/batches/{batch_id}",
            headers=headers,
        )
        
        if resp.status_code != 200:
            print(f"Status check failed: {resp.text}")
            return
        
        data = resp.json()
        print(f"Batch: {batch_id}")
        print(f"Status: {data['status']}")
        print(f"Total: {data.get('request_counts', {}).get('total', '?')}")
        print(f"Completed: {data.get('request_counts', {}).get('completed', '?')}")
        print(f"Failed: {data.get('request_counts', {}).get('failed', '?')}")
        
        if data["status"] == "completed":
            output_file_id = data.get("output_file_id")
            print(f"\nOutput file: {output_file_id}")
            print("Run 'apply' to download and apply results.")
            (BATCH_DIR / "output_file_id.txt").write_text(output_file_id or "")


async def apply_results():
    """Download batch results and apply to database."""
    # Try to get output file ID
    output_file = BATCH_DIR / "output_file_id.txt"
    output_jsonl = BATCH_DIR / "enrichment_output.jsonl"
    
    if output_jsonl.exists():
        print(f"Using cached output: {output_jsonl}")
    elif output_file.exists() and OPENAI_API_KEY:
        file_id = output_file.read_text().strip()
        if not file_id:
            print("No output file ID. Check batch status first.")
            return
        
        import httpx
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.get(
                f"https://api.openai.com/v1/files/{file_id}/content",
                headers=headers,
            )
            
            if resp.status_code != 200:
                print(f"Download failed: {resp.text}")
                return
            
            output_jsonl.write_bytes(resp.content)
            print(f"Downloaded: {output_jsonl}")
    else:
        print("No output file found. Check batch status.")
        return
    
    # Parse results and update DB
    pool = await asyncpg.create_pool(PG_DSN, min_size=2, max_size=10)
    
    updated = 0
    errors = 0
    
    with open(output_jsonl) as f:
        for line in f:
            try:
                result = json.loads(line)
                chunk_id = result["custom_id"]
                
                if result.get("error"):
                    errors += 1
                    continue
                
                enrichment = result["response"]["body"]["choices"][0]["message"]["content"].strip()
                
                async with pool.acquire() as conn:
                    await conn.execute(
                        "UPDATE chunks SET content_enriched = $1, updated_at = now() WHERE id = $2",
                        enrichment,
                        chunk_id,
                    )
                updated += 1
                
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"Error: {e}")
    
    await pool.close()
    print(f"\nApplied: {updated} enriched, {errors} errors")
    
    # Re-embed enriched chunks (content + enrichment)
    print("\nNote: Run re-embedding separately to update embeddings with enriched content.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "prepare"
    
    if cmd == "prepare":
        asyncio.run(prepare_batch())
    elif cmd == "status":
        asyncio.run(check_status())
    elif cmd == "apply":
        asyncio.run(apply_results())
    else:
        print(f"Unknown command: {cmd}. Use: prepare|status|apply")
