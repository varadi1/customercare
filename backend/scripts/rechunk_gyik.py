#!/usr/bin/env python3
"""Re-chunk GYIK PDF by Q&A pairs instead of fixed-size chunks.

The OEPT_GYIK_20260204.pdf contains 28 numbered Q&A pairs across 2 sections:
- Section A (Q1-17): "PÁLYÁZATI SZAKASSZAL KAPCSOLATOS KÉRDÉSEK"
- Section B (Q1-11): Műszaki/inverter kérdések (page 6+)

Usage:
    docker exec cc-backend python3 /app/scripts/rechunk_gyik.py [--dry-run]
"""

import asyncio
import hashlib
import json
import re
import sys
from datetime import date

import asyncpg
import fitz

import os
PG_DSN = os.environ.get("HANNA_PG_DSN", "postgresql://klara:klara_docs_2026@cc-db:5432/customercare")
GYIK_PDF = "/app/data/pdfs/OEPT_GYIK_20260204.pdf"
GYIK_DOC_ID = "OEPT_GYIK_20260204.pdf"
DRY_RUN = "--dry-run" in sys.argv


def extract_qa_pairs(pdf_path: str) -> list[dict]:
    """Extract numbered Q&A pairs from GYIK PDF."""
    doc = fitz.open(pdf_path)

    # Extract text page by page to track page boundaries
    page_texts = []
    for page in doc:
        page_texts.append(page.get_text())
    doc.close()

    full_text = "\n".join(page_texts)

    # Find all numbered items with their positions
    # Pattern: newline/start + optional whitespace + number + period + space/newline
    pattern = re.compile(r'(?:^|\n)\s*(\d{1,2})\.\s*\n?\s*(.+?)(?=\n\s*\d{1,2}\.\s|\n\s*[A-ZÁÉÍÓÖŐÚÜŰ]{10,}|\Z)', re.DOTALL)

    matches = list(pattern.finditer(full_text))
    print(f"Found {len(matches)} raw Q&A matches")

    # Detect section boundary: when numbers restart (e.g., 17 → 1)
    qa_pairs = []
    section = "Pályázati szakasszal kapcsolatos kérdések"
    section_letter = "A"
    prev_num = 0

    for m in matches:
        num = int(m.group(1))
        text = m.group(2).strip()

        # Detect section change
        if num <= prev_num and prev_num > 5:
            section = "Műszaki kérdések (inverter, szaldó, DSO)"
            section_letter = "B"

        prev_num = num

        # Clean text: remove headers, extra whitespace
        text = re.sub(r'NEMZETI ENERGETIKAI.*?RÉSZVÉNYTÁRSASÁG', '', text, flags=re.DOTALL)
        text = re.sub(r'\s+', ' ', text).strip()

        # Split into question and answer
        # Question is typically underlined/italic — first sentence(s) ending with ?
        q_match = re.match(r'(.+?\?)\s*(.+)', text, re.DOTALL)
        if q_match:
            question = q_match.group(1).strip()
            answer = q_match.group(2).strip()
        else:
            # No question mark — treat first line as question
            parts = text.split('. ', 1)
            question = parts[0] + '.'
            answer = parts[1] if len(parts) > 1 else text

        global_id = f"{section_letter}{num}"

        qa_pairs.append({
            "global_id": global_id,
            "num": num,
            "section": section,
            "section_letter": section_letter,
            "question": question,
            "answer": answer,
            "full_text": f"GYIK {global_id} — {section}\n\nKérdés: {question}\n\nVálasz: {answer}",
        })

    return qa_pairs


def generate_chunk_id(doc_id: str, global_id: str) -> str:
    raw = f"{doc_id}::gyik_{global_id}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def enrich_qa_chunk(qa: dict) -> str:
    return (
        f"Ez a szövegrész az OETP (Otthoni Energiatároló Program) Gyakran Ismételt "
        f"Kérdések dokumentumának {qa['global_id']}. kérdését és válaszát tartalmazza. "
        f"Szekció: {qa['section']}. "
        f"Kérdés: {qa['question'][:120]}. "
        f"Forrás: {GYIK_DOC_ID}. "
        f"A válasz hivatalos NEÜ állásfoglalás."
    )


async def main():
    print(f"=== GYIK Re-chunking {'(DRY RUN)' if DRY_RUN else ''} ===\n")

    qa_pairs = extract_qa_pairs(GYIK_PDF)
    print(f"\nExtracted {len(qa_pairs)} Q&A pairs:")
    for qa in qa_pairs:
        print(f"  {qa['global_id']:>3}: {qa['question'][:80]}... ({len(qa['full_text'])} chars)")

    if DRY_RUN:
        print("\nDRY RUN — no database changes.")
        return

    pool = await asyncpg.create_pool(PG_DSN, min_size=1, max_size=5)

    # Invalidate old GYIK chunks
    today = date.today().isoformat()
    result = await pool.execute(
        """UPDATE chunks
           SET metadata = metadata || $1::jsonb
           WHERE doc_id = $2 AND doc_type = 'gyik'
             AND (metadata->>'valid_to' IS NULL OR metadata->>'valid_to' = '')""",
        json.dumps({"valid_to": today, "superseded_by": "gyik_rechunked_v2"}),
        GYIK_DOC_ID,
    )
    print(f"\nInvalidated old chunks: {result}")

    # Embed
    print("Embedding new chunks...")
    import httpx
    BGE_URL = "http://host.docker.internal:8114"

    enrichments = [enrich_qa_chunk(qa) for qa in qa_pairs]
    texts_to_embed = [e + "\n\n" + qa["full_text"] for e, qa in zip(enrichments, qa_pairs)]

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{BGE_URL}/embed", json={"texts": texts_to_embed})
        resp.raise_for_status()
        embeddings = resp.json()["embeddings"]
    print(f"Got {len(embeddings)} embeddings")

    # Insert
    inserted = 0
    async with pool.acquire() as conn:
        for qa, enriched, embedding in zip(qa_pairs, enrichments, embeddings):
            chunk_id = generate_chunk_id(GYIK_DOC_ID, qa["global_id"])
            embedding_str = "[" + ",".join(str(float(x)) for x in embedding) + "]"

            await conn.execute(
                """INSERT INTO chunks (
                    id, doc_id, doc_type, program, chunk_index, title,
                    content, content_enriched, embedding, metadata,
                    authority_score, source_date, content_hash
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::vector, $10, $11, $12, $13)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    content_enriched = EXCLUDED.content_enriched,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    updated_at = now()
                """,
                chunk_id,
                GYIK_DOC_ID,
                "gyik",
                "OETP",
                qa["num"],
                f"GYIK {qa['global_id']}: {qa['question'][:80]}",
                qa["full_text"],
                enriched,
                embedding_str,
                json.dumps({
                    "category": "oetp",
                    "chunk_type": "gyik",
                    "version": 2,
                    "supersedes": "",
                    "valid_from": "2026-02-04",
                    "valid_to": "",
                    "indexed_at": today,
                    "qa_number": qa["num"],
                    "global_id": qa["global_id"],
                    "section": qa["section"],
                }),
                0.85,
                None,
                hashlib.sha256(qa["full_text"].encode()).hexdigest()[:16],
            )
            inserted += 1
            print(f"  Inserted {qa['global_id']}: {chunk_id} ({len(qa['full_text'])} chars)")

    # Verify
    new_count = await pool.fetchval(
        "SELECT COUNT(*) FROM chunks WHERE doc_type = 'gyik' AND (metadata->>'valid_to' IS NULL OR metadata->>'valid_to' = '')"
    )
    old_count = await pool.fetchval(
        "SELECT COUNT(*) FROM chunks WHERE doc_type = 'gyik' AND metadata->>'valid_to' IS NOT NULL AND metadata->>'valid_to' != ''"
    )
    print(f"\nDone: {inserted} new Q&A chunks inserted")
    print(f"Active GYIK chunks: {new_count}")
    print(f"Invalidated GYIK chunks: {old_count}")
    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
