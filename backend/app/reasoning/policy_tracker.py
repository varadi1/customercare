"""
Policy traceability — tracks which rules/documents back each answer.

When a new document is ingested that supersedes an older one,
marks old chunks as invalidated and flags affected reasoning traces.

Uses the existing chunks.metadata JSONB fields:
  - valid_from: when the chunk became effective
  - valid_to: when superseded (NULL = still valid)
  - supersedes: doc_id of the document this replaces
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Optional

import asyncpg

logger = logging.getLogger(__name__)

import os
PG_DSN = os.environ.get("CC_PG_DSN", "postgresql://klara:klara_docs_2026@cc-db:5432/customercare")


async def invalidate_superseded_chunks(
    conn: asyncpg.Connection,
    new_doc_id: str,
    superseded_doc_id: str,
) -> dict[str, Any]:
    """Mark chunks from superseded document as no longer valid.

    Sets valid_to in metadata JSONB for all chunks of the old document.
    Returns count of invalidated chunks and affected traces.
    """
    now_iso = datetime.utcnow().isoformat()

    # Mark old chunks
    result = await conn.execute(
        """
        UPDATE chunks
        SET metadata = jsonb_set(
            COALESCE(metadata, '{}'::jsonb),
            '{valid_to}',
            to_jsonb($1::text)
        )
        WHERE doc_id = $2
          AND (metadata->>'valid_to') IS NULL
        """,
        now_iso,
        superseded_doc_id,
    )

    invalidated = int(result.split()[-1]) if result else 0

    # Find affected reasoning traces (that used chunks from the superseded doc)
    affected_traces = await conn.fetch(
        """
        SELECT id, query_text, top_chunks, outcome, created_at
        FROM reasoning_traces
        WHERE outcome IN ('SENT_AS_IS', 'SENT_MODIFIED')
          AND top_chunks::text LIKE $1
        """,
        f"%{superseded_doc_id}%",
    )

    # Flag affected traces
    if affected_traces:
        for trace in affected_traces:
            await conn.execute(
                """
                UPDATE reasoning_traces
                SET rag_scores = jsonb_set(
                    COALESCE(rag_scores, '{}'::jsonb),
                    '{superseded_warning}',
                    $1::jsonb
                )
                WHERE id = $2
                """,
                json.dumps({
                    "superseded_doc": superseded_doc_id,
                    "new_doc": new_doc_id,
                    "flagged_at": now_iso,
                }),
                trace["id"],
            )

    logger.info(
        "Invalidated %d chunks from doc %s (superseded by %s), %d traces affected",
        invalidated, superseded_doc_id, new_doc_id, len(affected_traces),
    )

    return {
        "invalidated_chunks": invalidated,
        "affected_traces": len(affected_traces),
        "superseded_doc": superseded_doc_id,
        "new_doc": new_doc_id,
    }


async def check_answer_validity(
    conn: asyncpg.Connection,
    chunk_ids: list[str],
) -> dict[str, Any]:
    """Check if chunks used in an answer are still valid.

    Returns validity status for each chunk.
    """
    if not chunk_ids:
        return {"all_valid": True, "chunks": []}

    rows = await conn.fetch(
        """
        SELECT id, doc_id,
               metadata->>'valid_to' as valid_to,
               metadata->>'valid_from' as valid_from,
               metadata->>'supersedes' as supersedes
        FROM chunks
        WHERE id = ANY($1)
        """,
        chunk_ids,
    )

    chunks_status = []
    any_invalid = False

    for row in rows:
        valid_to = row["valid_to"]
        is_valid = valid_to is None
        if not is_valid:
            any_invalid = True

        chunks_status.append({
            "chunk_id": row["id"],
            "doc_id": row["doc_id"],
            "valid": is_valid,
            "valid_from": row["valid_from"],
            "valid_to": valid_to,
        })

    return {
        "all_valid": not any_invalid,
        "chunks": chunks_status,
    }


async def get_supersession_chain(
    conn: asyncpg.Connection,
    doc_id: str,
) -> list[dict]:
    """Get the chain of document versions (current → previous → ...).

    Follows the supersedes metadata field.
    """
    chain = []
    current = doc_id
    seen = set()

    while current and current not in seen:
        seen.add(current)
        row = await conn.fetchrow(
            """
            SELECT DISTINCT doc_id,
                   MIN(metadata->>'valid_from') as valid_from,
                   MAX(metadata->>'valid_to') as valid_to,
                   MIN(metadata->>'supersedes') as supersedes
            FROM chunks
            WHERE doc_id = $1
            GROUP BY doc_id
            """,
            current,
        )

        if not row:
            break

        chain.append({
            "doc_id": row["doc_id"],
            "valid_from": row["valid_from"],
            "valid_to": row["valid_to"],
            "supersedes": row["supersedes"],
        })

        current = row["supersedes"]

    return chain
