"""
Reasoning trace CRUD — persistent storage for query→response→outcome learning.

Each email interaction creates a trace:
  query_text → draft_text → [human review] → sent_text → outcome

Outcomes:
  PENDING      — draft created, not yet sent
  SENT_AS_IS   — similarity >= 0.85
  SENT_MODIFIED — similarity 0.30-0.85
  REJECTED     — similarity < 0.30
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import asyncpg

logger = logging.getLogger(__name__)

# Outcome thresholds (same as feedback.py)
SENT_AS_IS_THRESHOLD = 0.85
MODIFIED_THRESHOLD = 0.30


async def create_trace(
    conn: asyncpg.Connection,
    query_text: str,
    category: str,
    email_message_id: Optional[str] = None,
    sender_name: Optional[str] = None,
    sender_email: Optional[str] = None,
    program: str = "OETP",
    phases: Optional[List[str]] = None,
    confidence: Optional[str] = None,
    draft_text: Optional[str] = None,
    top_chunks: Optional[List[Dict[str, Any]]] = None,
    query_embedding: Optional[List[float]] = None,
) -> int:
    """Create a new reasoning trace. Returns the trace ID."""
    chunks_json = json.dumps(top_chunks, ensure_ascii=False) if top_chunks else None
    embedding_str = _format_vector(query_embedding) if query_embedding else None

    row = await conn.fetchrow(
        """
        INSERT INTO reasoning_traces
            (query_text, query_embedding, email_message_id,
             sender_name, sender_email, category, program, phases,
             confidence, draft_text, top_chunks)
        VALUES
            ($1, $2::vector, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb)
        RETURNING id
        """,
        query_text,
        embedding_str,
        email_message_id,
        sender_name,
        sender_email,
        category,
        program,
        phases,
        confidence,
        draft_text,
        chunks_json,
    )
    trace_id = row["id"]
    logger.info("Created reasoning trace %d (category=%s, program=%s)", trace_id, category, program)
    return trace_id


async def resolve_trace(
    conn: asyncpg.Connection,
    trace_id: int,
    sent_text: str,
    similarity_score: float,
) -> str:
    """Resolve a trace with the actual sent text and similarity score.

    Returns the determined outcome.
    """
    outcome = _classify_outcome(similarity_score)

    await conn.execute(
        """
        UPDATE reasoning_traces
        SET sent_text = $1,
            similarity_score = $2,
            outcome = $3,
            resolved_at = NOW()
        WHERE id = $4
        """,
        sent_text,
        similarity_score,
        outcome,
        trace_id,
    )
    logger.info("Resolved trace %d: outcome=%s (similarity=%.3f)", trace_id, outcome, similarity_score)
    return outcome


async def find_similar_traces(
    conn: asyncpg.Connection,
    query_embedding: List[float],
    limit: int = 3,
    program: Optional[str] = None,
    min_similarity: float = 0.5,
) -> List[Dict[str, Any]]:
    """Find similar past traces using pgvector cosine similarity.

    Only returns resolved traces (not PENDING).
    Prefers SENT_AS_IS outcomes.
    """
    embedding_str = _format_vector(query_embedding)

    query = """
        SELECT id, query_text, category, program, confidence,
               draft_text, sent_text, outcome, similarity_score,
               top_chunks, created_at,
               1 - (query_embedding <=> $1::vector) AS cosine_sim
        FROM reasoning_traces
        WHERE outcome != 'PENDING'
          AND query_embedding IS NOT NULL
    """
    params = [embedding_str]
    param_idx = 2

    if program:
        query += f" AND program = ${param_idx}"
        params.append(program)
        param_idx += 1

    query += f"""
        HAVING 1 - (query_embedding <=> $1::vector) > ${param_idx}
        ORDER BY
            CASE outcome
                WHEN 'SENT_AS_IS' THEN 0
                WHEN 'SENT_MODIFIED' THEN 1
                WHEN 'REJECTED' THEN 2
            END,
            cosine_sim DESC
        LIMIT ${param_idx + 1}
    """
    params.extend([min_similarity, limit])

    # PostgreSQL doesn't support HAVING without GROUP BY on non-aggregates
    # Rewrite as subquery
    full_query = f"""
        SELECT * FROM (
            SELECT id, query_text, category, program, confidence,
                   draft_text, sent_text, outcome, similarity_score,
                   top_chunks::text, created_at,
                   1 - (query_embedding <=> $1::vector) AS cosine_sim
            FROM reasoning_traces
            WHERE outcome != 'PENDING'
              AND query_embedding IS NOT NULL
              {"AND program = $2" if program else ""}
        ) sub
        WHERE cosine_sim > ${param_idx}
        ORDER BY
            CASE outcome
                WHEN 'SENT_AS_IS' THEN 0
                WHEN 'SENT_MODIFIED' THEN 1
                WHEN 'REJECTED' THEN 2
            END,
            cosine_sim DESC
        LIMIT ${param_idx + 1}
    """

    # Rebuild clean params
    clean_params = [embedding_str]
    if program:
        clean_params.append(program)
    clean_params.extend([min_similarity, limit])

    rows = await conn.fetch(full_query, *clean_params)

    results = []
    for row in rows:
        results.append({
            "id": row["id"],
            "query_text": row["query_text"],
            "category": row["category"],
            "program": row["program"],
            "confidence": row["confidence"],
            "draft_text": row["draft_text"],
            "sent_text": row["sent_text"],
            "outcome": row["outcome"],
            "similarity_score": row["similarity_score"],
            "cosine_sim": row["cosine_sim"],
            "created_at": str(row["created_at"]),
        })

    return results


def _classify_outcome(similarity_score: float) -> str:
    """Classify outcome based on draft vs sent similarity."""
    if similarity_score >= SENT_AS_IS_THRESHOLD:
        return "SENT_AS_IS"
    elif similarity_score >= MODIFIED_THRESHOLD:
        return "SENT_MODIFIED"
    else:
        return "REJECTED"


def _format_vector(embedding: List[float]) -> str:
    """Format embedding list as pgvector string '[0.1,0.2,...]'."""
    return "[" + ",".join(str(x) for x in embedding) + "]"
