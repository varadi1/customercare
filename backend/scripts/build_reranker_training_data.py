#!/usr/bin/env python3
"""
Build training data for BGE reranker fine-tuning from chunk survival feedback.

Usage:
  python scripts/build_reranker_training_data.py --output /app/data/reranker_train.jsonl
  python scripts/build_reranker_training_data.py --min-appearances 3 --stats

Output format (JSONL, one per line):
  {"query": "...", "positive": "...", "negative": "..."}

Positive = chunk that survived (human kept it in sent email)
Negative = chunk that did NOT survive (human removed it)
"""
import argparse
import asyncio
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("build_reranker_data")

import os
PG_DSN = os.environ.get(
    "HANNA_PG_DSN",
    "postgresql://klara:klara_docs_2026@cc-db:5432/customercare",
)


async def build_training_pairs(
    min_appearances: int = 3,
    days: int = 90,
) -> list[dict]:
    """Build query-positive-negative triples from chunk survival data.

    A trace with top_chunks gives us:
      - query = the email text
      - positive = chunk that survived (high sent_overlap)
      - negative = chunk that did NOT survive (low sent_overlap)

    Only uses chunks that appear in min_appearances+ traces.
    """
    import asyncpg
    from datetime import datetime, timedelta

    conn = await asyncpg.connect(PG_DSN)
    try:
        since = datetime.utcnow() - timedelta(days=days)

        # Get all feedback analytics with chunk survival
        rows = await conn.fetch(
            """
            SELECT fa.chunk_survival, rt.query_text, rt.category
            FROM feedback_analytics fa
            JOIN reasoning_traces rt ON fa.trace_id = rt.id
            WHERE fa.chunk_survival IS NOT NULL
              AND fa.chunk_survival != '[]'::jsonb
              AND fa.created_at >= $1
            """,
            since,
        )

        if not rows:
            logger.warning("No chunk survival data found")
            return []

        # Count per-chunk appearances
        chunk_appearances: dict[str, int] = defaultdict(int)
        for row in rows:
            survival = row["chunk_survival"]
            if isinstance(survival, str):
                survival = json.loads(survival)
            for entry in survival:
                cid = entry.get("chunk_id", "")
                if cid:
                    chunk_appearances[cid] += 1

        # Filter to chunks with enough appearances
        valid_chunks = {cid for cid, count in chunk_appearances.items() if count >= min_appearances}
        logger.info("Chunks with %d+ appearances: %d / %d total",
                    min_appearances, len(valid_chunks), len(chunk_appearances))

        # Fetch chunk texts
        if not valid_chunks:
            return []

        chunk_texts = {}
        chunk_rows = await conn.fetch(
            "SELECT id, content FROM chunks WHERE id = ANY($1)",
            list(valid_chunks),
        )
        chunk_texts = {r["id"]: r["content"] or "" for r in chunk_rows}

        # Build triples
        pairs = []
        for row in rows:
            query = row["query_text"] or ""
            if len(query) < 20:
                continue

            survival = row["chunk_survival"]
            if isinstance(survival, str):
                survival = json.loads(survival)

            survived = []
            not_survived = []

            for entry in survival:
                cid = entry.get("chunk_id", "")
                if cid not in valid_chunks or cid not in chunk_texts:
                    continue
                text = chunk_texts[cid]
                if not text or len(text) < 20:
                    continue

                if entry.get("survived"):
                    survived.append(text)
                else:
                    not_survived.append(text)

            # Create pairs: each survived + each not_survived
            for pos in survived:
                for neg in not_survived:
                    pairs.append({
                        "query": query[:1000],
                        "positive": pos[:1000],
                        "negative": neg[:1000],
                    })

        logger.info("Built %d training triples from %d traces", len(pairs), len(rows))
        return pairs

    finally:
        await conn.close()


def export_to_jsonl(pairs: list[dict], output_path: str) -> int:
    """Export training pairs to JSONL format for BGE reranker fine-tuning."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    logger.info("Exported %d pairs to %s", len(pairs), path)
    return len(pairs)


def split_train_eval(pairs: list[dict], eval_ratio: float = 0.2) -> tuple[list, list]:
    """Split pairs into training and evaluation sets."""
    import random
    random.seed(42)
    shuffled = pairs.copy()
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - eval_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]


async def print_stats(days: int = 90, min_appearances: int = 3):
    """Print statistics about available training data."""
    import asyncpg
    from datetime import datetime, timedelta

    conn = await asyncpg.connect(PG_DSN)
    try:
        since = datetime.utcnow() - timedelta(days=days)

        total = await conn.fetchval(
            "SELECT COUNT(*) FROM feedback_analytics WHERE created_at >= $1", since
        )
        with_survival = await conn.fetchval(
            """SELECT COUNT(*) FROM feedback_analytics
               WHERE chunk_survival IS NOT NULL AND chunk_survival != '[]'::jsonb
               AND created_at >= $1""",
            since,
        )

        logger.info("=" * 50)
        logger.info("RERANKER TRAINING DATA STATISTICS")
        logger.info("=" * 50)
        logger.info("Period: %d days", days)
        logger.info("Total feedback analytics records: %d", total)
        logger.info("Records with chunk survival data: %d", with_survival)
        logger.info("Min appearances threshold: %d", min_appearances)

        if with_survival > 0:
            pairs = await build_training_pairs(min_appearances=min_appearances, days=days)
            train, eval_set = split_train_eval(pairs)
            logger.info("Training triples: %d (train=%d, eval=%d)", len(pairs), len(train), len(eval_set))

            if len(pairs) < 50:
                logger.warning("Recommended minimum: 50 triples. Need more feedback data.")
            else:
                logger.info("Sufficient data for fine-tuning!")

    finally:
        await conn.close()


async def main():
    parser = argparse.ArgumentParser(description="Build reranker training data")
    parser.add_argument("--output", type=str, default="/app/data/reranker_train.jsonl")
    parser.add_argument("--min-appearances", type=int, default=3)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--stats", action="store_true", help="Print stats only")
    args = parser.parse_args()

    if args.stats:
        await print_stats(days=args.days, min_appearances=args.min_appearances)
        return

    pairs = await build_training_pairs(
        min_appearances=args.min_appearances,
        days=args.days,
    )

    if not pairs:
        logger.error("No training pairs generated. Exiting.")
        sys.exit(1)

    train, eval_set = split_train_eval(pairs, eval_ratio=args.eval_ratio)

    train_path = args.output
    eval_path = train_path.replace(".jsonl", "_eval.jsonl")

    export_to_jsonl(train, train_path)
    export_to_jsonl(eval_set, eval_path)

    logger.info("Done! Train: %s (%d), Eval: %s (%d)",
                train_path, len(train), eval_path, len(eval_set))


if __name__ == "__main__":
    asyncio.run(main())
