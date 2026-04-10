#!/usr/bin/env python3
"""
Evaluate base vs fine-tuned reranker on CustomerCare golden set.

Usage:
  python scripts/eval_reranker.py
  python scripts/eval_reranker.py --finetuned-url http://localhost:8112
  python scripts/eval_reranker.py --eval-data /app/data/reranker_train_eval.jsonl

Metrics: NDCG@5, MRR, average score difference.
"""
import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("eval_reranker")

BASE_RERANKER_URL = "http://host.docker.internal:8102"


async def rerank_batch(
    query: str,
    documents: list[str],
    url: str = BASE_RERANKER_URL,
) -> list[float]:
    """Call reranker API and return scores."""
    import httpx

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{url}/rerank",
            json={"query": query, "documents": documents, "top_n": len(documents)},
        )
        resp.raise_for_status()
        data = resp.json()
        # Sort by index to get scores in document order
        results = sorted(data.get("results", []), key=lambda x: x.get("index", 0))
        return [r.get("relevance_score", 0) for r in results]


def ndcg_at_k(scores: list[float], labels: list[int], k: int = 5) -> float:
    """Compute NDCG@k."""
    import math

    # Sort by score descending, take top k
    paired = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)[:k]

    dcg = sum(label / math.log2(i + 2) for i, (_, label) in enumerate(paired))

    # Ideal DCG
    ideal_labels = sorted(labels, reverse=True)[:k]
    idcg = sum(label / math.log2(i + 2) for i, label in enumerate(ideal_labels))

    return dcg / idcg if idcg > 0 else 0


def mrr(scores: list[float], labels: list[int]) -> float:
    """Compute Mean Reciprocal Rank."""
    paired = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    for i, (_, label) in enumerate(paired):
        if label == 1:
            return 1.0 / (i + 1)
    return 0.0


async def evaluate_from_jsonl(
    eval_path: str,
    base_url: str = BASE_RERANKER_URL,
    finetuned_url: str = "",
) -> dict:
    """Evaluate reranker(s) on JSONL eval data."""
    data = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    if not data:
        logger.error("No eval data found")
        return {}

    logger.info("Evaluating on %d examples...", len(data))

    base_ndcg_scores = []
    base_mrr_scores = []
    ft_ndcg_scores = []
    ft_mrr_scores = []

    for item in data:
        query = item["query"]
        docs = [item["positive"], item["negative"]]
        labels = [1, 0]

        # Base model
        try:
            base_scores = await rerank_batch(query, docs, base_url)
            base_ndcg_scores.append(ndcg_at_k(base_scores, labels, k=2))
            base_mrr_scores.append(mrr(base_scores, labels))
        except Exception as e:
            logger.warning("Base reranker failed: %s", e)

        # Fine-tuned model (if URL provided)
        if finetuned_url:
            try:
                ft_scores = await rerank_batch(query, docs, finetuned_url)
                ft_ndcg_scores.append(ndcg_at_k(ft_scores, labels, k=2))
                ft_mrr_scores.append(mrr(ft_scores, labels))
            except Exception as e:
                logger.warning("Fine-tuned reranker failed: %s", e)

    result = {
        "eval_count": len(data),
        "base": {
            "ndcg@k": round(sum(base_ndcg_scores) / len(base_ndcg_scores), 4) if base_ndcg_scores else 0,
            "mrr": round(sum(base_mrr_scores) / len(base_mrr_scores), 4) if base_mrr_scores else 0,
            "evaluated": len(base_ndcg_scores),
        },
    }

    if finetuned_url and ft_ndcg_scores:
        ft_ndcg = sum(ft_ndcg_scores) / len(ft_ndcg_scores)
        ft_mrr_val = sum(ft_mrr_scores) / len(ft_mrr_scores)
        base_ndcg = result["base"]["ndcg@k"]
        base_mrr_val = result["base"]["mrr"]

        result["finetuned"] = {
            "ndcg@k": round(ft_ndcg, 4),
            "mrr": round(ft_mrr_val, 4),
            "evaluated": len(ft_ndcg_scores),
        }
        result["improvement"] = {
            "ndcg@k": round(ft_ndcg - base_ndcg, 4),
            "mrr": round(ft_mrr_val - base_mrr_val, 4),
        }

    return result


async def main():
    parser = argparse.ArgumentParser(description="Evaluate reranker models")
    parser.add_argument("--eval-data", type=str, default="/app/data/reranker_train_eval.jsonl")
    parser.add_argument("--base-url", type=str, default=BASE_RERANKER_URL)
    parser.add_argument("--finetuned-url", type=str, default="", help="Fine-tuned reranker URL")
    parser.add_argument("--output", type=str, default="", help="Save results to JSON")
    args = parser.parse_args()

    if not Path(args.eval_data).exists():
        logger.error("Eval data not found: %s", args.eval_data)
        logger.info("Run build_reranker_training_data.py first to generate eval data")
        sys.exit(1)

    result = await evaluate_from_jsonl(
        eval_path=args.eval_data,
        base_url=args.base_url,
        finetuned_url=args.finetuned_url,
    )

    logger.info("=" * 60)
    logger.info("RERANKER EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info("Eval examples: %d", result.get("eval_count", 0))
    logger.info("")
    logger.info("BASE MODEL (%s):", args.base_url)
    base = result.get("base", {})
    logger.info("  NDCG@k: %.4f", base.get("ndcg@k", 0))
    logger.info("  MRR:    %.4f", base.get("mrr", 0))

    if "finetuned" in result:
        ft = result["finetuned"]
        imp = result.get("improvement", {})
        logger.info("")
        logger.info("FINE-TUNED MODEL (%s):", args.finetuned_url)
        logger.info("  NDCG@k: %.4f (%+.4f)", ft.get("ndcg@k", 0), imp.get("ndcg@k", 0))
        logger.info("  MRR:    %.4f (%+.4f)", ft.get("mrr", 0), imp.get("mrr", 0))

        if imp.get("ndcg@k", 0) > 0.02:
            logger.info("")
            logger.info("RECOMMENDATION: Fine-tuned model shows >2%% improvement. Deploy recommended.")
        elif imp.get("ndcg@k", 0) > 0:
            logger.info("")
            logger.info("RECOMMENDATION: Marginal improvement. Consider more training data.")
        else:
            logger.info("")
            logger.info("RECOMMENDATION: No improvement. Keep base model.")

    if args.output:
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    asyncio.run(main())
