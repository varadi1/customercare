#!/usr/bin/env python3
"""
Fine-tune BGE reranker v2-m3 on Hanna's chunk survival data.

Usage:
  python scripts/finetune_reranker.py --train-data /app/data/reranker_train.jsonl
  python scripts/finetune_reranker.py --train-data train.jsonl --eval-data eval.jsonl --epochs 3

Requires: pip install sentence-transformers FlagEmbedding (see requirements-finetune.txt)

Runs on macOS with MPS GPU acceleration.
Output model saved to /app/data/models/reranker-finetuned/
"""
import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("finetune_reranker")

BASE_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_OUTPUT = "/app/data/models/reranker-finetuned"


def load_training_data(path: str) -> list[dict]:
    """Load JSONL training data."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    logger.info("Loaded %d training examples from %s", len(data), path)
    return data


def finetune(
    train_data: list[dict],
    eval_data: list[dict] | None = None,
    base_model: str = BASE_MODEL,
    output_dir: str = DEFAULT_OUTPUT,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
):
    """Fine-tune the BGE reranker using sentence-transformers CrossEncoder.

    Training data format: {"query": str, "positive": str, "negative": str}
    """
    try:
        from sentence_transformers import CrossEncoder
        from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
    except ImportError:
        logger.error("sentence-transformers not installed. Run: pip install sentence-transformers>=3.0.0")
        sys.exit(1)

    import torch

    # Check for MPS (Apple Silicon GPU)
    if torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = "cuda"
        logger.info("Using CUDA GPU")
    else:
        device = "cpu"
        logger.info("Using CPU (this will be slow)")

    # Load base model
    logger.info("Loading base model: %s", base_model)
    model = CrossEncoder(base_model, device=device)

    # Prepare training data: list of (query, passage, label)
    train_samples = []
    for item in train_data:
        # Positive pair (label=1)
        train_samples.append({
            "query": item["query"],
            "passage": item["positive"],
            "label": 1,
        })
        # Negative pair (label=0)
        train_samples.append({
            "query": item["query"],
            "passage": item["negative"],
            "label": 0,
        })

    logger.info("Training samples: %d (%d positive, %d negative)",
                len(train_samples), len(train_data), len(train_data))

    # Prepare evaluation data
    evaluator = None
    if eval_data:
        eval_samples = {}
        for i, item in enumerate(eval_data):
            eval_samples[f"q{i}"] = {
                "query": item["query"],
                "positive": [item["positive"]],
                "negative": [item["negative"]],
            }
        evaluator = CERerankingEvaluator(eval_samples, name="hanna-eval")
        logger.info("Evaluation samples: %d", len(eval_data))

    # Fine-tune
    logger.info("Starting fine-tuning: epochs=%d, batch=%d, lr=%.1e", epochs, batch_size, learning_rate)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    from sentence_transformers.cross_encoder import CrossEncoderModelCardData

    model.fit(
        train_dataloader=train_samples,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        evaluator=evaluator,
        evaluation_steps=max(100, len(train_samples) // batch_size // 2),
        output_path=str(output_path),
        save_best_model=True if evaluator else False,
    )

    # Save final model
    model.save(str(output_path))
    logger.info("Fine-tuned model saved to %s", output_path)

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BGE reranker on Hanna data")
    parser.add_argument("--train-data", type=str, required=True, help="Training JSONL file")
    parser.add_argument("--eval-data", type=str, default="", help="Evaluation JSONL file")
    parser.add_argument("--base-model", type=str, default=BASE_MODEL)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    args = parser.parse_args()

    train_data = load_training_data(args.train_data)
    eval_data = load_training_data(args.eval_data) if args.eval_data else None

    if len(train_data) < 10:
        logger.error("Too few training examples (%d). Need at least 10.", len(train_data))
        sys.exit(1)

    finetune(
        train_data=train_data,
        eval_data=eval_data,
        base_model=args.base_model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
