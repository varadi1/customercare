#!/usr/bin/env python3
"""
DSPy prompt optimization CLI for CustomerCare.

Usage:
  python scripts/run_dspy_optimization.py --min-pairs 30
  python scripts/run_dspy_optimization.py --dry-run          # Just build trainset, don't optimize
  python scripts/run_dspy_optimization.py --model openai/gpt-4o-mini --max-steps 30
  python scripts/run_dspy_optimization.py --export            # Push result to Langfuse

Requires: pip install dspy>=2.5.0
"""
import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("dspy_optimization")


async def main():
    parser = argparse.ArgumentParser(description="DSPy prompt optimization for CustomerCare")
    parser.add_argument("--min-pairs", type=int, default=30, help="Minimum training pairs required")
    parser.add_argument("--max-pairs", type=int, default=200, help="Maximum training pairs to use")
    parser.add_argument("--days", type=int, default=90, help="Look-back period in days")
    parser.add_argument("--model", type=str, default="openai/gpt-4o-mini", help="LLM for optimization")
    parser.add_argument("--max-steps", type=int, default=50, help="Max optimization steps")
    parser.add_argument("--num-candidates", type=int, default=5, help="Number of prompt candidates")
    parser.add_argument("--dry-run", action="store_true", help="Only build trainset, don't optimize")
    parser.add_argument("--export", action="store_true", help="Push optimized prompt to Langfuse")
    parser.add_argument("--output", type=str, default="", help="Save result to JSON file")
    args = parser.parse_args()

    from app.reasoning.dspy_optimizer import build_trainset, run_optimization, export_to_langfuse

    # Step 1: Build training data
    logger.info("Building training data (min=%d, max=%d, days=%d)...", args.min_pairs, args.max_pairs, args.days)
    trainset = await build_trainset(
        min_pairs=args.min_pairs,
        max_pairs=args.max_pairs,
        days=args.days,
    )

    if not trainset:
        logger.error("Not enough training pairs. Wait for more feedback data.")
        sys.exit(1)

    logger.info("Built %d training examples", len(trainset))

    if args.dry_run:
        logger.info("Dry run — showing sample examples:")
        for i, ex in enumerate(trainset[:3]):
            logger.info(
                "  Example %d: subject=%s, category=%s, confidence=%s, body_len=%d",
                i + 1,
                ex.email_subject[:50],
                ex.category,
                ex.confidence,
                len(ex.body),
            )
        logger.info("Dry run complete. Use without --dry-run to optimize.")
        return

    # Step 2: Run optimization
    logger.info("Starting DSPy MIPROv2 optimization (model=%s, steps=%d, candidates=%d)...",
                args.model, args.max_steps, args.num_candidates)

    result = await run_optimization(
        trainset=trainset,
        model=args.model,
        max_steps=args.max_steps,
        num_candidates=args.num_candidates,
    )

    # Step 3: Report results
    logger.info("=" * 60)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("=" * 60)
    logger.info("  Baseline score:  %.4f", result["baseline_score"])
    logger.info("  Optimized score: %.4f", result["optimized_score"])
    logger.info("  Improvement:     %+.4f", result["improvement"])
    logger.info("  Training size:   %d", result["train_size"])
    logger.info("  Validation size: %d", result["val_size"])
    logger.info("")

    if result["optimized_prompt"]:
        logger.info("Optimized prompt (first 500 chars):")
        logger.info(result["optimized_prompt"][:500])
    else:
        logger.warning("Could not extract optimized prompt text")

    # Step 4: Export to Langfuse (optional)
    if args.export and result["optimized_prompt"]:
        if result["improvement"] > 0:
            success = export_to_langfuse(result["optimized_prompt"])
            if success:
                logger.info("Optimized prompt exported to Langfuse!")
            else:
                logger.error("Failed to export to Langfuse")
        else:
            logger.warning("No improvement — skipping Langfuse export")

    # Step 5: Save result (optional)
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Result saved to %s", output_path)


if __name__ == "__main__":
    asyncio.run(main())
