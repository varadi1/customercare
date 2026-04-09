"""
DSPy prompt optimization for Hanna draft generation.

Level 3 of the learning system:
  - HannaDraftSignature — DSPy Signature for email draft generation
  - build_trainset() — Create training data from reasoning_traces
  - run_optimization() — MIPROv2 optimizer to find better prompts
  - export_to_langfuse() — Push optimized prompt for production use

Usage:
  python scripts/run_dspy_optimization.py --min-pairs 30 [--dry-run]

Requires: pip install dspy>=2.5.0
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

PG_DSN = os.environ.get(
    "HANNA_PG_DSN",
    "postgresql://klara:klara_docs_2026@cc-db:5432/customercare",
)


# ─── DSPy Module ─────────────────────────────────────────────────────────────

def _get_dspy():
    """Lazy import of dspy to avoid import errors when not installed."""
    try:
        import dspy
        return dspy
    except ImportError:
        raise ImportError("DSPy not installed. Run: pip install dspy>=2.5.0")


def create_draft_module():
    """Create a DSPy module for Hanna draft generation.

    Returns a configured dspy.Module instance.
    """
    dspy = _get_dspy()

    class HannaDraftSignature(dspy.Signature):
        """Generate a customer service email reply in Hungarian for the OETP program.
        Use ONLY the provided facts. Never invent dates, amounts, or deadlines.
        Reply in 2-4 sentences, formal but friendly tone (magázás).
        If facts are insufficient, set confidence to 'skip'.
        """
        email_text: str = dspy.InputField(desc="The customer's email text (Hungarian)")
        email_subject: str = dspy.InputField(desc="Email subject line")
        facts: str = dspy.InputField(desc="Verified facts from the knowledge base (numbered)")
        category: str = dspy.InputField(desc="Email category (e.g., inverter, szaldo, palyazat)")

        body: str = dspy.OutputField(desc="HTML reply body (2-4 sentences, Hungarian, magázás)")
        confidence: str = dspy.OutputField(desc="high, medium, or skip")

    class HannaDraftModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.generate = dspy.ChainOfThought(HannaDraftSignature)

        def forward(self, email_text, email_subject, facts, category):
            return self.generate(
                email_text=email_text,
                email_subject=email_subject,
                facts=facts,
                category=category,
            )

    return HannaDraftModule()


# ─── Training Data ───────────────────────────────────────────────────────────

async def build_trainset(
    min_pairs: int = 30,
    max_pairs: int = 200,
    days: int = 90,
) -> list:
    """Build DSPy training examples from reasoning_traces.

    Selects resolved traces (SENT_AS_IS, SENT_MODIFIED) where we have
    both draft_text and sent_text.

    Returns list of dspy.Example objects.
    """
    dspy = _get_dspy()
    import asyncpg
    from datetime import datetime, timedelta

    conn = await asyncpg.connect(PG_DSN)
    try:
        since = datetime.utcnow() - timedelta(days=days)

        rows = await conn.fetch(
            """
            SELECT rt.query_text, rt.category, rt.draft_text, rt.sent_text,
                   rt.top_chunks, rt.confidence, rt.similarity_score
            FROM reasoning_traces rt
            WHERE rt.outcome IN ('SENT_AS_IS', 'SENT_MODIFIED')
              AND rt.sent_text IS NOT NULL
              AND LENGTH(rt.sent_text) > 20
              AND rt.draft_text IS NOT NULL
              AND rt.created_at >= $1
            ORDER BY rt.similarity_score DESC
            LIMIT $2
            """,
            since,
            max_pairs,
        )

        if len(rows) < min_pairs:
            logger.warning(
                "Not enough training pairs: %d (need %d). Wait for more feedback data.",
                len(rows), min_pairs,
            )
            return []

        examples = []
        for row in rows:
            # Build facts string from top_chunks
            chunks = row["top_chunks"]
            if isinstance(chunks, str):
                try:
                    chunks = json.loads(chunks)
                except (json.JSONDecodeError, TypeError):
                    chunks = []

            # Fetch actual chunk texts for the facts
            chunk_ids = [c.get("id", "") for c in (chunks or []) if c.get("id")]
            facts_text = ""
            if chunk_ids:
                chunk_rows = await conn.fetch(
                    "SELECT id, content FROM chunks WHERE id = ANY($1)",
                    chunk_ids,
                )
                facts_parts = []
                for i, cr in enumerate(chunk_rows, 1):
                    facts_parts.append(f"[{i}] {(cr['content'] or '')[:400]}")
                facts_text = "\n".join(facts_parts)

            # Extract subject from query_text (first line or first 80 chars)
            query = row["query_text"] or ""
            subject = query.split("\n")[0][:80] if "\n" in query else query[:80]

            example = dspy.Example(
                email_text=query[:2000],
                email_subject=subject,
                facts=facts_text[:3000],
                category=row["category"] or "altalanos",
                body=row["sent_text"][:2000],  # Gold label = what human actually sent
                confidence=row["confidence"] or "medium",
            ).with_inputs("email_text", "email_subject", "facts", "category")

            examples.append(example)

        logger.info("Built %d training examples for DSPy optimization", len(examples))
        return examples

    finally:
        await conn.close()


# ─── Metric ──────────────────────────────────────────────────────────────────

def draft_quality_metric(example, prediction, trace=None) -> float:
    """Metric for DSPy optimization: how close is the prediction to the gold (sent) version.

    Combines:
      - Semantic similarity (word overlap, cheap proxy)
      - Length appropriateness (not too short, not too long)
      - Confidence match

    Returns 0-1 score.
    """
    import re

    gold = example.body or ""
    pred = getattr(prediction, "body", "") or ""

    if not pred or len(pred) < 10:
        return 0.0

    # Word overlap (Jaccard-like)
    def _words(text):
        return set(re.findall(r"[a-záéíóöőúüű]{3,}", text.lower()))

    gold_words = _words(gold)
    pred_words = _words(pred)

    if not gold_words or not pred_words:
        return 0.1

    intersection = gold_words & pred_words
    union = gold_words | pred_words
    word_sim = len(intersection) / len(union) if union else 0

    # Length ratio penalty
    gold_len = len(gold)
    pred_len = len(pred)
    if gold_len > 0:
        ratio = pred_len / gold_len
        length_score = 1.0 - min(abs(1.0 - ratio), 1.0) * 0.5
    else:
        length_score = 0.5

    # Confidence match bonus
    gold_conf = example.confidence or "medium"
    pred_conf = getattr(prediction, "confidence", "medium") or "medium"
    conf_bonus = 0.1 if gold_conf == pred_conf else 0.0

    score = word_sim * 0.6 + length_score * 0.3 + conf_bonus
    return min(1.0, max(0.0, score))


# ─── Optimization ────────────────────────────────────────────────────────────

async def run_optimization(
    trainset: list,
    model: str = "openai/gpt-4o-mini",
    max_steps: int = 50,
    num_candidates: int = 5,
) -> dict[str, Any]:
    """Run DSPy MIPROv2 optimization on the draft generation module.

    Args:
        trainset: List of dspy.Example objects from build_trainset()
        model: LLM model for optimization (use a cheap model)
        max_steps: Maximum optimization iterations
        num_candidates: Number of prompt candidates to try

    Returns:
        {
            "optimized_prompt": str,
            "baseline_score": float,
            "optimized_score": float,
            "improvement": float,
            "num_examples": int,
        }
    """
    dspy = _get_dspy()

    # Configure DSPy LM
    lm = dspy.LM(model)
    dspy.configure(lm=lm)

    module = create_draft_module()

    # Split trainset
    split = int(len(trainset) * 0.8)
    train = trainset[:split]
    val = trainset[split:]

    if len(val) < 5:
        val = train[-5:]

    # Baseline evaluation
    baseline_scores = []
    for ex in val[:20]:
        try:
            pred = module(**{k: getattr(ex, k) for k in ["email_text", "email_subject", "facts", "category"]})
            score = draft_quality_metric(ex, pred)
            baseline_scores.append(score)
        except Exception:
            baseline_scores.append(0.0)

    baseline_avg = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0

    # Optimize with MIPROv2
    optimizer = dspy.MIPROv2(
        metric=draft_quality_metric,
        num_candidates=num_candidates,
        max_bootstrapped_demos=3,
        max_labeled_demos=5,
    )

    optimized_module = optimizer.compile(
        module,
        trainset=train,
        max_bootstrapped_demos=3,
        max_labeled_demos=5,
    )

    # Evaluate optimized
    opt_scores = []
    for ex in val[:20]:
        try:
            pred = optimized_module(**{k: getattr(ex, k) for k in ["email_text", "email_subject", "facts", "category"]})
            score = draft_quality_metric(ex, pred)
            opt_scores.append(score)
        except Exception:
            opt_scores.append(0.0)

    opt_avg = sum(opt_scores) / len(opt_scores) if opt_scores else 0

    # Extract optimized prompt
    optimized_prompt = ""
    try:
        # DSPy stores the optimized instructions in the module's predictor
        predictor = optimized_module.generate
        if hasattr(predictor, "extended_signature"):
            optimized_prompt = predictor.extended_signature.instructions
        elif hasattr(predictor, "signature"):
            optimized_prompt = predictor.signature.instructions
    except Exception as e:
        logger.warning("Could not extract optimized prompt: %s", e)

    result = {
        "optimized_prompt": optimized_prompt,
        "baseline_score": round(baseline_avg, 4),
        "optimized_score": round(opt_avg, 4),
        "improvement": round(opt_avg - baseline_avg, 4),
        "num_examples": len(trainset),
        "train_size": len(train),
        "val_size": len(val),
    }

    logger.info(
        "DSPy optimization complete: baseline=%.3f, optimized=%.3f, improvement=%+.3f",
        baseline_avg, opt_avg, opt_avg - baseline_avg,
    )

    return result


# ─── Langfuse Export ─────────────────────────────────────────────────────────

def export_to_langfuse(optimized_prompt: str, version: str = "") -> bool:
    """Push optimized prompt to Langfuse for production use.

    The main.py _get_draft_system_prompt() fetches from Langfuse,
    so this makes the optimized prompt active in production.
    """
    try:
        from ..observability import _get_langfuse
        lf = _get_langfuse()
        if not lf:
            logger.warning("Langfuse not available, cannot export prompt")
            return False

        from datetime import datetime, timezone
        ver = version or datetime.now(timezone.utc).strftime("dspy-%y%m%d-%H%M")

        lf.create_prompt(
            name="draft_generate_system",
            prompt=optimized_prompt,
            labels=["dspy-optimized", ver],
            type="text",
        )
        lf.flush()
        logger.info("Exported optimized prompt to Langfuse: %s", ver)
        return True
    except Exception as e:
        logger.warning("Failed to export prompt to Langfuse: %s", e)
        return False
