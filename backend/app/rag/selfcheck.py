"""SelfCheck — multi-sample consistency check for hallucination detection.

Generate N drafts for the same question, measure semantic consistency.
If generations disagree, the model is uncertain → likely hallucinating.

Only runs for medium confidence drafts (high = already confident, low = already flagged).
Cost: N-1 extra LLM calls per checked draft.
"""

from __future__ import annotations

import math
import re

from ..llm_client import chat_completion


async def selfcheck(
    messages: list[dict],
    original_response: str,
    n_samples: int = 2,
    temperature: float = 0.7,
    similarity_threshold: float = 0.5,
) -> dict:
    """Run SelfCheck: generate N additional samples and compare consistency.

    Args:
        messages: The original LLM messages (system + user)
        original_response: The original LLM response text
        n_samples: Number of additional samples to generate (default 2)
        temperature: Higher temp = more diversity between samples
        similarity_threshold: Below this = inconsistent

    Returns:
        {
            "consistent": bool,
            "avg_similarity": float,
            "min_similarity": float,
            "n_samples": int,
            "details": [{"sample": int, "similarity": float}]
        }
    """
    if not original_response or not messages:
        return {"consistent": True, "avg_similarity": 1.0, "min_similarity": 1.0, "n_samples": 0, "details": []}

    # Strip HTML for comparison
    orig_plain = re.sub(r"<[^>]+>", "", original_response).strip()
    if len(orig_plain) < 30:
        return {"consistent": True, "avg_similarity": 1.0, "min_similarity": 1.0, "n_samples": 0, "details": []}

    samples = []
    for i in range(n_samples):
        try:
            result = await chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=1000,
                json_mode=True,
            )
            import json
            sample_data = json.loads(result["content"])
            sample_body = sample_data.get("body", "")
            sample_plain = re.sub(r"<[^>]+>", "", sample_body).strip()
            samples.append(sample_plain)
        except Exception:
            continue

    if not samples:
        return {"consistent": True, "avg_similarity": 1.0, "min_similarity": 1.0, "n_samples": 0, "details": []}

    # Compare each sample to the original using word overlap (fast, no embedding needed)
    similarities = []
    for i, sample in enumerate(samples):
        sim = _word_overlap_similarity(orig_plain, sample)
        similarities.append({"sample": i + 1, "similarity": round(sim, 3)})

    avg_sim = sum(s["similarity"] for s in similarities) / len(similarities)
    min_sim = min(s["similarity"] for s in similarities)

    return {
        "consistent": min_sim >= similarity_threshold,
        "avg_similarity": round(avg_sim, 3),
        "min_similarity": round(min_sim, 3),
        "n_samples": len(samples),
        "details": similarities,
    }


def _word_overlap_similarity(text_a: str, text_b: str) -> float:
    """Jaccard-like word overlap between two texts.

    More robust than exact string matching for paraphrased responses.
    """
    if not text_a or not text_b:
        return 0.0

    # Normalize
    words_a = set(re.findall(r"[a-záéíóöőúüű]{3,}", text_a.lower()))
    words_b = set(re.findall(r"[a-záéíóöőúüű]{3,}", text_b.lower()))

    if not words_a or not words_b:
        return 0.0

    intersection = words_a & words_b
    union = words_a | words_b

    return len(intersection) / len(union)
