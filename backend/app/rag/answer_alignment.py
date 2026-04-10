"""Answer-Question Alignment Check — does the draft actually answer the question?

Detects the "echo problem": when the draft repeats back what the customer
already said instead of answering what they asked.

Strategy:
1. Extract the actual QUESTION from the email (what does the customer want to know?)
2. Check if the draft ANSWERS that question or just echoes context

Uses a lightweight LLM call — cheaper than CoVe because it's a single yes/no judgment.
"""

from __future__ import annotations

import json
import re

from ..llm_client import chat_completion

ALIGNMENT_SYSTEM = """Te egy minőség-ellenőr vagy. Egy ügyfélszolgálati email-választ kell értékelned.

A feladatod: eldönteni, hogy a válasz TÉNYLEGESEN MEGVÁLASZOLJA-e az ügyfél kérdését, vagy csak visszamondja amit az ügyfél már tud.

ÉRTÉKELÉSI SZABÁLYOK:
- "answers": A válasz érdemi információt ad amit az ügyfél NEM tudott
- "echoes": A válasz csak megerősíti/visszamondja amit az ügyfél már leírt a levelében
- "partial": A válasz részben válaszol, de a fő kérdésre nem
- "irrelevant": A válasz teljesen más témáról szól

VÁLASZ (JSON):
{
  "verdict": "answers|echoes|partial|irrelevant",
  "reason": "rövid indoklás magyarul"
}"""


async def check_alignment(
    email_text: str,
    draft_text: str,
    lf_trace=None,
) -> dict:
    """Check if the draft actually answers the customer's question.

    Args:
        email_text: The original customer email
        draft_text: The generated draft (plain text, no HTML)

    Returns:
        {"verdict": "answers|echoes|partial|irrelevant", "reason": str, "aligned": bool}
    """
    if not email_text or not draft_text:
        return {"verdict": "skip", "reason": "empty input", "aligned": True}

    # Strip greeting and signature from draft
    clean_draft = re.sub(r"^Tisztelt\s+[^!]+!\s*", "", draft_text)
    clean_draft = re.sub(r"\s*Üdvözlettel:.*$", "", clean_draft, flags=re.DOTALL)
    clean_draft = clean_draft.strip()

    if len(clean_draft) < 15:
        return {"verdict": "skip", "reason": "draft too short", "aligned": True}

    try:
        result = await chat_completion(
            messages=[
                {"role": "system", "content": ALIGNMENT_SYSTEM},
                {"role": "user", "content": f"ÜGYFÉL LEVELE:\n{email_text[:1500]}\n\nVÁLASZ TERVEZET:\n{clean_draft[:1000]}"},
            ],
            temperature=0.0,
            max_tokens=200,
            json_mode=True,
        )

        data = json.loads(result["content"])
        verdict = data.get("verdict", "unknown")
        reason = data.get("reason", "")

        out = {
            "verdict": verdict,
            "reason": reason,
            "aligned": verdict in ("answers", "partial"),
        }

        if lf_trace:
            lf_trace.alignment(
                result=out,
                usage=result.get("usage"),
                model=result.get("model", ""),
                provider=result.get("provider", ""),
                duration_ms=result.get("duration_ms", 0),
            )

        return out

    except Exception as e:
        return {"verdict": "error", "reason": str(e), "aligned": True}
