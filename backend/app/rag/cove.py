"""Chain of Verification (CoVe) — post-generation fact-checking.

After the LLM generates a draft, CoVe:
1. Extracts verifiable claims from the draft
2. For each claim, checks if it's supported by the source facts
3. Returns list of unsupported claims

This is a lightweight implementation using the same LLM (not a separate model).
Only runs for medium/low confidence drafts to save cost.
"""

from __future__ import annotations

import json
import re

from ..llm_client import chat_completion

COVE_SYSTEM = """Te egy tényellnőr asszisztens vagy. A feladatod: egy email-válasz tervezet állításait egyenként ellenőrizni a megadott forrás-tények alapján.

Minden állítást értékelj:
- "supported" = a forrás-tények egyértelműen alátámasztják
- "unsupported" = a forrás-tények NEM tartalmaznak ilyen információt
- "contradicted" = a forrás-tények ELLENTMONDANAK az állításnak

VÁLASZ (szigorúan JSON):
{
  "claims": [
    {"text": "az állítás szövege", "verdict": "supported|unsupported|contradicted", "source_fact": 1}
  ],
  "overall": "all_supported|has_unsupported|has_contradicted"
}"""


async def verify_draft(
    draft_text: str,
    facts: list[dict],
    max_claims: int = 8,
) -> dict:
    """Run Chain of Verification on a draft.

    Args:
        draft_text: The generated draft (plain text, no HTML)
        facts: List of verified facts [{text, source, verified}]
        max_claims: Max claims to check (cost control)

    Returns:
        dict with claims, overall verdict, and list of unsupported/contradicted claims
    """
    if not draft_text or not facts:
        return {"claims": [], "overall": "no_data", "issues": []}

    # Build facts block for the verifier
    facts_block = ""
    for i, f in enumerate(facts, 1):
        facts_block += f'[TÉNY {i}] ({f.get("source", "?")})\n"{f["text"]}"\n\n'

    # Strip greeting and signature from draft for cleaner verification
    clean_draft = re.sub(r"^Tisztelt\s+[^!]+!\s*", "", draft_text)
    clean_draft = re.sub(r"\s*Üdvözlettel:.*$", "", clean_draft, flags=re.DOTALL)
    clean_draft = clean_draft.strip()

    if len(clean_draft) < 20:
        return {"claims": [], "overall": "too_short", "issues": []}

    try:
        messages = [
            {"role": "system", "content": COVE_SYSTEM},
            {"role": "user", "content": f"DRAFT:\n{clean_draft}\n\nFORRÁS-TÉNYEK:\n{facts_block}"},
        ]

        result = await chat_completion(
            messages=messages,
            temperature=0.0,
            max_tokens=800,
            json_mode=True,
        )

        data = json.loads(result["content"])
        claims = data.get("claims", [])[:max_claims]
        overall = data.get("overall", "unknown")

        # Extract issues
        issues = [
            c for c in claims
            if c.get("verdict") in ("unsupported", "contradicted")
        ]

        return {
            "claims": claims,
            "overall": overall,
            "issues": issues,
            "issue_count": len(issues),
        }

    except Exception as e:
        return {"claims": [], "overall": "error", "error": str(e), "issues": []}
