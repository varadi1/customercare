"""Legal risk check — verify draft doesn't make unsafe eligibility claims.

Two-level system:
1. Scan draft for eligibility/legal claims (regex, deterministic)
2. If found → query legal RAG for contradictions/restrictions

This is NOT an LLM call — it's a deterministic scan + RAG lookup.
Runs AFTER draft generation, BEFORE saving to Outlook.
"""

from __future__ import annotations

import re
from typing import Optional

import httpx

# Patterns that indicate the draft makes a legal/eligibility claim
ELIGIBILITY_CLAIM_PATTERNS = [
    # Positive claims (most dangerous — may be wrong)
    re.compile(r"(pályázhat|jogosult|támogatható|nincs.*kizáró|nem.*kizáró|nincs.*akadály|lehetséges|engedélyezett)", re.IGNORECASE),
    # Negative claims (less dangerous, but still should be verified)
    re.compile(r"(nem pályázhat|nem jogosult|nem támogatható|kizáró ok|kizárja|nem engedélyezett)", re.IGNORECASE),
    # Specific rule references
    re.compile(r"(felhívás\s+\d+\.\d+|pályázati feltétel|összeférhetetlenség)", re.IGNORECASE),
]

# Keywords to send to legal RAG for verification
LEGAL_QUERY_KEYWORDS = [
    "összeférhetetlenség", "kizáró ok", "jogosultság", "támogathatóság",
    "tulajdonrész", "érdekeltség", "vállalkozás", "gazdasági tevékenység",
    "felszámolás", "csőd", "végrehajtás", "adótartozás", "köztartozás",
]

LEGAL_RAG_URL = "http://host.docker.internal:8103"


async def check_legal_risk(
    draft_text: str,
    email_text: str,
    oetp_ids: list[str] | None = None,
) -> dict:
    """Check if the draft makes legally risky claims.

    Args:
        draft_text: The generated draft (plain text)
        email_text: The original customer email
        oetp_ids: OETP application IDs from the email

    Returns:
        {
            "has_legal_claims": bool,
            "claims_found": [str],
            "legal_rag_consulted": bool,
            "legal_rag_result": str | None,
            "risk_level": "none" | "low" | "high",
            "recommendation": str,
        }
    """
    result = {
        "has_legal_claims": False,
        "claims_found": [],
        "legal_rag_consulted": False,
        "legal_rag_result": None,
        "risk_level": "none",
        "recommendation": "",
    }

    if not draft_text:
        return result

    # Level 1: Scan for eligibility claims (deterministic regex)
    claims = []
    for pattern in ELIGIBILITY_CLAIM_PATTERNS:
        matches = pattern.findall(draft_text)
        claims.extend(matches)

    if not claims:
        return result

    result["has_legal_claims"] = True
    result["claims_found"] = list(set(claims))[:5]

    # Level 2: Query legal RAG for potential contradictions
    # Build a focused query from email + draft context
    combined_context = f"{email_text[:500]} {draft_text[:500]}"
    legal_keywords_found = [kw for kw in LEGAL_QUERY_KEYWORDS if kw.lower() in combined_context.lower()]

    if not legal_keywords_found:
        # Claims found but no legal keywords — low risk, just flag
        result["risk_level"] = "low"
        result["recommendation"] = f"Jogosultsági állítás a draftban: {claims[:3]}"
        return result

    # Query legal RAG
    legal_query = " ".join(legal_keywords_found[:5]) + " " + " ".join(claims[:3])

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            # Try the answer endpoint for structured response
            resp = await client.post(
                f"{LEGAL_RAG_URL}/answer",
                json={"query": legal_query[:500]},
            )

            if resp.status_code == 200:
                legal_data = resp.json()
                result["legal_rag_consulted"] = True

                confidence = legal_data.get("confidence", "insufficient")
                answer = legal_data.get("answer", "")

                result["legal_rag_result"] = answer[:300] if answer else None

                # Check if legal RAG found relevant restrictions
                restriction_words = [
                    "nem vehet részt", "kizáró", "összeférhetetlenség",
                    "nem jogosult", "tilos", "nem támogatható",
                    "kizárja", "feltétele", "korlátozás",
                ]
                has_restriction = any(w in answer.lower() for w in restriction_words) if answer else False

                if has_restriction:
                    result["risk_level"] = "high"
                    result["recommendation"] = (
                        "⚖️ JOGI KOCKÁZAT: A jogszabály RAG összeférhetetlenségi/kizáró "
                        "szabályt talált. A draft jogosultsági állítása ellenőrzést igényel."
                    )
                elif confidence in ("high", "medium"):
                    result["risk_level"] = "low"
                    result["recommendation"] = "Jogi RAG nem talált kizáró szabályt, de emberi ellenőrzés javasolt."
                else:
                    result["risk_level"] = "low"
                    result["recommendation"] = "Jogi RAG nem adott egyértelmű választ — emberi ellenőrzés javasolt."

    except Exception as e:
        # Legal RAG unavailable — flag for human review
        result["risk_level"] = "low"
        result["recommendation"] = f"Jogi RAG nem elérhető ({e}) — emberi ellenőrzés javasolt."

    return result
