"""Domain guardrails for email drafts.

Config-driven via program.yaml guardrails section. Each guardrail can be
toggled on/off. Custom rules loaded from guardrails.custom_rules.

Built-in guardrails:
1. Numerical consistency (Ft, %, kW, dates)
2. Eligibility claims only from official docs
3. Contradicting chunks detection
4. App ID cross-check (uses app_id_pattern from database config)
5. Forbidden phrases + AI-speak detection
6. Provenance grounding
7. Response completeness
"""

from __future__ import annotations

import re
from bs4 import BeautifulSoup

from ..config import get_program_config, get_db_config


def run_all_guardrails(
    body_html: str,
    verified_facts: list[dict],
    top_chunks: list[dict],
    email_app_ids: list[str] | None = None,
    email_text: str = "",
    citations: dict | None = None,
) -> dict:
    """Run all domain guardrails and return aggregated results.

    Returns:
        {
            "pass": bool,
            "warnings": [{"rule": str, "severity": "high"|"medium", "detail": str}],
            "suggested_confidence": "high"|"medium"|"low" or None,
        }
    """
    pcfg = get_program_config()
    gr_cfg = pcfg.get("guardrails", {})

    warnings = []
    plain = BeautifulSoup(body_html, "html.parser").get_text() if body_html else ""
    facts_text = " ".join(f.get("text", "") for f in verified_facts)

    # 1. Numerical consistency
    if gr_cfg.get("numerical_check", True):
        warnings.extend(_check_numerical(plain, facts_text))

    # 2. Eligibility claims from non-authoritative sources
    if gr_cfg.get("eligibility_check", True):
        warnings.extend(_check_eligibility_source(plain, top_chunks))

    # 3. Contradicting chunks
    warnings.extend(_check_contradictions(top_chunks))

    # 4. App ID cross-check
    if email_app_ids:
        warnings.extend(_check_app_id_match(plain, email_app_ids))

    # 5. Forbidden phrases + AI-speak
    if gr_cfg.get("ai_speak_check", True):
        warnings.extend(_check_forbidden_phrases(plain))

    # 6. Provenance grounding
    warnings.extend(check_provenance(body_html, citations or {}))

    # 7. Response completeness
    if email_text:
        warnings.extend(check_completeness(email_text, body_html))

    # 8. Custom rules from program.yaml
    for rule in gr_cfg.get("custom_rules", []):
        pattern = rule.get("pattern", "")
        if pattern and re.search(pattern, plain, re.IGNORECASE):
            warnings.append({
                "rule": rule.get("name", "custom"),
                "severity": rule.get("severity", "medium"),
                "detail": rule.get("message", f"Custom rule triggered: {pattern}"),
            })

    # Determine suggested confidence
    high_severity = [w for w in warnings if w["severity"] == "high"]
    medium_severity = [w for w in warnings if w["severity"] == "medium"]

    suggested = None
    if high_severity:
        suggested = "low"
    elif len(medium_severity) >= 2:
        suggested = "low"
    elif medium_severity:
        suggested = "medium"

    return {
        "pass": len(warnings) == 0,
        "warnings": warnings,
        "suggested_confidence": suggested,
    }


# ── 1. Numerical consistency ──

def _check_numerical(plain: str, facts_text: str) -> list[dict]:
    """Check Ft amounts, percentages, dates, kW values."""
    warnings = []
    plain_lower = plain.lower()
    facts_lower = facts_text.lower()

    # Ft amounts
    def _amounts(text):
        amounts = set()
        for m in re.finditer(r"([\d.]+)\s*(?:ft|forint)", text):
            try:
                amounts.add(int(m.group(1).replace(".", "")))
            except ValueError:
                pass
        for m in re.finditer(r"([\d,]+)\s*milli[oó]", text):
            try:
                amounts.add(int(float(m.group(1).replace(",", ".")) * 1_000_000))
            except ValueError:
                pass
        return amounts

    draft_amounts = _amounts(plain_lower)
    fact_amounts = _amounts(facts_lower)
    if draft_amounts and fact_amounts:
        novel = draft_amounts - fact_amounts
        if novel:
            warnings.append({
                "rule": "numerical_ft",
                "severity": "high",
                "detail": f"Ft összeg a draftban ({novel}) nincs a forrásban ({fact_amounts})",
            })

    # Percentages
    draft_pcts = set(re.findall(r"(\d+)\s*(?:%|százalék)", plain_lower))
    fact_pcts = set(re.findall(r"(\d+)\s*(?:%|százalék)", facts_lower))
    if draft_pcts and fact_pcts:
        novel = draft_pcts - fact_pcts
        if novel:
            warnings.append({
                "rule": "numerical_pct",
                "severity": "high",
                "detail": f"Százalék ({novel}) nincs a forrásban",
            })

    # Dates
    draft_dates = set(re.findall(r"\d{4}[\.\s]+\d{2}[\.\s]+\d{2}", plain))
    fact_dates = set(re.findall(r"\d{4}[\.\s]+\d{2}[\.\s]+\d{2}", facts_text))
    if draft_dates:
        novel = draft_dates - fact_dates
        if novel:
            warnings.append({
                "rule": "numerical_date",
                "severity": "high",
                "detail": f"Dátum ({novel}) nincs a forrásban",
            })

    return warnings


# ── 2. Eligibility claims from authoritative sources only ──

ELIGIBILITY_PATTERNS = [
    re.compile(r"(jogosult|nem jogosult|támogatható|nem támogatható|kizár|elutasít)", re.IGNORECASE),
    re.compile(r"(pályáz(?:hat|ó)|nem pályázhat)", re.IGNORECASE),
    re.compile(r"(feltétel.*teljesül|nem teljesül)", re.IGNORECASE),
]

def _get_authoritative_types() -> set[str]:
    pcfg = get_program_config()
    types = {
        name for name, cfg in pcfg.get("doc_types", {}).items()
        if isinstance(cfg, dict) and cfg.get("authority", 0) >= 0.75
    }
    return types or {"felhívás", "melléklet", "közlemény", "gyik", "segédlet"}

AUTHORITATIVE_TYPES = _get_authoritative_types()


def _check_eligibility_source(plain: str, top_chunks: list[dict]) -> list[dict]:
    """Flag if eligibility claims are made but top sources are only emails."""
    warnings = []

    has_eligibility_claim = any(p.search(plain) for p in ELIGIBILITY_PATTERNS)
    if not has_eligibility_claim:
        return []

    # Check if any top chunk is from an authoritative source
    chunk_types = {c.get("chunk_type", c.get("metadata", {}).get("doc_type", "")) for c in top_chunks}
    has_official = bool(chunk_types & AUTHORITATIVE_TYPES)

    if not has_official:
        warnings.append({
            "rule": "eligibility_source",
            "severity": "high",
            "detail": f"Jogosultsági állítás, de nincs hivatalos forrás a top chunkokban (types: {chunk_types})",
        })

    return warnings


# ── 3. Contradicting chunks ──

def _check_contradictions(top_chunks: list[dict]) -> list[dict]:
    """Detect basic contradictions between top chunks.

    Simple heuristic: if one chunk says "nem támogatható" and another says
    "támogatható" for the same concept, flag it.
    """
    warnings = []
    if len(top_chunks) < 2:
        return []

    texts = [c.get("text", "").lower() for c in top_chunks]

    contradiction_pairs = [
        ("támogatható", "nem támogatható"),
        ("jogosult", "nem jogosult"),
        ("lehetséges", "nem lehetséges"),
        ("szükséges", "nem szükséges"),
        ("kötelező", "nem kötelező"),
    ]

    for positive, negative in contradiction_pairs:
        has_positive = any(positive in t and negative not in t for t in texts)
        has_negative = any(negative in t for t in texts)
        if has_positive and has_negative:
            warnings.append({
                "rule": "contradiction",
                "severity": "high",
                "detail": f"Ellentmondás a chunkokban: '{positive}' vs. '{negative}'",
            })

    return warnings


# ── 4. OETP ID cross-check ──

def _check_app_id_match(plain: str, email_app_ids: list[str]) -> list[dict]:
    """Check that application IDs in the draft match the email's IDs."""
    warnings = []
    db_cfg = get_db_config()
    pattern = db_cfg.get("app_id_pattern", r"[A-Z]+-\d{4}-\d+")
    draft_ids = set(re.findall(pattern, plain, re.IGNORECASE))
    email_ids = set(email_app_ids)

    if draft_ids and email_ids and not (draft_ids & email_ids):
        warnings.append({
            "rule": "app_id_mismatch",
            "severity": "high",
            "detail": f"Draft app IDs ({draft_ids}) ≠ email IDs ({email_ids})",
        })

    return warnings


# ── 5. Forbidden phrases ──

FORBIDDEN_PATTERNS = [
    # Never promise specific timeline unless in sources
    (re.compile(r"(néhány napon belül|hamarosan megkapja|rövidesen értesítjük)", re.IGNORECASE),
     "Konkrét időkeret ígérete — csak forrásból szabad"),
    # Never claim "free" unless sources say so
    (re.compile(r"(teljesen ingyenes|díjmentes|költségmentes)", re.IGNORECASE),
     "Ingyenesség állítása — ellenőrizendő"),
]


def _check_forbidden_phrases(plain: str) -> list[dict]:
    warnings = []
    for pattern, reason in FORBIDDEN_PATTERNS:
        if pattern.search(plain):
            warnings.append({
                "rule": "forbidden_phrase",
                "severity": "medium",
                "detail": reason,
            })

    # AI-speak detection — phrases that reveal LLM origin
    ai_speak = [
        "rendelkezésre álló tények",
        "rendelkezésre álló ellenőrzött",
        "megadott tények alapján",
        "ellenőrzött tények",
        "forrás-chunk",
        "nem szerepel információ",
        "nem tartalmaz információt",
        "nem adnak választ",
        "nem adnak egyértelmű választ",
        "a tények alapján",
        "a tények nem",
        "kollégánk hamarosan válaszol",
    ]
    plain_lower = plain.lower()
    for phrase in ai_speak:
        if phrase in plain_lower:
            warnings.append({
                "rule": "ai_speak",
                "severity": "high",
                "detail": f"AI-speak detektálva: '{phrase}'",
            })
            break  # One is enough to flag

    return warnings


# ── 6. Provenance grounding — every paragraph must trace to a source ──

def check_provenance(body_html: str, citations: dict) -> list[dict]:
    """Check that substantive paragraphs have source citations.

    A paragraph with factual claims but no [N] citation is ungrounded.
    """
    warnings = []
    plain = BeautifulSoup(body_html, "html.parser").get_text() if body_html else ""

    # Split into paragraphs
    paragraphs = [p.strip() for p in plain.split("\n") if len(p.strip()) > 30]

    # Filter out greeting, closing, and deferral phrases
    # Build skip patterns from program.yaml signature + standard patterns
    pcfg = get_program_config()
    sig_text = pcfg.get("email", {}).get("signature", "").lower()
    sig_words = [w for w in re.findall(r"[a-záéíóöőúüű]{4,}", sig_text)]
    skip_patterns = ["tisztelt", "üdvözlettel", "kollégánk", "kérdésére"] + sig_words[:5]
    content_paras = [
        p for p in paragraphs
        if not any(s in p.lower() for s in skip_patterns)
    ]

    if not content_paras:
        return []

    uncited = [p for p in content_paras if not re.search(r"\[\d+\]", p)]
    if uncited and len(uncited) > len(content_paras) // 2:
        warnings.append({
            "rule": "provenance_gap",
            "severity": "medium",
            "detail": f"{len(uncited)}/{len(content_paras)} tartalmi bekezdés forrás nélkül",
        })

    return warnings


# ── 7. Response completeness — did we answer all parts of the question? ──

def check_completeness(email_text: str, body_html: str) -> list[dict]:
    """Check if the draft addresses all questions in the email.

    Simple heuristic: count question marks in email, check if draft
    has enough substance to cover them.
    """
    warnings = []
    question_count = email_text.count("?")

    if question_count <= 1:
        return []  # Single question — completeness is fine

    plain = BeautifulSoup(body_html, "html.parser").get_text() if body_html else ""
    # Count substantial sentences in draft (excluding greeting/closing)
    sentences = [s for s in re.split(r"[.!]\s+", plain)
                 if len(s.strip()) > 20
                 and "tisztelt" not in s.lower()
                 and "üdvözlettel" not in s.lower()]

    if len(sentences) < question_count // 2:
        warnings.append({
            "rule": "completeness",
            "severity": "medium",
            "detail": f"Email {question_count} kérdést tartalmaz, de a draft csak {len(sentences)} érdemi mondatot",
        })

    return warnings
