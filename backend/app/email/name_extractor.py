"""Extract sender's real name from email body signature.

The Graph API display name is often wrong (English order, missing accents,
company name instead of person). The email body signature is more reliable.

Strategy:
1. Look for name-like patterns in the last 10 lines of the email body
2. Match against known Hungarian name patterns
3. Fall back to Graph API name if nothing found
"""

from __future__ import annotations

import re

# Patterns that indicate a signature line with a name
_GREETING_PATTERNS = [
    # "Üdvözlettel, Kiss Zoltán"
    re.compile(r"[Üü]dvözlettel[,:]?\s+(.+)", re.IGNORECASE),
    # "Tisztelettel, Kiss Zoltán" or "Tisztelettel: Kiss Zoltán"
    re.compile(r"[Tt]isztelettel[,:]?\s+(.+)", re.IGNORECASE),
    # "Köszönettel, Kiss Zoltán"
    re.compile(r"[Kk]öszönettel[,:]?\s+(.+)", re.IGNORECASE),
    # "Köszönöm, Kiss Zoltán" or "Köszönöm:" then name on next line
    re.compile(r"[Kk]öszönöm[,:]?\s+(.+)", re.IGNORECASE),
]

# Company suffixes — if sender_name contains these, it's a company not a person
_COMPANY_SUFFIXES = [
    "kft", "kft.", "bt", "bt.", "zrt", "zrt.", "nyrt", "nyrt.",
    "rt", "rt.", "ev", "ev.", "e.v.",
    "gmbh", "ltd", "llc", "inc", "ag", "s.r.o.",
]

# Lines to skip in signature area
_SKIP_PATTERNS = [
    re.compile(r"^(tel|phone|mobil|fax|web|www|http|email|e-mail)\s*[:.+]", re.IGNORECASE),
    re.compile(r"^\+?\d[\d\s/-]{7,}"),  # Phone number
    re.compile(r"^[\w.+-]+@[\w.-]+\.\w+"),  # Email address
    re.compile(r"^\d{4}\s+\w"),  # Address (starts with zip code)
    re.compile(r"^(cím|address|számlázási|telephely|székhely)", re.IGNORECASE),
    re.compile(r"^-{3,}"),  # Separator
    re.compile(r"^_{3,}"),
    re.compile(r"^n\s+(NAPELEM|HŐSZIVATTYÚ|PASSZÍV)", re.IGNORECASE),  # Product list
]

# Hungarian name pattern: 2-4 capitalized words, optionally with "né"
_NAME_PATTERN = re.compile(
    r"^((?:Dr\.?\s+)?[A-ZÁÉÍÓÖŐÚÜŰ][a-záéíóöőúüű]+(?:né)?"
    r"(?:\s+[A-ZÁÉÍÓÖŐÚÜŰ][a-záéíóöőúüű]+){1,3})\s*$"
)

# Title/role words that are NOT part of the name
_ROLE_WORDS = {
    "ügyvezető", "igazgató", "tulajdonos", "vezető", "mérnök",
    "munkatárs", "asszisztens", "titkár", "elnök", "tag",
}


def extract_name_from_body(email_body: str) -> str | None:
    """Extract sender's real name from email body signature.

    Args:
        email_body: Plain text email body

    Returns:
        Name string if found, None if not
    """
    if not email_body:
        return None

    lines = email_body.strip().split("\n")

    # Strategy 1: Look for "Üdvözlettel, Name" pattern
    for line in lines[-15:]:
        stripped = line.strip()
        for pattern in _GREETING_PATTERNS:
            match = pattern.match(stripped)
            if match:
                candidate = match.group(1).strip()
                candidate = _clean_name(candidate)
                if candidate and _looks_like_name(candidate):
                    return candidate

    # Strategy 2: Look for standalone name in last 10 lines
    # (after "Köszönöm" or similar, the name is often on the next line)
    sig_area = lines[-10:]
    for i, line in enumerate(sig_area):
        stripped = line.strip()

        # Skip non-name lines
        if not stripped or len(stripped) < 3:
            continue
        if any(p.match(stripped) for p in _SKIP_PATTERNS):
            continue

        # Check if it looks like a name (possibly with company/role appended)
        candidate = _clean_name(stripped)
        if candidate and _looks_like_name(candidate):
            return candidate

    return None


def _clean_name(name: str) -> str:
    """Clean up a name candidate."""
    # Remove trailing punctuation
    name = name.rstrip(".,;:!?")
    # Remove company/role if appended (e.g. "Kiss Zoltán - Solergy Bt." or "Kiss Zoltán - Ügyvezető")
    for sep in [" - ", " – ", " | ", " / "]:
        if sep in name:
            parts = name.split(sep)
            # Find the part that looks like a person name (not company, not role)
            for part in parts:
                part = part.strip()
                if _looks_like_name(part) and not is_company_name(part):
                    return part
            # If no part matched, use the first one
            name = parts[0].strip()

    return name.strip()


def _looks_like_name(text: str) -> bool:
    """Check if text looks like a Hungarian person name."""
    if not text or len(text) < 4:
        return False

    words = text.split()
    if len(words) < 2 or len(words) > 5:
        return False

    # All words should start with uppercase
    if not all(w[0].isupper() or w.startswith("dr.") for w in words):
        return False

    # Should not contain company suffixes
    text_lower = text.lower()
    if any(f" {s}" in f" {text_lower} " or text_lower.endswith(f" {s}") for s in _COMPANY_SUFFIXES):
        return False

    # Should not be a role/title
    if any(w.lower() in _ROLE_WORDS for w in words):
        return False

    # Should not contain digits (except in "Dr.")
    if any(c.isdigit() for c in text.replace("Dr.", "")):
        return False

    return True


def is_company_name(sender_name: str) -> bool:
    """Check if the sender name is a company, not a person."""
    if not sender_name:
        return False

    lower = sender_name.lower().strip()

    # Check company suffixes
    for suffix in _COMPANY_SUFFIXES:
        if lower.endswith(f" {suffix}") or f" {suffix} " in f" {lower} ":
            return True
        if lower.endswith(f" {suffix}."):
            return True

    # Very long names with dashes are usually companies
    if len(sender_name) > 40 and " - " in sender_name:
        return True

    return False
