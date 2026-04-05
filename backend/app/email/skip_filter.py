"""Email skip filter: detect emails that do NOT need a Hanna draft.

Returns a skip_reason string (or None if the email should be processed).
Integrated into /draft/context response so all scripts can use it.
"""

from __future__ import annotations

import re


def _strip_quoted(text: str) -> str:
    """Remove quoted/forwarded parts from email text."""
    lines = text.split("\n")
    clean = []
    for line in lines:
        if line.strip().startswith(">"):
            break
        if re.match(r"^-{3,}\s*(Eredeti|Original|Forwarded)", line, re.IGNORECASE):
            break
        if re.match(r"^(From|Feladó|Tárgy|Subject|Sent|Dátum|Date):", line):
            break
        if "írta:" in line and "@" in line:
            break
        clean.append(line)
    return "\n".join(clean).strip()


# ─── Thank-you / Acknowledgment detection ────────────────────────────────────

# Short acknowledgment patterns (Hungarian)
_THANKYOU_PATTERNS = [
    r"\bköszön",           # köszönöm, köszönjük, köszönöm szépen
    r"\bköszi\b",
    r"\bmegkaptam\b",
    r"\bmegkaptuk\b",
    r"\btudomásul\b",
    r"\brendben\b",
    r"\bköszönettel\b",
    r"\bköszönöm a választ",
    r"\bköszönöm a gyors",
    r"\bköszönöm a tájékoztat",
    r"\bköszönöm szépen a választ",
    r"\bígy fogunk eljárni\b",
    r"\bkitartást kívánok\b",
]

# If the stripped body (without quotes/signatures) is shorter than this,
# and matches a thank-you pattern, it's a pure acknowledgment
_THANKYOU_MAX_CHARS = 250


def _is_thankyou(stripped_text: str) -> bool:
    """Detect if the email is a short thank-you / acknowledgment."""
    text_lower = stripped_text.lower().strip()

    # Remove common closings to measure actual content
    # (e.g., "Üdvözlettel, Kiss Róbert" or signature blocks)
    content = re.split(
        r"(?:^|\n)\s*(?:üdvözlettel|tisztelettel|köszönettel|--)\b",
        text_lower, maxsplit=1, flags=re.IGNORECASE,
    )[0].strip()

    if len(content) > _THANKYOU_MAX_CHARS:
        return False

    for pattern in _THANKYOU_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return True

    return False


# ─── Admin data-change requests (no Hanna needed) ───────────────────────────

# Meghatalmazott requesting email/data changes — colleague handles manually
_ADMIN_CHANGE_PATTERNS = [
    # Email address change requests
    (r"\be-?mail\s*(cím|address)", r"módosít|változtat|javít|cserél|frissít|helyesb"),
    (r"\be-?mail\s*(cím|address)", r"rosszul|hibás|téves|elírt"),
    # Phone number changes
    (r"\btelefonszám", r"módosít|változtat|javít|cserél|frissít"),
    # Contact data changes
    (r"\bkapcsolattart\w*\s*adat", r"módosít|változtat|javít|cserél|frissít"),
    # Generic data correction
    (r"\badat\w*\s*(módosít|javít|változtat|helyesb)", None),
    (r"\bkérelem\s*adat\s*javít", None),
]


def _is_admin_data_change(subject: str, stripped_text: str) -> bool:
    """Detect if the email is a data/email modification request.

    These are typically sent by meghatalmazottak (representatives)
    requesting changes to applicant contact info. Colleagues handle
    these manually — no Hanna draft needed.
    """
    combined = (subject + " " + stripped_text).lower()

    for pattern_pair in _ADMIN_CHANGE_PATTERNS:
        primary, secondary = pattern_pair
        if re.search(primary, combined, re.IGNORECASE):
            if secondary is None:
                return True
            if re.search(secondary, combined, re.IGNORECASE):
                return True

    return False


# ─── Auto-reply / System messages ────────────────────────────────────────────

_AUTOREPLY_SUBJECTS = [
    r"automatic reply",
    r"auto[- ]?reply",
    r"out of office",
    r"házon kívül",
    r"undeliverable",
    r"delivery.*fail",
    r"kézbesít.*hiba",
    r"mailer[- ]daemon",
]


def _is_autoreply(subject: str, sender_email: str) -> bool:
    """Detect auto-reply and system bounce messages."""
    subj_lower = subject.lower()
    for pattern in _AUTOREPLY_SUBJECTS:
        if re.search(pattern, subj_lower):
            return True
    if "mailer-daemon" in sender_email.lower() or "postmaster" in sender_email.lower():
        return True
    return False


# ─── Main filter ─────────────────────────────────────────────────────────────

def check_skip(
    email_text: str,
    email_subject: str = "",
    sender_email: str = "",
) -> dict:
    """Check if an email should be skipped (no Hanna draft needed).

    Returns:
        {"skip": True/False, "reason": "...", "category": "..."}

    Categories:
        - "thankyou": acknowledgment / thank-you email
        - "admin_data_change": email/data modification request
        - "autoreply": auto-reply or bounce message
        - None: email should be processed normally
    """
    # Strip quoted parts to analyze only the new content
    stripped = _strip_quoted(email_text)

    # 1. Auto-reply (cheapest check first)
    if _is_autoreply(email_subject, sender_email):
        return {
            "skip": True,
            "reason": "Automatikus válasz / rendszerüzenet — nem kell draft",
            "skip_category": "autoreply",
        }

    # 2. Thank-you / acknowledgment
    if _is_thankyou(stripped):
        return {
            "skip": True,
            "reason": "Köszönő / visszajelző levél — nem kell draft",
            "skip_category": "thankyou",
        }

    # 3. Admin data change (email/phone/contact modification)
    if _is_admin_data_change(email_subject, stripped):
        return {
            "skip": True,
            "reason": "Adatmódosítási kérelem (email/telefon/kapcsolattartó) — kolléga kezeli manuálisan",
            "skip_category": "admin_data_change",
        }

    return {"skip": False, "reason": None, "skip_category": None}
