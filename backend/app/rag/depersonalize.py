"""Depersonalize email text before RAG ingestion.

Removes personally identifiable information (PII) from email text
while preserving the knowledge content. This prevents:
1. Name confusion in draft generation (RAG chunk mentions "Kovács János"
   but the current email is from someone else)
2. OETP-ID leakage (draft mentions another person's application number)
3. GDPR compliance (no personal data stored in knowledge base)

Strategy:
- Remove greeting line ("Tisztelt XY!")
- Remove OETP application IDs (OETP-2026-XXXXXX → [pályázati azonosító])
- Remove email addresses
- Remove phone numbers
- Remove personal names from signature blocks
- Keep the substantive content (rules, procedures, answers)
"""

from __future__ import annotations

import re


def depersonalize(text: str) -> str:
    """Remove PII from email text while keeping knowledge content.

    Args:
        text: Raw email text (plain text, not HTML)

    Returns:
        Cleaned text with PII replaced by generic placeholders
    """
    if not text:
        return text

    lines = text.split("\n")
    cleaned = []

    for line in lines:
        line = _clean_line(line)
        if line is not None:
            cleaned.append(line)

    result = "\n".join(cleaned)

    # Remove trailing signature block (name + phone + address at the end)
    # Pattern: last few lines that are short and contain PII placeholders or just a name
    lines_final = result.rstrip().split("\n")
    while lines_final:
        last = lines_final[-1].strip()
        # Remove trailing empty lines, phone placeholders, address placeholders, short name-only lines
        if not last:
            lines_final.pop()
        elif last == "[telefonszám]" or last == "[cím]" or last == "[email]":
            lines_final.pop()
        elif len(last) < 40 and not any(c.isdigit() for c in last) and last[0].isupper():
            # Short line starting with uppercase, no digits — likely a name
            # But don't remove content lines
            if re.match(r"^[A-ZÁÉÍÓÖŐÚÜŰa-záéíóöőúüű\s.,-]+$", last) and len(last.split()) <= 4:
                lines_final.pop()
            else:
                break
        else:
            break

    result = "\n".join(lines_final)

    # Final cleanup: multiple blank lines → single
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()


def _clean_line(line: str) -> str | None:
    """Clean a single line. Returns None to skip the line entirely."""
    stripped = line.strip()

    # Skip greeting lines
    if re.match(r"^Tisztelt\s+[^!]+!\s*$", stripped):
        return None

    # Skip signature blocks
    if re.match(r"^Üdvözlettel[,:]\s*$", stripped, re.IGNORECASE):
        return None
    if re.match(r"^(Nemzeti Energetikai|NEÜ Zrt|Zártkörűen Működő|1037|Montevideo)", stripped):
        return None

    # Skip "From:" / "Sent:" / "To:" header lines in quoted sections
    if re.match(r"^(From|Feladó|Sent|Elküldve|To|Címzett|Subject|Tárgy|Cc|Másolat|Date|Dátum|Küldés ideje)\s*:", stripped, re.IGNORECASE):
        return None

    # Skip email addresses on their own line (from header blocks)
    if re.match(r"^<?[\w.+-]+@[\w.-]+>?\s*$", stripped):
        return None

    # Skip name lines in headers: "Gortka István <email>" or just "Name Surname"
    if re.match(r"^[A-ZÁÉÍÓÖŐÚÜŰa-záéíóöőúüű\s.,-]+<[\w.+-]+@[\w.-]+>\s*$", stripped):
        return None

    # Skip standalone date lines (e.g. "Tuesday, April 7, 2026 10:20 AM")
    if re.match(r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)", stripped, re.IGNORECASE):
        return None
    if re.match(r"^\d{4}\.\s*(január|február|március|április|május|június|július|augusztus|szeptember|október|november|december)", stripped, re.IGNORECASE):
        return None

    # Skip separator lines
    if re.match(r"^[-_=]{5,}$", stripped):
        return None

    # Apply PII replacements
    line = _replace_oetp_ids(line)
    line = _replace_emails(line)
    line = _replace_phones(line)
    line = _replace_addresses(line)

    return line


def _replace_oetp_ids(text: str) -> str:
    """OETP-2026-123456 → [pályázati azonosító]"""
    return re.sub(r"OETP-\d{4}-\d{4,8}", "[pályázati azonosító]", text)


def _replace_emails(text: str) -> str:
    """user@example.com → [email]
    But keep official NEÜ email addresses.
    """
    official = {
        "lakossagitarolo@neuzrt.hu",
        "info@neuzrt.hu",
        "oetpkivitelezo@neuzrt.hu",
        "lakossagienergetika@neuzrt.hu",
    }

    def _sub(match):
        addr = match.group(0).lower()
        if addr in official:
            return match.group(0)  # Keep official addresses
        return "[email]"

    return re.sub(r"[\w.+-]+@[\w.-]+\.\w+", _sub, text)


def _replace_phones(text: str) -> str:
    """+36 XX XXX XXXX → [telefonszám]"""
    # Hungarian phone formats
    text = re.sub(r"\+36\s*\d[\d\s/-]{7,12}", "[telefonszám]", text)
    text = re.sub(r"06\s*\d[\d\s/-]{7,10}", "[telefonszám]", text)
    return text


def _replace_addresses(text: str) -> str:
    """Street addresses → [cím]
    But keep NEÜ office address.
    """
    # Skip if it's the NEÜ address
    if "Montevideo" in text or "1037" in text:
        return text

    # Hungarian address patterns: XXXX Város, Utca NNN
    text = re.sub(
        r"\d{4}\s+[A-ZÁÉÍÓÖŐÚÜŰa-záéíóöőúüű]+,\s+[A-ZÁÉÍÓÖŐÚÜŰa-záéíóöőúüű\s.]+\d+[./a-zA-Z]*",
        "[cím]",
        text,
    )

    return text
