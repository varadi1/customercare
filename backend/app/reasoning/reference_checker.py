"""
Reference checker — validates section references in Hanna's answers.

Checks that point/section references (e.g., "3.3. pont", "4.1. pont")
actually exist in the source chunks used for the answer.

Also checks for program name errors (Otthonfelújítási vs OETP).
"""
from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)

# Pattern: "3.3. pont", "4.1 pont", "10.2.5. pont"
SECTION_REF_PATTERN = re.compile(r'(\d+\.[\d.]+\.?\s*pont)', re.IGNORECASE)

# Wrong program names
WRONG_PROGRAM_PATTERNS = [
    re.compile(r'Otthonfelújítási\s+Program', re.IGNORECASE),
    re.compile(r'Otthon\s*felújítás', re.IGNORECASE),
]


def check_references(
    answer_text: str,
    source_chunks: list[str],
) -> dict:
    """Validate references in the answer against source chunks.

    Returns:
        {
            "valid": True/False,
            "issues": ["3.3. pont not found in sources", ...],
            "referenced_sections": ["3.3. pont", ...],
            "wrong_program": False,
        }
    """
    issues = []
    answer_sections = set(SECTION_REF_PATTERN.findall(answer_text))

    # Check each referenced section exists in sources
    all_sources = " ".join(source_chunks)
    missing_sections = []
    for section in answer_sections:
        # Normalize: "3.3. pont" → look for "3.3" in sources
        section_num = re.match(r'([\d.]+)', section)
        if section_num:
            num = section_num.group(1).rstrip(".")
            if num not in all_sources:
                missing_sections.append(section)

    if missing_sections:
        issues.append(f"Section reference(s) not found in sources: {', '.join(missing_sections)}")

    # Check for wrong program name
    wrong_program = False
    for pattern in WRONG_PROGRAM_PATTERNS:
        if pattern.search(answer_text):
            wrong_program = True
            issues.append("Answer mentions 'Otthonfelújítási Program' — should be 'Otthoni Energiatároló Program (OETP)'")
            break

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "referenced_sections": list(answer_sections),
        "wrong_program": wrong_program,
    }


def should_downgrade_confidence(check_result: dict) -> bool:
    """Determine if confidence should be lowered based on reference issues."""
    if check_result.get("wrong_program"):
        return True  # Always downgrade for wrong program
    if check_result.get("issues"):
        return True  # Any reference issue → downgrade
    return False
