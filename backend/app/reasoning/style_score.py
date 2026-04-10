"""
Style scoring — measures how well CC's style matches colleague conventions.

Components:
1. Greeting match (personalized > Pályázó > Érdeklődő > generic)
2. Length ratio (target: colleague avg ± 30%)
3. Closing match (Üdvözlettel + NEÜ signature)
4. Formality (conditional mood, polite forms)
5. Brevity (short answer for simple questions)

Returns 0.0-1.0 score.
"""
from __future__ import annotations


def compute_style_score(
    draft_text: str,
    colleague_text: str,
    sender_name: str = "",
) -> dict:
    """Compute style similarity between CC and colleague responses.

    Returns dict with overall score and component breakdown.
    """
    if not draft_text or not colleague_text:
        return {"overall": 0.0, "components": {}}

    # 1. Greeting match
    greeting_score = _score_greeting(draft_text, colleague_text, sender_name)

    # 2. Length ratio
    length_score = _score_length(draft_text, colleague_text)

    # 3. Closing match
    closing_score = _score_closing(draft_text, colleague_text)

    # 4. Formality match
    formality_score = _score_formality(draft_text, colleague_text)

    # 5. Brevity appropriateness
    brevity_score = _score_brevity(draft_text, colleague_text)

    # Weighted average
    weights = {
        "greeting": 0.20,
        "length": 0.25,
        "closing": 0.15,
        "formality": 0.15,
        "brevity": 0.25,
    }

    overall = (
        greeting_score * weights["greeting"]
        + length_score * weights["length"]
        + closing_score * weights["closing"]
        + formality_score * weights["formality"]
        + brevity_score * weights["brevity"]
    )

    return {
        "overall": round(overall, 3),
        "components": {
            "greeting": round(greeting_score, 3),
            "length": round(length_score, 3),
            "closing": round(closing_score, 3),
            "formality": round(formality_score, 3),
            "brevity": round(brevity_score, 3),
        },
    }


def _score_greeting(draft: str, colleague: str, sender_name: str) -> float:
    """Score greeting similarity."""
    d_line = draft.strip().split("\n")[0] if draft else ""
    c_line = colleague.strip().split("\n")[0] if colleague else ""

    # Exact match
    if d_line == c_line:
        return 1.0

    # Both personalized (different person = ok)
    if "Tisztelt" in d_line and "Tisztelt" in c_line:
        # Both use formal greeting — check if personalized
        h_personal = d_line.replace("Tisztelt ", "").replace("!", "")
        c_personal = c_line.replace("Tisztelt ", "").replace("!", "")

        if h_personal in ("Pályázó", "Érdeklődő") and c_personal in ("Pályázó", "Érdeklődő"):
            return 0.9  # Both use standard greeting
        if h_personal not in ("Pályázó", "Érdeklődő", "Meghatalmazott") and \
           c_personal not in ("Pályázó", "Érdeklődő", "Meghatalmazott"):
            return 1.0  # Both personalized
        if "Meghatalmazott" in d_line:
            return 0.2  # CC uses wrong greeting
        return 0.7  # Partial match

    return 0.3  # Different format


def _score_length(draft: str, colleague: str) -> float:
    """Score based on response length similarity."""
    d_len = len(draft)
    c_len = len(colleague)

    if c_len == 0:
        return 0.0

    ratio = d_len / c_len

    # Perfect: 0.7-1.3x colleague length
    if 0.7 <= ratio <= 1.3:
        return 1.0
    # Acceptable: 0.5-2.0x
    elif 0.5 <= ratio <= 2.0:
        return 0.7
    # Too different
    elif 0.3 <= ratio <= 3.0:
        return 0.4
    else:
        return 0.2


def _score_closing(draft: str, colleague: str) -> float:
    """Score closing similarity.

    CC always includes the NEÜ signature block (deterministic),
    so we check if it's present and well-formed rather than comparing
    with the colleague text (which may have the signature stripped).
    """
    d_lower = draft.lower()[-300:]
    has_udvozlettel = "üdvözlettel" in d_lower
    has_neu = "nemzeti energetikai" in d_lower
    has_address = "montevideo" in d_lower or "1037" in d_lower

    if has_udvozlettel and has_neu and has_address:
        return 1.0  # Full proper signature
    if has_udvozlettel and has_neu:
        return 0.8
    if has_udvozlettel:
        return 0.5
    return 0.2


def _score_formality(draft: str, colleague: str) -> float:
    """Score formality match (conditional mood, polite forms)."""
    d_lower = draft.lower()
    c_lower = colleague.lower()

    # Conditional mood markers
    conditional_words = ["amennyiben", "abban az esetben", "szíveskedj", "kérjük"]
    h_formal = sum(1 for w in conditional_words if w in d_lower)
    c_formal = sum(1 for w in conditional_words if w in c_lower)

    # Direct markers
    direct_words = ["kell", "szükséges", "kötelező", "tilos"]
    h_direct = sum(1 for w in direct_words if w in d_lower)
    c_direct = sum(1 for w in direct_words if w in c_lower)

    # Similar formality level
    if abs(h_formal - c_formal) <= 1 and abs(h_direct - c_direct) <= 1:
        return 1.0
    elif abs(h_formal - c_formal) <= 2:
        return 0.7
    else:
        return 0.4


def _score_brevity(draft: str, colleague: str) -> float:
    """Score whether CC's length is appropriate to the question complexity."""
    c_len = len(colleague)
    d_len = len(draft)

    # Short colleague response (<100 chars) = simple question → CC should also be short
    if c_len < 100:
        if d_len < 200:
            return 1.0  # Both brief
        elif d_len < 400:
            return 0.5  # CC over-explains
        else:
            return 0.2  # Way too long

    # Medium response (100-300)
    if c_len < 300:
        if d_len < 500:
            return 1.0
        else:
            return 0.6

    # Long response (>300) = complex question → both should be detailed
    return 0.9  # Both are detailed, fine
