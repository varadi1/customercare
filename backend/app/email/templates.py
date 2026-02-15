"""Template system for common OETP email patterns."""

from __future__ import annotations

import re

TEMPLATES = {
    "pod_javitas": {
        "name": "POD azonosító javítás",
        "keywords": ["POD", "pod", "azonosító", "hibás", "elírás", "javít", "HU-XXXX"],
        "strong_keywords": ["POD", "pod", "HU-XXXX"],
        "confidence": "high",
        "response_template": """Tisztelt Pályázó!

Köszönjük megkeresését.

Tájékoztatjuk, hogy a pályázatában szereplő POD azonosító javítására a hiánypótlási szakaszban van lehetőség. Amennyiben a Támogató hiánypótlásra szólítja fel, a hiánypótlás keretében a helyes POD azonosítót meg tudja adni.

Kérjük, tartsa készenleten a villamosenergia-szolgáltatójától kapott számlát vagy szerződést, amelyen a helyes POD azonosító szerepel.

Üdvözlettel,
NEÜ Zrt.""",
    },
    "ertesitesi_kozpont": {
        "name": "Értesítési központ belépés",
        "keywords": ["értesítési központ", "belépés", "bejelentkezés", "nem találom", "nem látom", "KAÜ", "ügyfélkapu"],
        "strong_keywords": ["értesítési központ", "KAÜ", "ügyfélkapu"],
        "confidence": "high",
        "response_template": """Tisztelt Pályázó!

Köszönjük megkeresését.

Az Értesítési Központba az alábbi linken keresztül tud belépni:
https://otthonienergiatarolo.neuzrt.hu/

A belépéshez KAÜ (Központi Azonosítási Ügynök) azonosítás szükséges, amely Ügyfélkapus bejelentkezéssel történik.

Amennyiben a pályázatot meghatalmazott útján nyújtotta be, a pályázat a meghatalmazott fiókjában jelenik meg. Ebben az esetben a meghatalmazottján keresztül tud tájékozódni a pályázat állapotáról, vagy az Értesítési Központban meghatalmazottat tud beállítani, hogy Ön is láthassa a pályázatot.

Üdvözlettel,
NEÜ Zrt.""",
    },
    "tulajdonviszony": {
        "name": "Tulajdonviszony probléma",
        "keywords": ["tulajdoni lap", "tulajdonos", "tulajdonviszony", "helyrajzi szám", "hrsz", "nem egyezik"],
        "strong_keywords": ["tulajdoni lap", "tulajdonviszony", "hrsz"],
        "confidence": "medium",
        "response_template": """Tisztelt Pályázó!

Köszönjük megkeresését.

A pályázatban szereplő tulajdonviszonyokkal kapcsolatos eltérés a hiánypótlási szakaszban javítható. Amennyiben a Támogató hiánypótlásra szólítja fel, kérjük, csatolja a friss (30 napnál nem régebbi) tulajdoni lap másolatot, valamint szükség esetén a társasházi alapító okiratot vagy használati megállapodást.

Üdvözlettel,
NEÜ Zrt.""",
    },
    "nem_jogosult_telek": {
        "name": "Építési telek — nem jogosult",
        "keywords": ["építési telek", "üres telek", "nincs rajta ház", "épül a ház", "építés alatt"],
        "strong_keywords": ["építési telek", "üres telek"],
        "confidence": "high",
        "response_template": """Tisztelt Pályázó!

Köszönjük megkeresését.

Tájékoztatjuk, hogy az OETP pályázat keretében kizárólag meglévő, lakóingatlannak minősülő ingatlanra lehet pályázni. Építési telekre, illetve építés alatt álló ingatlanra a pályázat nem nyújtható be.

A pályázat benyújtásának feltétele, hogy a pályázat tárgyát képező ingatlan a pályázat benyújtásakor lakóingatlanként legyen nyilvántartva az ingatlan-nyilvántartásban.

Üdvözlettel,
NEÜ Zrt.""",
    },
    "gazdasagi_tevekenyseg": {
        "name": "Gazdasági tevékenység a lakóhelyen",
        "keywords": ["egyéni vállalkozó", "e.v.", "székhely", "telephely", "home office", "vállalkozás", "gazdasági tevékenység"],
        "strong_keywords": ["egyéni vállalkozó", "gazdasági tevékenység"],
        "confidence": "medium",
        "response_template": """Tisztelt Pályázó!

Köszönjük megkeresését.

Tájékoztatjuk, hogy a Pályázati Felhívás értelmében az ingatlanban a fenntartási időszak végéig nem folyhat gazdasági tevékenység. Amennyiben az ingatlan egyéni vállalkozás székhelyeként vagy telephelyeként van bejegyezve, illetve ha az ingatlanban home office keretében gazdasági tevékenységet végez, ez kizáró ok lehet.

A feltételeknek a fenntartási időszak végéig (a támogatási szerződés megkötésétől számított 3 évig) kell fennállniuk.

Javasoljuk, hogy a pályázat benyújtása előtt egyeztessen a székhely/telephely bejegyzés megszüntetéséről.

Üdvözlettel,
NEÜ Zrt.""",
    },
}


def match_template(email_text: str) -> tuple[str | None, float]:
    """Match email text against templates using keyword matching.

    Returns (template_key, confidence_score) or (None, 0.0) if no match.
    Threshold: at least 2 keyword matches OR 1 strong keyword + relevance.
    """
    text_lower = email_text.lower()
    best_key = None
    best_score = 0.0

    for key, tmpl in TEMPLATES.items():
        keywords = tmpl["keywords"]
        strong_keywords = tmpl.get("strong_keywords", [])

        # Count keyword matches
        matches = 0
        strong_matches = 0
        for kw in keywords:
            if kw.lower() in text_lower:
                matches += 1
                if kw in strong_keywords:
                    strong_matches += 1

        # Scoring: strong keyword match = 0.4, regular = 0.15
        score = strong_matches * 0.4 + (matches - strong_matches) * 0.15

        # Threshold: 2+ keywords OR 1 strong keyword
        if matches >= 2 or strong_matches >= 1:
            # Normalize to 0-1 range (cap at 1.0)
            normalized = min(score, 1.0)
            if normalized > best_score:
                best_score = normalized
                best_key = key

    if best_key is None:
        return (None, 0.0)

    return (best_key, round(best_score, 3))


def list_templates() -> list[dict]:
    """Return all templates as a list."""
    result = []
    for key, tmpl in TEMPLATES.items():
        result.append({
            "key": key,
            "name": tmpl["name"],
            "confidence": tmpl["confidence"],
            "keywords": tmpl["keywords"],
        })
    return result
