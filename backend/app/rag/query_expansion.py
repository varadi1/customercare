"""Query expansion — domain-aware multi-query rewriting for OETP search.

Generates 4-5 query variants that use different terminology to bridge
the gap between colloquial user language and official document terms.

Key insight: users say "villanyóra", documents say "fogyasztásmérő".
Users say "törölni", documents say "elállás". Without synonym bridging,
semantic search and BM25 both fail on these queries.
"""

from __future__ import annotations

import json
from openai import OpenAI
from ..config import settings

_client: OpenAI | None = None

EXPANSION_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """Te egy OETP (Otthoni Energiatároló Program) domain-aware query rewriter vagy.

FELADATOD: A felhasználói kérdést 4-5 különböző keresési variánssá alakítod, amelyek ELTÉRŐ SZÓKINCCSEL közelítik meg ugyanazt. A cél: az összes releváns dokumentumot megtalálni, beleértve azokat is amelyek más terminológiát használnak.

KÖTELEZŐ SZABÁLYOK:
1. Az ELSŐ elem mindig az eredeti kérdés
2. A MÁSODIK elem HIVATALOS/FORMÁLIS szóhasználattal fogalmazza újra (pályázati felhívás nyelve)
3. A HARMADIK elem GYIK-stílusú egyszerű kérdés formájában
4. A 4-5. elem alternatív szakkifejezésekkel, szinonimákkal
5. JSON tömb, semmi más
6. Minden variáns MAGYAR nyelvű

OETP SZINONIMA SZÓTÁR (KRITIKUS — használd a variánsokban!):
- villanyóra = fogyasztásmérő = mérőóra = mérőhely = felhasználási hely
- POD azonosító = csatlakozási pont azonosító = felhasználási hely azonosító = POD kód
- törölni/visszavonni pályázatot = elállás = pályázat megszüntetése = lemondás a támogatásról
- meghatalmazás = képviseleti jogosultság = megbízás = felhatalmazás pályázat benyújtására
- betáplálás = hálózatra visszatáplálás = villamosenergia-termelés hálózatba adása = közcélú hálózat
- KAÜ = Központi Azonosítási Ügynök = ügyfélkapu = elektronikus azonosítás
- helyrajzi szám = hrsz = ingatlan-nyilvántartási azonosító = tulajdoni lap szerinti azonosító
- energiatároló = akkumulátor = háztartási energiatároló = battery storage
- napelem = fotovoltaikus rendszer = PV rendszer = napelemes erőmű
- inverter = váltóirányító = feszültségátalakító
- szaldó elszámolás = nettó elszámolás = net metering
- hiánypótlás = kiegészítő dokumentumok benyújtása = pótlás
- igazolási szakasz = megvalósítás igazolása = projekt lezárás = elszámolás
- kivitelező = vállalkozó = telepítő = szerelő cég

KONTEXTUS: OETP = Otthoni Energiatároló Program (lakossági), NEÜ = Nemzeti Energetikai Ügynökség (lebonyolító)

PÉLDÁK:
Input: "mikor kapom a pénzt?"
Output: ["mikor kapom a pénzt?", "támogatás folyósításának határideje és feltételei", "Mikor utalják ki az OETP támogatást?", "kifizetési ütemezés OETP előleg végszámla", "pénzügyi teljesítés támogatási összeg átutalás"]

Input: "villanyóra nem az én nevemen van"
Output: ["villanyóra nem az én nevemen van", "fogyasztásmérő és pályázó személyének eltérése esetén szükséges dokumentumok", "Mi a teendő ha a mérőóra más nevén van?", "felhasználási hely tulajdonos eltérés csatlakozási pont", "villamos energia vételezési hely néveltérés pályázati jogosultság"]

Input: "törölni akarom a pályázatot"
Output: ["törölni akarom a pályázatot", "pályázattól való elállás feltételei és eljárásrendje", "Hogyan lehet visszavonni a benyújtott OETP pályázatot?", "pályázat megszüntetése lemondás támogatásról", "benyújtott kérelem visszavonása OETP eljárás"]"""


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def expand_query(query: str) -> list[str]:
    """Expand a user query into multiple search variants with domain synonyms.

    Returns:
        List of 4-5 search queries (original first, then domain-aware variants)
    """
    if len(query.split()) > 20:
        return [query]

    try:
        client = get_client()
        response = client.chat.completions.create(
            model=EXPANSION_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.2,
            max_tokens=350,
        )

        content = response.choices[0].message.content.strip()
        queries = json.loads(content)

        if isinstance(queries, list) and len(queries) >= 1:
            if query not in queries:
                queries = [query] + queries[:4]
            return queries[:5]

        return [query]

    except Exception as e:
        print(f"[hanna] Query expansion failed, using original: {e}")
        return [query]


async def expand_query_async(query: str) -> list[str]:
    """Async version of domain-aware query expansion (multi-provider)."""
    if len(query.split()) > 20:
        return [query]

    try:
        from ..llm_client import chat_completion

        llm_result = await chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.2,
            max_tokens=350,
            json_mode=True,
        )

        content = llm_result["content"].strip()
        # Handle both raw JSON array and wrapped JSON
        if content.startswith("["):
            queries = json.loads(content)
        else:
            # Try to extract JSON array from response
            import re
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                queries = json.loads(match.group())
            else:
                queries = [content]

        if isinstance(queries, list) and len(queries) >= 1:
            if query not in queries:
                queries = [query] + queries[:4]
            return queries[:5]

        return [query]

    except Exception as e:
        print(f"[hanna] Async query expansion failed: {e}")
        return [query]
