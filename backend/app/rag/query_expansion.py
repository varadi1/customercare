"""Query expansion — expand vague user queries into precise search terms."""

from __future__ import annotations

import json
from openai import OpenAI
from ..config import settings

_client: OpenAI | None = None

EXPANSION_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """Te egy magyar nyelvű query expansion modul vagy az OETP (Otthonfelújítási Program) ügyfélszolgálati rendszerhez.

A felhasználók sokszor pontatlanul, röviden kérdeznek. A te feladatod:
1. Megérteni mit akarnak valójában tudni
2. Generálni 2-3 alternatív keresési kifejezést, amelyek más-más szemszögből közelítik meg ugyanazt a kérdést

Szabályok:
- Minden keresési kifejezés MAGYAR nyelvű
- Használj szakkifejezéseket ahol releváns (pl. "kifizetés" → "folyósítás", "támogatás utalása")
- Az eredeti kérdés legyen az első elem
- Maximum 3 keresési kifejezés összesen (az eredeti + 2 alternatíva)
- JSON tömb formátumban válaszolj, semmi más

Kontextus: OETP = Otthonfelújítási Program, lakossági energetikai pályázat, NEÜ = Nemzeti Energetikai Ügynökség

Példa:
Input: "mikor kapom a pénzt?"
Output: ["mikor kapom a pénzt", "támogatás folyósítás határidő feltételei", "OETP kifizetési ütemezés mikor utalják"]

Input: "kell-e engedély?"
Output: ["kell-e engedély", "építési engedély szükségessége pályázat", "engedélyköteles tevékenység felújítás OETP"]"""


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def expand_query(query: str) -> list[str]:
    """Expand a user query into multiple search variants.
    
    Args:
        query: The original user query
        
    Returns:
        List of 2-3 search queries (original first, then expansions)
    """
    # Short queries benefit most from expansion
    # Very long/specific queries probably don't need it
    if len(query.split()) > 15:
        return [query]
    
    try:
        client = get_client()
        response = client.chat.completions.create(
            model=EXPANSION_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.3,
            max_tokens=200,
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON array
        queries = json.loads(content)
        
        if isinstance(queries, list) and len(queries) >= 1:
            # Ensure original query is first
            if query not in queries:
                queries = [query] + queries[:2]
            return queries[:3]
        
        return [query]
        
    except Exception as e:
        print(f"[hanna] Query expansion failed, using original: {e}")
        return [query]


async def expand_query_async(query: str) -> list[str]:
    """Async version of query expansion."""
    if len(query.split()) > 15:
        return [query]
    
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        
        response = await client.chat.completions.create(
            model=EXPANSION_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.3,
            max_tokens=200,
        )
        
        content = response.choices[0].message.content.strip()
        queries = json.loads(content)
        
        if isinstance(queries, list) and len(queries) >= 1:
            if query not in queries:
                queries = [query] + queries[:2]
            return queries[:3]
        
        return [query]
        
    except Exception as e:
        print(f"[hanna] Async query expansion failed: {e}")
        return [query]
