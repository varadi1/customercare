"""HyDE (Hypothetical Document Embeddings) for Hanna OETP.

Generates a hypothetical formal OETP program document passage from a user query,
then embeds THAT text for semantic search.
"""

from __future__ import annotations

import time

import httpx

from ..config import settings

HYDE_SYSTEM_PROMPT = """Te az Otthonfelújítási Program (OETP) hivatalos dokumentumainak szimulátora vagy. A felhasználó kérdésére generálj egy rövid (150-250 szó), formális OETP programdokumentum-részletet, amely tartalmazná a választ.

Szabályok:
- Használj hivatalos programnyelvet: "pályázati felhívás", "támogatási szerződés", "elszámolási határidő", "jogosultsági feltétel", "felhívás X. pontja szerint"
- Hivatkozz az OETP pályázati dokumentumokra, NEÜ (Nemzeti Energetikai Ügynökség) közleményekre
- Használj jellemző szerkezeteket: "A pályázó köteles...", "A támogatás feltétele...", "Az elszámolás során..."
- A szöveg legyen olyan, mint egy valódi OETP pályázati kiírás, GYIK válasz, vagy belső eljárásrend releváns szakasza
- NE válaszolj a kérdésre közvetlenül — generálj egy hipotetikus dokumentumrészletet, ami tartalmazná a választ
- Magyar nyelven írj

Kontextus: OETP = Otthoni Energiatároló Program, NEÜ = Nemzeti Energetikai Ügynökség, lakossági energetikai pályázat, napelem + energiatároló"""


async def generate_hypothetical_document_async(query: str) -> str | None:
    """Generate a hypothetical OETP document passage for the given query."""
    if len(query.split()) > 15:
        return None

    try:
        async with httpx.AsyncClient(timeout=settings.hyde_timeout) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.hyde_model,
                    "messages": [
                        {"role": "system", "content": HYDE_SYSTEM_PROMPT},
                        {"role": "user", "content": query},
                    ],
                    "max_tokens": settings.hyde_max_tokens,
                    "temperature": 0.4,
                },
            )
            resp.raise_for_status()
            hypo_doc = resp.json()["choices"][0]["message"]["content"].strip()
            if len(hypo_doc) < 30:
                return None
            return hypo_doc
    except Exception as e:
        print(f"[hanna-oetp] HyDE generation failed: {e}")
        return None


async def hyde_embed_query_async(query: str) -> list[float] | None:
    """Generate HyDE embedding: hypothetical doc → embed.

    Returns embedding vector, or None (caller falls back to raw query).
    """
    from .embeddings import embed_query

    t0 = time.time()
    hypo_doc = await generate_hypothetical_document_async(query)
    if hypo_doc is None:
        return None

    embedding = embed_query(hypo_doc)
    print(f"[hanna-oetp] HyDE: {len(hypo_doc)} chars, {time.time()-t0:.2f}s")
    return embedding
