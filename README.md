# Hanna — Intelligens RAG Backend Multi-Tudásbázis Kereséssel

Többrétegű RAG (Retrieval-Augmented Generation) backend, amely az **OETP (Otthoni Energiatároló Program)** ügyfélszolgálatot, az **Obsidian** tudásbázist és a **Cross-RAG** entitás-szinkronizációt egyetlen FastAPI szolgáltatásban egyesíti. A rendszer célja a hallucinációmentes, forrás-hű és megbízható válaszgenerálás.

---

## Architektúra Áttekintés

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              HANNA BACKEND (:8101)                              │
│                              FastAPI + uvicorn                                  │
│                              hanna-backend Docker                               │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────┐   ┌──────────────┐   ┌─────────────────┐   ┌──────────────┐   │
│  │ OETP RAG    │   │ Obsidian RAG │   │ Email Integráció│   │ Cross-RAG    │   │
│  │ /search     │   │ /obsidian/*  │   │ /emails/*       │   │ /cross-rag/* │   │
│  │ /ingest/*   │   │ /obsidian/   │   │ /draft/*        │   │ Entity sync  │   │
│  │ /stats      │   │  search/     │   │ Outlook 365     │   │ between DBs  │   │
│  └──────┬──────┘   │  ingest/     │   └────────┬────────┘   └──────┬───────┘   │
│         │          │  graph/*     │            │                    │           │
│         │          └──────┬───────┘            │                    │           │
│         │                 │                    │                    │           │
│  ┌──────▼─────────────────▼────────────────────▼────────────────────▼────────┐  │
│  │                    PostgreSQL + pgvector (:5433)                          │  │
│  │  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  ┌───────────────┐   │  │
│  │  │ hanna_oetp  │  │ neu_docs     │  │ cross_rag  │  │ KG táblák     │  │  │
│  │  │ chunks      │  │ obsidian_    │  │ entities   │  │ kg_entities   │  │  │
│  │  │ (OETP docs, │  │ chunks       │  │ relations  │  │ kg_relations  │  │  │
│  │  │  emails)    │  │ (Vault MD)   │  │            │  │ kg_entity_    │  │  │
│  │  │             │  │              │  │            │  │   chunks      │  │  │
│  │  └─────────────┘  └──────────────┘  └────────────┘  └───────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
                    │                │               │
         ┌──────────▼────┐  ┌───────▼──────┐  ┌────▼─────────┐
         │ BGE-M3        │  │ BGE v2-m3    │  │ MS Graph API │
         │ Embedding     │  │ Reranker     │  │ Outlook 365  │
         │ :8104 (search)│  │ :8102        │  │ shared inbox │
         │ :8114 (ingest)│  │ MPS GPU      │  └──────────────┘
         │ MPS GPU       │  │ Cohere fallb │
         └───────────────┘  └──────────────┘
```

### Fájlstruktúra

```
~/.openclaw/hanna/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app, összes endpoint
│   │   ├── config.py               # Konfiguráció (pydantic-settings, env vars)
│   │   ├── models.py               # Pydantic modellek (request/response)
│   │   ├── analytics.py            # Használati analitika
│   │   ├── cross_rag_api.py        # Cross-RAG REST endpointok
│   │   ├── rag/
│   │   │   ├── search.py           # Hybrid keresés: semantic + BM25 → RRF → rerank → authority
│   │   │   ├── ingest.py           # Dokumentum ingestálás: chunk → enrich → embed → PostgreSQL
│   │   │   ├── chunker.py          # Token-alapú chunking (tiktoken cl100k_base)
│   │   │   ├── embeddings.py       # BGE-M3 embedding (lokális, OpenAI fallback)
│   │   │   ├── contextual.py       # Contextual enrichment: doc_type-alapú prefix az embedding elé
│   │   │   ├── hyde.py             # HyDE: hipotetikus dokumentum generálás → embedding
│   │   │   ├── query_expansion.py  # Domain-aware expansion: OETP szinonima szótár, 4-5 variáns
│   │   │   ├── compression.py     # Post-rerank contextual compression (noise filtering)
│   │   │   ├── adaptive_k.py      # Query complexity → dynamic retrieval depth
│   │   │   ├── reranker.py         # Reranking: lokális BGE v2-m3 (:8102) + Cohere fallback
│   │   │   ├── authority.py        # Authority weighting + floor + email cap + diversity loop
│   │   │   ├── confidence.py       # Multi-faktor confidence számítás (high/medium/low)
│   │   │   ├── references.py       # Cross-reference resolution (Felhívás 4.2 pont → chunk)
│   │   │   ├── bm25.py             # Legacy BM25 index (ChromaDB-ből maradt, PostgreSQL tsvector váltotta)
│   │   │   ├── kg_search.py        # Knowledge Graph keresés: entity → 1-hop → chunks
│   │   │   ├── post_ingest_kg_oetp.py  # Determinisztikus KG extraction (zero LLM cost)
│   │   │   └── llm_enrichment.py   # LLM-alapú context prefix generálás (batch + inline)
│   │   ├── email/
│   │   │   ├── poller.py           # Outlook 365 polling (MS Graph API)
│   │   │   ├── drafts.py           # Draft mentés Outlook-ba + Hanna kategóriák
│   │   │   ├── draft_context.py    # Draft context builder: RAG + stílus + template + feedback
│   │   │   ├── skip_filter.py      # Skip filter: köszönő/autoreply/adatmódosítás kiszűrése
│   │   │   ├── style_learner.py    # Kolléga stílus mintafelismerés
│   │   │   ├── templates.py        # Válasz sablonok
│   │   │   ├── feedback.py         # Draft vs. elküldött összehasonlítás (tanulás)
│   │   │   ├── history.py          # Historikus email ingestálás
│   │   │   ├── attachments.py      # Csatolmány kezelés + GPT-4o Vision elemzés
│   │   │   ├── auth.py             # MS Graph API auth
│   │   │   └── draft_store.py      # Draft tárolás
│   │   └── obsidian/
│   │       ├── pg_ingest.py        # Obsidian vault → PostgreSQL szinkronizáció
│   │       ├── pg_search.py        # Obsidian hybrid keresés
│   │       ├── kg_extract.py       # Knowledge Graph entitás-kinyerés Obsidian-ból
│   │       ├── kg_search.py        # KG keresés az Obsidian gráfban
│   │       ├── cross_rag_enrich.py # Cross-RAG entitás szinkronizáció
│   │       ├── enrichment.py       # Obsidian kontextuális gazdagítás
│   │       ├── pg_schema.sql       # Obsidian DB séma
│   │       └── search.py           # Legacy keresés
│   ├── scripts/                    # Migrációs és karbantartó scriptek
│   ├── Dockerfile
│   └── requirements.txt
├── scripts/
│   ├── daily_ingest.sh             # Napi OETP email ingest (cron)
│   ├── obsidian_sync.sh            # Napi Obsidian vault szinkronizáció (cron)
│   ├── verify_ingest.py            # Egységes verifikáció (3 gyűjtemény)
│   ├── monitor_nffku.py            # Heti nffku.hu OETP oldal monitoring (LaunchAgent: hétfő 06:15)
│   └── migrate_bge_m3.py           # BGE-M3 migráció tool
├── backend/scripts/
│   └── rechunk_gyik.py             # GYIK PDF re-chunkolás kérdésenként (28 Q&A pár)
├── data/
│   ├── documents/                  # OETP PDF dokumentumok
│   ├── response_templates.json     # Válasz sablonok
│   ├── feedback_diffs.json         # Tanulási feedback (draft vs. elküldött)
│   ├── style_patterns.json         # Cached kolléga stílus minták
│   └── obsidian_hashes.json        # Obsidian sync hash state
├── reranker-service/               # Lokális reranker konfiguráció
├── docker-compose.yml              # Backend konténer definíció
└── .env                            # API kulcsok, credentials
```

---

## RAG Keresési Pipeline (OETP)

A keresési pipeline 7 lépésből áll, mindegyik a pontosságot és a hallucináció-mentességet szolgálja:

### 1. HyDE — Hypothetical Document Embeddings

**Fájl:** `rag/hyde.py`

A felhasználó rövid, pontatlan kérdéséből (pl. "mikor kapom a pénzt?") egy hipotetikus OETP programdokumentum-részletet generál gpt-4o-mini segítségével, majd AZT a szöveget embedeli. Ez dramatikusan javítja a szemantikus keresés pontosságát, mert a query embedding a dokumentum-térben lesz, nem a kérdés-térben.

- Csak rövid queryknél aktiválódik (≤15 szó)
- Domain-specifikus prompt: OETP szakkifejezéseket, hivatkozási formákat használ
- 3 másodperces timeout — ha nem sikerül, visszaesik raw query embeddingre
- Parallel fut a Query Expansion-nel (Stage 0)

### 2. Domain-Aware Query Expansion (Multi-Query Rewriting)

**Fájl:** `rag/query_expansion.py`

A felhasználói kérdést **4-5 alternatív keresési kifejezésre** bontja ki gpt-4o-mini-vel, beépített OETP szinonima szótárral. A cél: a colloquial ügyfélnyelv és a hivatalos dokumentum-terminológia közötti gap bridgelése.

- **OETP szinonima szótár**: villanyóra↔fogyasztásmérő, törölni↔elállás, POD↔csatlakozási pont, stb.
- **Struktúra**: eredeti + formális + GYIK-stílusú + 2× alternatív szakkifejezés
- Temperature: 0.2 (reprodukálható variánsok)
- Visszaesik az eredeti queryre, ha az LLM nem elérhető
- HyDE (Hypothetical Document Embeddings) kikapcsolva — a domain-aware expansion szuperszeálja

### 3. Hybrid Retrieval (Semantic + BM25)

**Fájl:** `rag/search.py`

Minden expanded queryhez párhuzamosan fut:

**Szemantikus keresés** — PostgreSQL + pgvector, BGE-M3 1024-dimenziós embeddingek, cosine distance (`<=>` operátor)

**BM25 kulcsszó keresés** — PostgreSQL tsvector (`content_tsvector` oszlop), `websearch_to_tsquery('hungarian', ...)`. OR-alapú tsquery: a természetes nyelvi kérdések (ügyfél emailek) nem dokumentum-terminológiával íródnak, ezért AND túl strict lenne.

**Knowledge Graph keresés** — Entity-alapú kibővítés: a queryből ILIKE-kal entitásokat talál a `kg_entities` táblában, majd 1-hop szomszédokat a `kg_relations`-ből, végül a kapcsolódó chunkokat a `kg_entity_chunks` join-ból kéri le.

**Priority source retrieval** — Külön kis keresés (top 3 per típus) kizárólag prioritásos dokumentumtípusokra (felhívás, gyik, segédlet, melléklet, közlemény). Ez garantálja, hogy a ~9000 email chunk ne nyomja el a ~550 hivatalos dokumentum chunkot.

### 4. Reciprocal Rank Fusion (RRF)

**Fájl:** `rag/search.py` → `_reciprocal_rank_fusion()`

Az összes keresési csatorna eredményeit (semantic + BM25 + KG + priority) egyetlen rangsorba fésüli. Képlet: `score = Σ(1 / (k + rank_i))`. Deduplikáció chunk ID alapján.

### 5. Diversity Cap + Priority Injection + Email Dedup

**Fájl:** `rag/search.py` → `_cap_per_source()`, `_inject_priority_chunks()`, `_get_email_group_key()`

- **Per-source cap**: Maximum 2 chunk forrásadokumentumonként
- **Email group cap**: Azonos email subfolder / mailbox max 2 chunk (a `sent:`, `email_reply:`, `subfolder:` prefixek csoportosítva)
- **Priority injection**: Ha egy prioritásos dokumentumtípus (pl. gyik) releváns volt a fused listában de nem jutott be a top-N candidates-be, lecseréli a leggyengébb email chunkot

### 5.5. Contextual Compression

**Fájl:** `rag/compression.py`

Post-rerank szűrés: eltávolítja a nagyon alacsony rerank_score-ú chunkokat, de **priority chunk típusokat soha nem szűr ki**. Score floor (0.005) + ratio vs top score (8%).

### 6. Reranking

**Fájl:** `rag/reranker.py`

Lokális **BGE v2-m3** reranker szolgáltatás (:8102, MPS GPU acceleration). Az eredeti (nem expanded) queryvel rerankel — ez kritikus, mert az expansion jó a recall-hoz, de a rerankernek a tényleges kérdést kell látnia a precision-höz.

- Fallback: **Cohere Rerank v3.5** API ha a lokális szolgáltatás nem elérhető
- Fallback: Ha egyik sem elérhető, az RRF sorrend marad

### 7. Authority Weighting + Floor

**Fájl:** `rag/authority.py`

Forrástípus-alapú súlyozás — a hivatalos dokumentumok mindig előnyt élveznek az email válaszokkal szemben:

| Típus | Súly | Leírás |
|-------|------|--------|
| `felhívás` | 1.00 | Pályázati felhívás — THE source of truth |
| `melléklet` | 0.95 | Felhívás mellékletei |
| `közlemény` | 0.90 | Hivatalos NEÜ közlemények |
| `gyik` | 0.85 | Gyakran Ismételt Kérdések |
| `segédlet` | 0.80 | Kitöltési segédletek, útmutatók |
| `dokumentum` | 0.55 | Általános dokumentumok |
| `email_reply` | 0.40 | Korábbi ügyfélszolgálati email válaszok |
| `email_question` | 0.30 | Beérkezett kérdések |

**Képlet:** `final_score = base_score × (1 - 0.55) + base_score × authority × 0.55`

**Authority floor garantálja:**
1. **Email cap**: max 2 email-típusú chunk a top 5-ben (email_reply, email_question, lesson)
2. **Priority floor**: prioritásos chunkokat promótálja a top 3-ba (threshold: 0.0 — ha a pipeline-on végigment, promótáljuk)
3. Ha a top 5 mind email → a legjobb nem-email felkerül a 2. pozícióba
4. **Diversity loop**: max 3 kör promóció — minden missing priority type-ot beemel a top 5-be (email slot-okat cserélve)

### 7.5. Adaptive k

**Fájl:** `rag/adaptive_k.py`

Query complexity classification → dynamic retrieval depth:
- **Simple** (k=5, retrieval=20): "Mi az OETP?", "Mennyi a támogatás?"
- **Medium** (k=7, retrieval=25): 2 koncepció, 10+ szó
- **Complex** (k=10, retrieval=30): multi-step, összehasonlítás, 3+ entitás

---

## Hallucináció-ellenes Védelmi Rétegek

### 1. Contextual Enrichment (Ingest-time)

**Fájl:** `rag/contextual.py`

Minden chunk elé egy kontextuális prefix kerül az embedding ELŐTT. A prefix tartalmazza a chunk típusát, forrását és a forrás autoritását. Például:

```
"Ez az OETP hivatalos pályázati felhívásának részlete. Forrás: Felhivas_OETP.pdf.
Ez a pályázat legfontosabb, legautoritatívabb dokumentuma."

[eredeti chunk szöveg]
```

Az enriched szöveg a `content_enriched` oszlopba kerül, az eredeti a `content`-be. Az embedding az enriched szövegből készül, de a keresési eredményben az enriched szöveg jelenik meg, hogy a kontextus ne vesszen el.

### 2. Cross-Reference Resolution

**Fájl:** `rag/references.py`

Ha egy keresési eredmény hivatkozik más dokumentumra (pl. "Felhívás 4.2. pont", "GYIK 12. pont", "1. számú melléklet"), a rendszer automatikusan feloldja a hivatkozást és behúzza a hivatkozott szekciót. Regex-alapú detektálás:

- `Felhívás X.Y. pont/fejezet` → felhívás chunk
- `GYIK N. pont/kérdés` → GYIK chunk
- `N. számú melléklet` → melléklet chunk
- `segédlet` hivatkozás → segédlet chunk

A hivatkozott chunkokat deduplikálva, a fő eredményektől elkülönítve adja vissza (`referenced_chunks` mező).

### 3. Relevance Gate + Abstain

**Fájl:** `main.py` → `/search` endpoint

Ha a legjobb találat score-ja a küszöbérték alatt van (alapértelmezés: 0.35), a rendszer NEM ad vissza hamis eredményt, hanem explicit jelzi: `"relevance_sufficient": false` + abstain message: *"A rendelkezésre álló dokumentumok alapján erre a kérdésre nem található megbízható válasz."*

### 4. Multi-Factor Confidence

**Fájl:** `rag/confidence.py`

A confidence (🟢/🟡/🔴) nem egyetlen szám, hanem több faktor kombinációja:

- **Template match** — ha a kérdés egy ismert válasz-sablonra illeszkedik → high
- **Authoritative source** — ha felhívás/gyik/melléklet chunk >0.45 score-ral a top 3-ban → high
- **Email-only support** — ha csak email válaszok támasztják alá, dokumentum backing nélkül → medium/low
- **Freshness penalty** — ha az email válasz >30 napos → low (elavult info)
- **Score threshold** — <0.40 → low, ≥0.45 + document backing → high

### 5. Category-Specific Confidence Thresholds

**Fájl:** `email/draft_context.py` → `CATEGORY_CONFIDENCE_THRESHOLDS`

Nem minden téma egyformán megbízható a RAG-ban. Kategóriánként eltérő küszöbértékek:

- `inverter`, `napelem`, `szaldo`: high ≥0.45 (jól dokumentált témák)
- `meghatalmazott`: high ≥0.65 (összetett, gyakran változó szabályok)
- `ertesites_kau`: high ≥0.60 (KAÜ rendszer-specifikus kérdések)

### 6. VerbatimRAG Fact Extraction

**Fájl:** `main.py` → `/draft/generate` endpoint

A draft generálásnál opcionálisan egy külön VerbatimRAG szolgáltatás (:8108) kinyeri a top chunkokból az exact szövegrészleteket (span extraction), amelyek ténylegesen válaszolnak a kérdésre. Az LLM CSAK ezeket a verified spaneket kapja meg — nem a teljes chunkot.

### 7. NLI Faithfulness Verification

**Fájl:** `main.py` → `/draft/generate` endpoint

A generált draft-ot egy NLI (Natural Language Inference) szolgáltatás (:8107) ellenőrzi: a válasz minden állítása következik-e a forrás chunkokból? Ha `unfaithful` → confidence automatikusan `low`-ra esik.

### 8. Grounded LLM Prompt

**Fájl:** `main.py` → `DRAFT_GENERATE_SYSTEM`

Az LLM rendszerpromptja explicit megtiltja a hallucin%ációt:
- "CSAK az [ELLENŐRZÖTT TÉNY] blokkokban szereplő információkat használhatod"
- "SOHA ne egészítsd ki saját tudásból, ne találj ki dátumokat, összegeket, határidőket"
- "Ha a tények NEM fedik le a kérdést → írd meg hogy kollégánk hamarosan válaszol"
- JSON response format: `used_facts` listában jelöli, melyik tényeket használta

### 9. Skip Filter

**Fájl:** `email/skip_filter.py`

Olyan emaileket szűr ki, amelyekre NEM kell választ generálni, ezzel megelőzve a felesleges/hibás draft készítést:
- Köszönő/visszaigazoló levelek ("megkaptam", "köszönöm")
- Auto-reply / rendszerüzenetek ("out of office", "mailer-daemon")
- Adatmódosítási kérelmek (email/telefon változtatás — kolléga kezeli manuálisan)

### 10. Chunk Validity & Versioning

**Fájl:** `rag/ingest.py`, `rag/search.py`

- Minden chunk rendelkezik `valid_from`/`valid_to` metaadattal
- `only_valid=true` (alapértelmezés) automatikusan kiszűri a lejárt chunkokat
- `supersedes` mező: új verzió automatikusan érvényteleníti a régi chunkokat
- `content_hash`: duplikáció-elkerülés ingestáláskor

### 11. Jogi Kontextus Jelzés

**Fájl:** `email/draft_context.py` → `_needs_legal_context()`

Ha az email jogi/gazdasági kérdést tartalmaz (vállalkozás, de minimis, jogszabály, stb.), a rendszer jelzi: `should_consult_reka: true` — azaz a jogi RAG rendszert (Réka) is konzultálni kell. Nem próbálja meg maga megválaszolni a jogi kérdéseket.

---

## Knowledge Graph

### OETP KG (`hanna_oetp` DB)

**Fájl:** `rag/post_ingest_kg_oetp.py`

Determinisztikus, **zero LLM cost** entitás-kinyerés minden ingestált dokumentumhoz:

1. **Dokumentum entitás** — doc_type + source metaadatból
2. **Program entitás** — OETP/NPP2/Távhő
3. **Jogszabály-hivatkozás regex** — Magyar minták (`55/2025 Korm. rendelet`, `2011. évi CXCV. törvény`, EU rendeletek, rövidítések: Étv., Ptk., Kbt.)
4. **Text-match**: meglévő entitásokat (fogalom, szereplő, program, munkálat_típus) case-insensitive word-boundary regex-szel keres a chunk szövegében
5. **Entity-chunk linking**: `kg_entity_chunks` tábla (entity_id, chunk_id, confidence, extraction_method)
6. **Cross-RAG sync**: entitások szinkronizálása a központi `cross_rag` adatbázisba

**DB séma:**
- `kg_entities` (id, name, type, aliases, metadata)
- `kg_relations` (id, source_id, target_id, relation_type, source_chunk_id, weight)
- `kg_entity_chunks` (entity_id, chunk_id, confidence, extraction_method)

### Obsidian KG

**Fájl:** `obsidian/kg_extract.py`, `obsidian/kg_search.py`

Külön KG az Obsidian vault-hoz, saját entitásokkal és relációkkal.

---

## Embedding & Indexálás

### Embedding Modellek

**Fájl:** `rag/embeddings.py`

| Funkció | Modell | Port | Dimenzió |
|---------|--------|------|----------|
| Keresés | BGE-M3 (lokális, MPS GPU) | :8104 | 1024 |
| Ingestálás | BGE-M3 (dedikált instance) | :8114 | 1024 |
| Fallback | OpenAI text-embedding-3-small | API | 1024 |

Két külön BGE-M3 instance fut: az ingest instance (:8114) nem terheli a keresési instance-t (:8104) batch feldolgozáskor.

### Chunking

**Fájl:** `rag/chunker.py`

- **Token-alapú**: tiktoken `cl100k_base` encoding, 500 token chunk, 100 token overlap
- **Markdown-aware**: fejezet-szintű szétbontás, heading-ek kontextusként való megőrzése
- Nagy bekezdések automatikus force-split-je

### PostgreSQL séma (OETP — `hanna_oetp` DB)

```sql
chunks (
    id VARCHAR PRIMARY KEY,
    doc_id VARCHAR,             -- forrás dokumentum azonosító
    doc_type VARCHAR,           -- felhívás | melléklet | közlemény | gyik | segédlet | email_reply | ...
    program VARCHAR,            -- OETP | NPP2 | Távhő
    chunk_index INTEGER,
    title VARCHAR,
    content TEXT,               -- eredeti chunk szöveg
    content_enriched TEXT,      -- contextual prefix + eredeti szöveg
    embedding vector(1024),     -- BGE-M3 embedding (az enriched szövegből)
    content_tsvector tsvector,  -- BM25 kereséshez (Hungarian config)
    metadata JSONB,             -- valid_from, valid_to, version, supersedes, indexed_at
    authority_score FLOAT,      -- 0.30–1.00 (doc_type alapján)
    source_date TIMESTAMP,
    content_hash VARCHAR,       -- duplikáció-elkerülés
    created_at, updated_at
)
```

HNSW index a vector kereséshez, GIN index a tsvector-hoz.

### PostgreSQL séma (Obsidian — `neu_docs` DB)

```sql
obsidian_chunks (
    id SERIAL PRIMARY KEY,
    chunk_id VARCHAR(64) UNIQUE,
    file_path, file_name, folder,  -- PARA struktúra: inbox/projects/areas/resources/archive
    chunk_index INTEGER,
    content TEXT,
    embedding vector(1024),
    tsv tsvector,               -- auto-generated, 'simple' config
    file_hash VARCHAR(64),      -- incremental sync-hez
    context_prefix TEXT,        -- LLM-generated summary
    original_content TEXT,
    metadata JSONB
)
```

---

## Email Integráció

### MS Graph API (Outlook 365)

- **Auth**: Azure AD App, client credentials flow
- **Shared mailboxok**: `info@neuzrt.hu`, `lakossagienergetika@neuzrt.hu`
- **Polling**: 2 óránként (cron), 4 órás overlap ablakkal
- **Thread kezelés**: `conversationId` alapú + subject/sender fallback

### Draft Generálás Pipeline

1. **Skip filter** → köszönő/autoreply/admin kiszűrése
2. **RAG keresés** → hybrid pipeline a teljes email szöveggel
3. **Kategória detektálás** → email osztályozás (inverter, napelem, jogosultság, stb.)
4. **Stílus elemzés** → kolléga stílus minták (üdvözlés, lezárás, szóhossz, hangnem)
5. **Template matching** → ismert kérdéstípusokra sablon illesztés
6. **Feedback hints** → korábbi draft vs. elküldött összehasonlításból tanulságok
7. **VerbatimRAG** → verified fact extraction a top chunkokból
8. **LLM reformulation** → tények átfogalmazása koherens email válasszá
9. **NLI verification** → faithfulness ellenőrzés
10. **Confidence jelzés** → 🟢 high / 🟡 medium / 🔴 low ikon a draft-ban

### Feedback Loop

**Fájl:** `email/feedback.py`

A `/emails/feedback/check` endpoint összehasonlítja a Hanna által generált draft-ot a ténylegesen elküldött válasszal. A különbségek a `feedback_diffs.json`-be kerülnek, és a következő draft generálásnál figyelembe veszi őket.

---

## Obsidian RAG

### Szinkronizáció

**Fájl:** `obsidian/pg_ingest.py`, `scripts/obsidian_sync.sh`

- Hash-alapú incremental sync: csak a módosult fájlok kerülnek újra-indexelésre
- `obsidian_sync_state` tábla tárolja fájlonként a hash-t
- Vault mount: `/app/obsidian-vault` → `~/obsidian-git-backup`
- PARA struktúra felismerés (folder mező)

### Keresés

**Fájl:** `obsidian/pg_search.py`

Hybrid keresés (semantic + BM25) az Obsidian chunkokban, folder-szűréssel.

---

## Cross-RAG

Központi `cross_rag` DB, ahova minden tudásbázis (OETP, Obsidian) szinkronizálja az entitásait. Lehetővé teszi a rendszerek közötti keresést és entitás-összekapcsolást.

---

## Verifikáció

**Fájl:** `scripts/verify_ingest.py`

Egységes verifikációs script mind a 3 gyűjteményre:
1. **Obsidian RAG** — chunk count, last sync, stale files
2. **OETP/Email** — stats, KG health, doc_type distribution
3. **Cross-RAG** — entity count, system linkek

Futtatás: `python3 scripts/verify_ingest.py --fix --report`

---

## Stack

| Komponens | Technológia |
|-----------|-------------|
| Backend | FastAPI (Python), `hanna-backend` Docker konténer, :8101 |
| Vector DB | PostgreSQL + pgvector (hanna_oetp + neu_docs DB), :5433 |
| Embeddings | BGE-M3 (lokális, MPS GPU), :8104 (search) / :8114 (ingest) |
| Reranker | BGE v2-m3 (lokális, MPS GPU, :8102), Cohere v3.5 fallback |
| Query Expansion | gpt-4o-mini |
| HyDE | gpt-4o-mini |
| Draft Generation | gpt-4o-mini (JSON mode, few-shot) |
| NLI Verification | Lokális szolgáltatás, :8107 |
| VerbatimRAG | Lokális szolgáltatás, :8108 |
| Email | Microsoft Graph API (Outlook 365, Azure AD) |
| Knowledge Graph | PostgreSQL táblák (entities + relations + entity-chunks) |
| Csatolmány elemzés | GPT-4o Vision |
| Chunking | tiktoken (cl100k_base), 500 token / 100 overlap |
| Hosting | Docker Compose, Mac Studio M3 Ultra |

---

## Futtatás

```bash
# Docker Compose (Backend)
cd ~/.openclaw/hanna
docker compose up -d

# Natív szolgáltatások (LaunchAgent-ek):
# com.openclaw.bge-m3-search   — BGE-M3 embedding (:8104)
# com.openclaw.bge-m3-ingest   — BGE-M3 embedding (:8114)
# com.openclaw.hanna-reranker  — BGE v2-m3 reranker (:8102)
```

---

## API Endpoints

### OETP RAG

| Endpoint | Metódus | Leírás |
|----------|---------|--------|
| `/health` | GET | Health check (DB + search) |
| `/livez` | GET | Lightweight liveness probe (no DB) |
| `/search` | POST | Hybrid RAG keresés (7-lépéses pipeline) |
| `/stats` | GET | Tudásbázis statisztikák + KG stats |
| `/ingest/text` | POST | Szöveges dokumentum ingest + auto KG extraction |
| `/ingest/pdf` | POST | PDF upload és ingest |
| `/ingest/email-pair` | POST | Kérdés-válasz email pár ingest |
| `/ingest/expire` | POST | Dokumentum lejáratása (soft delete) |
| `/rag/find-chunks` | POST | Chunk keresés szöveg alapján |
| `/rag/invalidate` | POST | Chunk érvénytelenítés (valid_to = today) |

### Email

| Endpoint | Metódus | Leírás |
|----------|---------|--------|
| `/emails/poll` | POST | Új emailek lekérdezése (összes mailbox) |
| `/emails/thread/{mailbox}/{id}` | GET | Email thread lekérés |
| `/emails/draft` | POST | Válasz-tervezet mentése Outlook-ba |
| `/emails/drafts/{mailbox}` | GET | Drafts listázása |
| `/emails/mark-sent/{mailbox}` | POST | "Hanna - elküldve" kategória |
| `/emails/feedback/check` | POST | Draft vs. elküldött összehasonlítás |
| `/emails/history/ingest` | POST | Historikus email ingestálás |
| `/emails/{mailbox}/messages/{id}/attachments` | GET | Csatolmányok listázása |
| `/emails/{mailbox}/messages/{id}/analyze-images` | POST | Csatolmány képelemzés (GPT-4o Vision) |

### Draft Context & Generation

| Endpoint | Metódus | Leírás |
|----------|---------|--------|
| `/draft/context` | POST | RAG + stílus + template + feedback egy hívásban |
| `/draft/generate` | POST | Grounded email draft generálás (VerbatimRAG + NLI) |

### Stílus

| Endpoint | Metódus | Leírás |
|----------|---------|--------|
| `/style/analyze` | GET | Kolléga stílus elemzés |
| `/style/templates` | GET | Kategória sablonok |
| `/style/patterns` | GET | Cached stílus minták |

### Obsidian

| Endpoint | Metódus | Leírás |
|----------|---------|--------|
| `/obsidian/search/hybrid` | GET | Hybrid keresés (q, top_k) |
| `/obsidian/ingest` | POST | Vault szinkronizáció indítás |
| `/obsidian/ingest/status` | GET | Szinkronizáció állapot |
| `/obsidian/stats` | GET | Statisztika |
| `/obsidian/last-sync` | GET | Utolsó szinkron |
| `/obsidian/graph/*` | GET/POST | Knowledge Graph endpointok |

### Egyéb

| Endpoint | Metódus | Leírás |
|----------|---------|--------|
| `/reranker/status` | GET | Reranker állapot |
| `/bm25/rebuild` | POST | BM25 index újraépítés |
| `/cross-rag/*` | GET/POST | Cross-RAG entity endpointok |

---

## OpenClaw RAG Ökoszisztéma

Hanna az **OpenClaw** 5 rendszeres RAG ökoszisztéma egyik tagja:

| Rendszer | DB | Tartalom | Chunk szám |
|----------|---------|----------|------------|
| **Hanna OETP** | `hanna_oetp` | OETP pályázat, emailek, GYIK | ~10,000 |
| **Jogszabály RAG** | `jogszabaly_rag` | Magyar jogszabályok (Kbt, Étv, Ptk, EU) | ~95,000 |
| **NEÜ Docs** | `neu_docs` | Céges dokumentumok, szerződések | ~48,000 |
| **Obsidian RAG** | `neu_docs` | Személyes tudásbázis (PARA vault) | ~23,000 |
| **UAE Legal RAG** | `uae_legal_rag` | Emirátusi jogszabályok | ~18,000 |

A **Cross-RAG** rendszer entitás-szinten összeköti az 5 rendszert: ha egy fogalom (pl. "közbeszerzés") megjelenik a Hanna OETP tudásbázisban és a jogszabály RAG-ban is, az entitás mindkét helyről elérhető.

Minden rendszer ugyanazt az alap architektúrát használja (BGE-M3 + pgvector + hybrid search + reranking), de domain-specifikus kiegészítésekkel:
- **Hanna**: Authority weighting, email stílus tanulás, HyDE, draft generálás
- **Jogszabály RAG**: Hivatkozás-feloldás, topic→docid mapping, autoritási szint hierarchia
- **UAE Legal**: Determinisztikus + LLM KG extraction, free zone pattern matching

### 2026-os Best Practice Lefedettség

Egy 2026. márciusi cross-system audit szerint az architektúra a produkciós RAG best practice **~90%-át lefedi**:

| Komponens | Állapot |
|-----------|---------|
| Hybrid search (semantic + BM25 + RRF) | ✅ Minden rendszerben |
| Contextual enrichment (chunk-szintű) | ✅ Minden rendszerben |
| Knowledge Graph (entity extraction + linking) | ✅ Minden rendszerben |
| Reranking (lokális, GPU) | ✅ Minden rendszerben |
| Authority scoring | ✅ Jogi rendszerekben |
| HyDE query transformation | ✅ Hanna + jogszabály |
| Query expansion | ✅ Hanna |
| Cross-reference resolution | ✅ Hanna + jogszabály |
| Relevancia küszöb + abstain | ✅ Minden rendszerben |
| NLI faithfulness verification | ✅ Hanna |
| VerbatimRAG span extraction | ✅ Hanna |
| Structured JSON output + few-shot grounding | ✅ Minden rendszerben |
| Temporal filtering (valid_from/valid_to) | ✅ Minden rendszerben |
| Cascade routing (kérdéstípus → LLM) | ✅ Hanna |

### Cascade Routing

A `/draft/generate` endpoint `model` paraméterrel elfogadja a használandó LLM-et. Az alapértelmezett `gpt-4o-mini`, de a hívó rendszer (OpenClaw agent) a kérdés komplexitása és a retrieval confidence alapján dönthet:

- **Egyszerű ténykérdés + high confidence** → `gpt-4o-mini` (~$0.001/query)
- **Összetett/bizonytalan kérdés** → `gpt-4o` vagy nagyobb modell

Az OETP ügyfélszolgálati kérdések ~60-70%-a egyszerű ténykérdés, ezért a cascade routing **5-10x költségcsökkentést** eredményez a naiv "mindig a legerősebb modell" megközelítéshez képest.

---

*Készítette: Bob — 2026-02-12 | Frissítve: 2026-03-19*
