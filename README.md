# Hanna 📋 — OETP Ügyfélszolgálati RAG Agent

Intelligens ügyfélszolgálati asszisztens az **OETP (Otthoni Energiatároló Program)** pályázathoz. A NEÜ (Nemzeti Energetikai Ügynökség Zrt.) ügyfélszolgálati csapatát támogatja válasz-tervezetek készítésével.

## Architektúra

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│  OpenClaw Agent  │────▶│  FastAPI Backend      │────▶│  PostgreSQL     │
│  (Hanna)         │     │  :8101                │     │  + pgvector     │
│  Claude Opus 4.6 │     │  hanna-backend        │     │  hanna_oetp DB  │
└─────────────────┘     └──────────┬───────────┘     └─────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
              ┌──────────┐  ┌──────────┐  ┌──────────┐
              │ BGE-M3   │  │ BGE v2   │  │ MS Graph │
              │ Embed    │  │ Reranker │  │ API      │
              │ :8104    │  │ :8102    │  │ Outlook  │
              └──────────┘  └──────────┘  └──────────┘
```

## RAG Pipeline

A keresési pipeline 5 lépésből áll:

1. **Query Expansion** — gpt-4o-mini kibontja a felhasználói kérdést 2-3 keresési variánsra
2. **Hybrid Retrieval** — Szemantikus (pgvector, BGE-M3) + BM25 (PostgreSQL tsvector)
3. **RRF Fusion** — Reciprocal Rank Fusion az összes találat összefésülésére
4. **Reranking** — Lokális BGE v2-m3 reranker (:8102, MPS GPU)
5. **Authority Weighting** — Forrástípus alapú súlyozás

### Authority Súlyok

| Típus | Súly | Leírás |
|-------|------|--------|
| `palyazat_felhivas` | 1.00 | Hivatalos pályázati felhívás |
| `palyazat_melleklet` | 0.95 | Felhívás mellékletei |
| `kozlemeny` | 0.90 | Hivatalos közlemények |
| `gyik` / `faq` | 0.85 | Gyakran Ismételt Kérdések |
| `segedlet` | 0.80 | Kitöltési segédletek |
| `document` | 0.55 | Egyéb dokumentumok |
| `email_reply` | 0.40 | Korábbi email válaszok |
| `email_question` | 0.30 | Beérkezett kérdések |

## Tudásbázis

- **9,723 chunk** PostgreSQL+pgvector-ban
- **Knowledge Graph**: 932 entitás, 3,720 reláció
- Források: pályázati felhívás + mellékletek, GYIK, segédletek, közlemények, ~9,100 email Q&A pár
- Programok: OETP (9,665), NPP2/RRF (56), Távhő (2)

### Dokumentum típusok

| Típus | Darab |
|-------|-------|
| Email válasz | 8,750 |
| Email kérdés | 418 |
| Dokumentum | 319 |
| Felhívás | 147 |
| Segédlet | 31 |
| Melléklet | 28 |
| GYIK | 22 |
| Közlemény | 8 |

## Email Integráció

- **Outlook 365 shared mailbox** (MS Graph API, Azure AD App)
- Figyelt postaláda: `lakossagitarolo@neuzrt.hu`
- **Draft generálás** — RAG-alapú válasz-tervezet mentése Outlook-ba, Hanna kategóriákkal
- **Confidence jelzés** — 🟢 magas / 🟡 közepes / 🔴 alacsony
- **Attachment elemzés** — GPT-4o-mini Vision csatolmány analízis
- **Stílus elemzés** — Kolléga stílus mintafelismerés a természetes válaszokhoz

## Korábbi migráció

- **2026-02-21**: ChromaDB → PostgreSQL+pgvector migráció lezárva
- Embedding modell: OpenAI text-embedding-3-small → **BGE-M3** (lokális, :8104)
- ChromaDB (:8100) megszüntetve, konténer törölve

## Stack

| Komponens | Technológia |
|-----------|-------------|
| Agent | OpenClaw + Claude Opus 4.6 |
| Backend | FastAPI (Python), `hanna-backend` Docker konténer |
| Vector DB | PostgreSQL + pgvector (`hanna_oetp` DB) |
| Embeddings | BGE-M3 (lokális, :8104, MPS GPU) |
| Reranker | BGE v2-m3 (lokális, :8102, MPS GPU) |
| Query Expansion | gpt-4o-mini |
| Email | Microsoft Graph API (Outlook 365) |
| Knowledge Graph | PostgreSQL táblák (entitások + relációk) |
| Hosting | Docker Compose (Mac Studio M3 Ultra) |

## Futtatás

```bash
# Docker compose (Backend + PostgreSQL)
cd ~/.openclaw/hanna
docker compose up -d

# Reranker és Embedding servicek (natív, LaunchAgent-ek)
# com.openclaw.hanna-reranker (:8102)
# com.openclaw.bge-m3-search (:8104)
```

## API Endpoints

| Endpoint | Metódus | Leírás |
|----------|---------|--------|
| `/health` | GET | Health check |
| `/search` | POST | Hybrid RAG keresés |
| `/draft/context` | POST | RAG + style + IDs egy hívásban |
| `/stats` | GET | Tudásbázis statisztikák |
| `/emails/poll` | POST | Új emailek lekérdezése |
| `/emails/draft` | POST | Válasz-tervezet mentése |
| `/emails/thread/{mailbox}/{id}` | GET | Email thread lekérés |
| `/emails/{mailbox}/messages/{id}/analyze-images` | POST | Csatolmány képelemzés |
| `/emails/mark-sent/{mailbox}` | POST | "Elküldve" kategória |
| `/style/analyze` | GET | Kolléga stílus elemzés |
| `/style/templates` | GET | Kategória sablonok |
| `/style/patterns` | GET | Cached stílus minták |
| `/bm25/rebuild` | POST | BM25 index újraépítés |

---

*Készítette: Bob ⚡ — 2026-02-12 | Frissítve: 2026-03-11*
