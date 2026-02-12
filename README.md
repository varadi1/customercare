# Hanna 📋 — OETP Ügyfélszolgálati RAG Agent

Intelligens ügyfélszolgálati asszisztens az **OETP (Otthonfelújítási Program)** pályázathoz. A NEÜ (Nemzeti Energetikai Ügynökség) ügyfélszolgálati csapatát támogatja válasz-tervezetek készítésével.

## Architektúra

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│  OpenClaw Agent  │────▶│  FastAPI      │────▶│  ChromaDB   │
│  (Hanna)         │     │  Backend      │     │  Vector DB  │
│  Discord bot     │     │  :8101        │     │  :8100      │
└─────────────────┘     └──────┬───────┘     └─────────────┘
                               │
                    ┌──────────┼──────────┐
                    ▼          ▼          ▼
              ┌──────────┐ ┌────────┐ ┌──────────┐
              │ BGE      │ │ OpenAI │ │ MS Graph │
              │ Reranker │ │ API    │ │ API      │
              │ :8102    │ │ embed  │ │ email    │
              └──────────┘ └────────┘ └──────────┘
```

## RAG Pipeline

A keresési pipeline 5 lépésből áll:

1. **Query Expansion** — gpt-4o-mini kibontja a felhasználói kérdést 2-3 keresési variánsra
2. **Hybrid Retrieval** — Szemantikus (ChromaDB, text-embedding-3-small) + BM25 keyword keresés minden variánsra
3. **RRF Fusion** — Reciprocal Rank Fusion az összes találat összefésülésére
4. **Reranking** — Lokális BGE v2-m3 reranker (Cohere API fallback)
5. **Authority Weighting** — Forrástípus alapú súlyozás (felhívás > GYIK > email válasz)

### Authority Súlyok

| Típus | Súly | Leírás |
|-------|------|--------|
| `palyazat_felhivas` | 1.00 | Hivatalos pályázati felhívás |
| `palyazat_melleklet` | 0.95 | Felhívás mellékletei |
| `kozlemeny` | 0.85 | Hivatalos közlemények |
| `gyik` | 0.80 | Gyakran Ismételt Kérdések |
| `segedlet` | 0.75 | Kitöltési segédletek |
| `document` | 0.65 | Egyéb dokumentumok |
| `email_reply` | 0.50 | Korábbi email válaszok |

### Contextual Embeddings

Minden chunk kontextus prefix-et kap beágyazás előtt, ami javítja a szemantikus keresés pontosságát:

```
"Ez az OETP hivatalos pályázati felhívásának részlete. Forrás: Felhivas_OETP.pdf..."

[eredeti chunk szöveg]
```

## Tudásbázis

- **~9,700 chunk** a ChromaDB-ben
- Források: pályázati felhívás + mellékletek, GYIK, segédletek, EU közlemény, ~3800 email válasz
- Figyelt postaládák: `info@neuzrt.hu`, `lakossagitarolo@neuzrt.hu`

## Email Integráció

- **MS Graph API** — Outlook 365 shared mailbox polling (Azure AD App, application permissions)
- **Draft generálás** — RAG-alapú válasz-tervezet mentése a postaládába
- **Attachment analízis** — Képes csatolmányok elemzése

## Stack

| Komponens | Technológia |
|-----------|-------------|
| Agent | OpenClaw + Claude Opus 4.6 |
| Backend | FastAPI (Python) |
| Vector DB | ChromaDB |
| Embeddings | OpenAI text-embedding-3-small |
| Reranker | BGE v2-m3 (lokális, MPS GPU) |
| Query Expansion | gpt-4o-mini |
| Email | Microsoft Graph API |
| Hosting | Docker Compose (Mac Studio) |

## Futtatás

```bash
# Docker compose (ChromaDB + Backend)
cd ~/.openclaw/hanna
docker compose up -d

# Reranker service (natív, MPS)
# LaunchAgent: com.openclaw.hanna-reranker
~/.openclaw/hanna/reranker-service/start.sh
```

## API Endpoints

| Endpoint | Metódus | Leírás |
|----------|---------|--------|
| `/health` | GET | Health check |
| `/search` | POST | Hybrid RAG keresés |
| `/stats` | GET | Tudásbázis statisztikák |
| `/ingest/text` | POST | Szöveg ingestálás |
| `/ingest/pdf` | POST | PDF ingestálás |
| `/ingest/email-pair` | POST | Email Q&A pár ingestálás |
| `/emails/poll` | POST | Új emailek lekérdezése |
| `/emails/draft` | POST | Válasz-tervezet mentése |
| `/reranker/status` | GET | Reranker állapot |

## Környezeti változók (.env)

```env
OPENAI_API_KEY=...
COHERE_API_KEY=...          # Fallback reranker
GRAPH_TENANT_ID=...         # Azure AD
GRAPH_CLIENT_ID=...
GRAPH_CLIENT_SECRET=...
CHROMA_HOST=chromadb
```

---

*Készítette: Bob ⚡ — 2026-02-12*
