# Hanna — Telepítési Útmutató

## Követelmények

| Komponens | Minimum | Ajánlott |
|-----------|---------|----------|
| **OS** | macOS 14+ (Apple Silicon) vagy Linux (CUDA GPU) | macOS 15 / Ubuntu 22.04 |
| **Python** | 3.12+ | 3.12 |
| **Docker** | Docker Desktop 4.x + Compose plugin | Legújabb |
| **RAM** | 16 GB | 32 GB |
| **GPU** | Apple M1+ (MPS) vagy NVIDIA (CUDA 12+) | M2 Pro+ / RTX 3090+ |
| **Disk** | 20 GB (modellek + DB) | 50 GB |

## Gyors Telepítés (5 perc)

```bash
# 1. Klónozás
git clone https://github.com/varadi1/hanna.git ~/DEV/hanna
cd ~/DEV/hanna

# 2. .env kitöltése
cp .env.example .env
# Szerkeszd: API kulcsok, MS Graph credentials

# 3. Teljes telepítés (Docker + natív GPU + LaunchAgents)
bash infra/install.sh

# 4. Első ingest
docker exec cc-backend python3 /app/scripts/scrape_nffku_oetp.py
```

## Részletes Telepítés

### 1. Környezeti változók

Másold és töltsd ki:
```bash
cp .env.example .env
```

**Kötelező kulcsok:**

| Változó | Leírás | Honnan |
|---------|--------|--------|
| `OPENAI_API_KEY` | GPT-5.4 fallback + KG extraction | platform.openai.com |
| `ANTHROPIC_API_KEY` | Opus 4.6 — elsődleges LLM | console.anthropic.com |
| `GRAPH_TENANT_ID` | MS Graph API tenant | portal.azure.com |
| `GRAPH_CLIENT_ID` | MS Graph app registration | portal.azure.com |
| `GRAPH_CLIENT_SECRET` | MS Graph app secret | portal.azure.com |
| `GRAPH_USER_EMAIL` | Hanna mailbox email | Azure AD |

**Opcionális kulcsok:**

| Változó | Leírás | Default |
|---------|--------|---------|
| `GOOGLE_API_KEY` | Gemini fallback | (nincs) |
| `COHERE_API_KEY` | Cohere reranker fallback | (nincs) |
| `DISCORD_BOT_TOKEN` | Monitoring alerts | (nincs) |
| `OETP_DB_PASSWORD` | OETP MySQL readonly | (nincs) |
| `AUTO_PROCESS_ENABLED` | Email processing | `false` |

### 2. Docker Szolgáltatások

Az `install.sh` automatikusan elindítja:

| Service | Container | Port | Leírás |
|---------|-----------|------|--------|
| PostgreSQL + pgvector | cc-db | 5438 | RAG tudásbázis + KG |
| FastAPI backend | cc-backend | 8101 | API + email pipeline |
| Langfuse | cc-langfuse | 3001 | Observability |
| Langfuse DB | cc-langfuse-db | — | Langfuse PostgreSQL |

```bash
# Manuális indítás
docker compose up -d

# Rebuild
docker compose up -d --build backend

# Logok
docker logs -f cc-backend
```

### 3. Natív GPU Szolgáltatások

Ezek Docker-en **kívül** futnak, hogy közvetlenül elérjék a GPU-t (MPS/CUDA).

| Service | Port | Model | Könyvtár |
|---------|------|-------|----------|
| BGE-M3 Search | 8104 | BAAI/bge-m3 | `~/DEV/local_llm/bge_m3/` |
| BGE-M3 Ingest | 8114 | BAAI/bge-m3 | `~/DEV/local_llm/bge_m3_ingest/` |
| BGE Reranker | 8102 | BAAI/bge-reranker-v2-m3 | `~/DEV/local_llm/reranker/` |

Az `install.sh`:
1. Létrehozza a könyvtárakat ha nem léteznek
2. Python venv-et készít a szükséges csomagokkal (`torch`, `sentence-transformers`, `fastapi`)
3. LaunchAgent-eket regisztrál (auto-start, auto-restart)

**Első indulás**: a modellek letöltése 2-5 percet vesz igénybe (~2 GB összesen).

**Kézi indítás** (ha LaunchAgent nem fut):
```bash
# BGE-M3 search
cd ~/DEV/local_llm/bge_m3 && .venv/bin/python app.py

# BGE-M3 ingest
cd ~/DEV/local_llm/bge_m3_ingest && .venv/bin/python app.py

# Reranker
cd ~/DEV/local_llm/reranker && .venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8102
```

**Linux (CUDA)**: a `torch` MPS-specifikus hívásokat (`torch.mps.empty_cache()`) CUDA megfelelőkre kell cserélni az `app.py`/`main.py` fájlokban. A modell és a logika azonos.

### 4. Állapot Ellenőrzés

```bash
# Pre-flight check (minden szolgáltatás)
bash scripts/preflight.sh

# Egyedi service health
curl http://localhost:8101/health   # Backend
curl http://localhost:8104/health   # BGE-M3 Search
curl http://localhost:8114/health   # BGE-M3 Ingest
curl http://localhost:8102/health   # Reranker
```

### 5. Adat Ingest

```bash
# NFFKU OETP oldal scrapelése + ingest
docker exec cc-backend python3 /app/scripts/scrape_nffku_oetp.py

# Email history bulk ingest
docker exec cc-backend python3 /app/scripts/bulk_ingest_sent.py --from 2026-02-02 --to 2026-04-07

# Inbox subfolder ingest
docker exec cc-backend python3 /app/scripts/ingest_subfolders.py --limit 500
```

## Automatikus Folyamatok

| Folyamat | Ütemezés | LaunchAgent |
|----------|----------|-------------|
| Healthcheck + auto-restart | 5 percenként | `com.openclaw.hanna-healthcheck` |
| NFFKU monitoring + ingest | Naponta 06:15 | `com.openclaw.nffku-monitor` |
| Email processing | 2 óránként | Backend scheduler (ha `AUTO_PROCESS_ENABLED=true`) |
| Feedback analytics | Naponta 05:00 | Backend scheduler |
| Authority refresh | Hétfőnként 06:00 | Backend scheduler |

## Troubleshooting

### GPU service nem indul
```bash
# Logok ellenőrzése
cat /tmp/bge_m3_service.log
cat /tmp/hanna-reranker.log

# Kézi indítás debug módban
cd ~/DEV/local_llm/bge_m3 && .venv/bin/python app.py
```

### Backend nem éri el az embedding-et
```bash
# Docker-ből a host elérése: host.docker.internal
curl http://host.docker.internal:8104/health

# Ha nem működik, ellenőrizd a Docker Desktop beállításokat
# (Settings → General → "Use kernel networking" kikapcsolva)
```

### DB connection refused
```bash
# Container állapot
docker ps -a | grep cc-db

# Kézi restart
docker compose restart db
docker compose restart backend
```

### Langfuse nem elérhető
```bash
# Első induláskor automatikusan inicializálja a DB-t
docker compose up -d langfuse
# Nyisd meg: http://localhost:3001 — regisztráció szükséges
# API kulcsokat a Settings → API Keys menüben generálhatod
```
