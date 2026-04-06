# Hanna Infrastructure

## Szolgáltatások

| Port | Szolgáltatás | Típus | LaunchAgent |
|------|-------------|-------|-------------|
| 8101 | Hanna backend | Docker | docker-compose.yml |
| 8102 | BGE v2-m3 reranker | LaunchAgent (MPS GPU) | com.openclaw.hanna-reranker.plist |
| 8104 | BGE-M3 search embedding | LaunchAgent (MPS GPU) | com.openclaw.bge-m3.plist |
| 8114 | BGE-M3 ingest embedding | LaunchAgent (MPS GPU) | com.openclaw.bge-m3-ingest.plist |
| 8191 | GLiNER2 NER | LaunchAgent | com.claude.gliner-server.plist |
| 5433 | PostgreSQL + pgvector | Külső (Docker) | — |
| 3307 | OETP MySQL (readonly) | Külső (szerver) | — |
| 8103 | Jogszabály RAG | Külső (Docker) | — |

## LaunchAgent-ek

| Plist | Ütemezés | Feladat |
|-------|----------|---------|
| com.openclaw.hanna-reranker | RunAtLoad + KeepAlive | BGE reranker service |
| com.openclaw.bge-m3 | RunAtLoad + KeepAlive | BGE-M3 search embedding |
| com.openclaw.bge-m3-ingest | RunAtLoad + KeepAlive | BGE-M3 ingest embedding |
| com.openclaw.reranker-healthcheck | 5 percenként | Reranker health check |
| com.openclaw.nffku-monitor | Hetente H 06:15 | NFFKU.hu OETP oldal monitoring |
| com.openclaw.obsidian-daily-sync | Naponta 04:00 | Obsidian vault → PostgreSQL sync |
| com.hanna.weekly-gap-report | Hetente H 06:00 | Knowledge gap riport |

## Beépített scheduler (Docker-ben)

| Ütemezés | Feladat |
|----------|---------|
| 2 óránként | Email poll + filter + draft generálás |
| Naponta 05:00 UTC | Feedback check (draft vs elküldött) |
| Naponta 06:00 UTC | Style patterns frissítés |
| Hetente H 06:00 UTC | Knowledge gap riport + authority refresh |

## Install

```bash
cd ~/DEV/hanna
bash infra/install.sh
```
