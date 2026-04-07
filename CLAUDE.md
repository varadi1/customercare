# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hanna is a multi-layer RAG (Retrieval-Augmented Generation) backend for the OETP (Otthoni Energiatároló Program — Home Energy Storage Program) customer service. Built on FastAPI, it processes incoming emails, searches a knowledge base, generates draft responses, and saves them to Outlook 365 via MS Graph API. The system prioritizes hallucination-free, source-faithful answers with 13 verification layers.

## Common Commands

```bash
# Build and run
docker-compose up -d                   # Start all (backend + db + langfuse)
docker-compose up -d --build backend   # Rebuild and start backend

# Tests
cd backend && python3 -m pytest tests/ -v                        # Unit tests
cd backend && ../.venv-eval/bin/python -m pytest tests/test_deepeval.py -v  # DeepEval quality (needs OPENAI_API_KEY)

# Evaluation
docker exec hanna-backend python3 /app/scripts/eval_live.py --limit 20 --report    # Live email eval
../.venv-eval/bin/python scripts/eval_ragas_weekly.py --limit 30 --report           # RAGAS batch eval

# Ingest
docker exec hanna-backend python3 /app/scripts/bulk_ingest_sent.py --from 2026-02-02 --to 2026-04-07
docker exec hanna-backend python3 /app/scripts/ingest_subfolders.py --limit 500
docker exec hanna-backend python3 /app/scripts/scrape_nffku_oetp.py

# Logs & monitoring
docker logs -f hanna-backend
bash scripts/healthcheck_discord.sh    # Manual health check
```

## Architecture

### Service Topology

| Service | Port | Container | Runtime |
|---------|------|-----------|---------|
| **Hanna Backend** (FastAPI) | 8101 | hanna-backend | Docker |
| **Hanna DB** (PostgreSQL+pgvector) | 5438 | hanna-db | Docker |
| **Langfuse** (observability) | 3001 | hanna-langfuse | Docker |
| **BGE-M3 Embedding** (search) | 8104 | — | macOS LaunchAgent, MPS GPU |
| **BGE-M3 Embedding** (ingest) | 8114 | — | macOS LaunchAgent, MPS GPU |
| **BGE Reranker v2-m3** | 8102 | — | macOS LaunchAgent, MPS GPU |

### Draft Generation Pipeline (15 verification layers)

```
Email → Skip filter (auto-reply, internal @neuzrt.hu, thank-you)
      → RAG search + temporal staleness check
      → Enrichment prefix strip (internal metadata removed from facts)
      → VerbatimRAG fact extraction
      → LLM generation (Opus 4.6 primary, GPT-5.4 fallback, 2 retries/provider)
      → "skip" confidence → no draft if insufficient info
      → Deterministic greeting (email body signature > Graph API > fallback)
      → Deterministic NEÜ signature block
      → [N] citation strip (internal only, not customer-facing)
      → Citation validation (uncited claims → medium confidence)
      → Domain guardrails (7 rules + AI-speak detector)
      → Accent guard (accent-free → low confidence)
      → NLI faithfulness verification
      → CoVe (Chain of Verification — claim-level fact check)
      → Answer-Question Alignment (echo/irrelevant → SKIP, no draft)
      → ⚖️ Legal risk check (eligibility claims → legal RAG verification)
      → SelfCheck (multi-sample consistency, medium conf only)
      → Final accent gate (drafts.py — blocks accent-free drafts entirely)
      → Confidence routing (low → "Hanna - emberi válasz kell")
```

### Database (hanna-db, PostgreSQL+pgvector :5438)

**Database**: `hanna_oetp` (own container, init script: `backend/db/init_hanna_oetp.sql`)

Tables:
- `chunks` — RAG knowledge base (~1400 chunks, 1024-dim BGE-M3 embeddings, Hungarian tsvector)
- `kg_entities` / `kg_relations` / `kg_entity_chunks` — Knowledge Graph
- `reasoning_traces` — Email processing audit trail
- `canonical_entities` / `entity_links` — Cross-RAG sync

### Key Directories

```
backend/app/
├── main.py              # FastAPI endpoints + draft generation pipeline
├── llm_client.py        # Multi-provider LLM: Opus 4.6 → GPT-5.4 → Gemini
├── config.py            # Pydantic Settings (env vars)
├── observability.py     # Langfuse tracing wrapper
├── rag/
│   ├── search.py        # Hybrid search: semantic + BM25 + KG → RRF → rerank
│   ├── ingest.py        # Chunk → enrich → embed → PostgreSQL
│   ├── authority.py     # Authority weighting + priority injection
│   ├── guardrails.py    # 7 domain-specific rules (numerical, eligibility, etc.)
│   ├── cove.py          # Chain of Verification (claim-level fact check)
│   ├── selfcheck.py     # Multi-sample consistency check
│   ├── answer_alignment.py  # Echo/irrelevant detection → skip
│   ├── depersonalize.py # PII removal before RAG ingestion
│   └── legal_check.py  # Eligibility claims → legal RAG verification
├── email/
│   ├── processor.py     # Autonomous pipeline: poll → filter → draft → save
│   ├── poller.py        # Graph API inbox polling
│   ├── drafts.py        # Outlook draft creation + final safety gate
│   ├── skip_filter.py   # Deterministic email classification
│   ├── history.py       # Sent items ingest (depersonalized)
│   ├── name_extractor.py # Extract real name from email body signature
│   └── feedback.py      # Draft vs. sent comparison loop
├── reasoning/
│   ├── style_score.py   # 5-component style matching
│   ├── traces.py        # Reasoning trace storage
│   └── authority_learner.py  # Dynamic authority adjustments
backend/scripts/
├── eval_live.py         # Live email eval (semantic + style + term overlap)
├── eval_ragas_weekly.py # RAGAS batch evaluation
├── eval_pipeline.py     # Baseline eval with difflib
├── bulk_ingest_sent.py  # Historical email bulk ingest by date range
├── ingest_subfolders.py # Inbox subfolder email ingest
├── scrape_nffku_oetp.py # NFFKU közlemény scraper
backend/tests/
├── test_deepeval.py     # DeepEval: faithfulness + relevancy (25 golden set entries)
├── promptfoo/promptfooconfig.yaml  # Red-teaming: 12 adversarial tests
backend/db/
└── init_hanna_oetp.sql  # DB schema (auto-runs on container creation)
```

## Key Design Decisions

- **Standalone DB** — `hanna-db` container with own pgvector volume. Init script ensures DB survives rebuilds.
- **Opus 4.6 primary** — Best instruction following for Hungarian customer service. Fallback: GPT-5.4, then Gemini.
- **Depersonalized RAG** — Email chunks have PII removed (names, OETP IDs, emails, phones → placeholders). Prevents name confusion and OETP ID leakage in drafts.
- **Skip over bad draft** — If Hanna can't answer properly (echo, irrelevant, insufficient), it skips instead of generating a bad draft. Better no draft than wrong draft.
- **Deterministic greeting + signature** — Never trust LLM for these. Name extracted from email body signature first (most reliable), falls back to Graph API with Hungarian name order + accent fix. Company senders get "Tisztelt Partnerünk!". Full NEÜ signature block — all code-based.
- **Legal risk check** — If draft makes eligibility claims (pályázhat, jogosult, támogatható), legal RAG (:8103) is consulted for contradictions. High risk → confidence=low.
- **LLM retry** — Each provider tried twice (2s backoff) before fallback: Opus → GPT-5.4 → Gemini = 6 attempts total.
- **LLM fail → skip** — If all providers fail, no draft created. Never dumps raw chunks as "response".
- **Internal email skip** — Emails from @neuzrt.hu/@nffku.hu are skipped immediately (Step 0).
- **No self-referencing** — Hanna replies FROM lakossagitarolo@neuzrt.hu, so never asks customers to "write to lakossagitarolo@neuzrt.hu".
- **Authority hierarchy** — felhívás (1.00) > melléklet (0.95) > közlemény (0.90) > GYIK (0.85) > segédlet (0.80) > dokumentum (0.55) > email (0.40/0.30).

## Configuration

All config in `backend/app/config.py` via Pydantic Settings. Key env vars:

- `HANNA_PG_DSN` — PostgreSQL connection (default: `postgresql://klara:klara_docs_2026@hanna-db:5432/hanna_oetp`)
- `AUTO_PROCESS_ENABLED` — Autonomous email processing (default: false)
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` — LLM providers
- `GRAPH_TENANT_ID`, `GRAPH_CLIENT_ID`, `GRAPH_CLIENT_SECRET` — MS Graph API
- `DISCORD_BOT_TOKEN`, `DISCORD_CHANNEL_ID` — Monitoring alerts
- `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` — Observability

## Monitoring

- **Healthcheck** (`scripts/healthcheck_discord.sh`): Every 5min via LaunchAgent, checks backend + DB + embeddings + reranker. Auto-restarts Docker containers, Discord alerts on failure/recovery.
- **Scheduler Discord**: Every 2h processing run sends summary to Discord (📬 polled, ✅ drafts, 🟢🟡🔴 confidence).
- **DB data check**: Verifies chunks table is non-empty after container restart.

## Evaluation Baseline (2026-04-07)

| Metric | Value |
|--------|-------|
| Golden set faithfulness (DeepEval) | **100%** (13/13) |
| MISMATCH rate | **0%** |
| OETP ID match | **100%** |
| Semantic similarity | 0.684 |
| Draft length ratio | ~1.7x colleague avg |
| Verification layers | 15 |
| Total chunks | 1,408 |
| Golden set entries | 25 |

## Language

The codebase, documentation, and domain terminology are primarily in **Hungarian**. Variable names and code structure are in English.
