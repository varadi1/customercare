# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CustomerCare is a config-driven, multi-layer RAG (Retrieval-Augmented Generation) backend for customer service (forked from Hanna, the OETP-specific version). Built on FastAPI, it processes incoming emails, searches a knowledge base, generates draft responses, and saves them to Outlook 365 via MS Graph API. The system prioritizes hallucination-free, source-faithful answers with 15 verification layers and a 5-level closed-loop learning system that improves from human corrections.

## Common Commands

```bash
# Build and run
docker-compose up -d                   # Start all (backend + db + langfuse)
docker-compose up -d --build backend   # Rebuild and start backend

# Tests
cd backend && python3 -m pytest tests/ -v                        # Unit tests
cd backend && ../.venv-eval/bin/python -m pytest tests/test_deepeval.py -v  # DeepEval quality (needs OPENAI_API_KEY)

# Evaluation
docker exec cc-backend python3 /app/scripts/eval_live.py --limit 20 --report    # Live email eval
../.venv-eval/bin/python scripts/eval_ragas_weekly.py --limit 30 --report           # RAGAS batch eval

# Ingest
docker exec cc-backend python3 /app/scripts/bulk_ingest_sent.py --from 2026-02-02 --to 2026-04-07
docker exec cc-backend python3 /app/scripts/ingest_subfolders.py --limit 500
docker exec cc-backend python3 /app/scripts/scrape_nffku_oetp.py

# Logs & monitoring
docker logs -f cc-backend
bash scripts/healthcheck_discord.sh    # Manual health check
```

## Architecture

### Service Topology

| Service | Port | Container | Runtime |
|---------|------|-----------|---------|
| **CC Backend** (FastAPI) | 8101 | cc-backend | Docker |
| **CC DB** (PostgreSQL+pgvector) | 5438 | cc-db | Docker |
| **Langfuse** (observability) | 3001 | cc-langfuse | Docker |
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
      → Confidence routing (low → "CC - emberi válasz kell")
```

### Database (cc-db, PostgreSQL+pgvector :5438)

**Database**: `customercare` (own container, init script: `backend/db/init_customercare.sql`)

Tables:
- `chunks` — RAG knowledge base (~1400 chunks, 1024-dim BGE-M3 embeddings, Hungarian tsvector, survival_rate)
- `kg_entities` / `kg_relations` / `kg_entity_chunks` — Knowledge Graph
- `reasoning_traces` — Email processing audit trail (query → draft → sent → outcome)
- `feedback_analytics` — Learning from draft-vs-sent differences (change_types, lesson, chunk_survival)
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
│   ├── ingest.py        # Chunk → enrich → embed → KG extract → PostgreSQL
│   ├── authority.py     # Authority weighting + priority injection
│   ├── guardrails.py    # 7 domain-specific rules (numerical, eligibility, etc.)
│   ├── cove.py          # Chain of Verification (claim-level fact check)
│   ├── selfcheck.py     # Multi-sample consistency check
│   ├── answer_alignment.py  # Echo/irrelevant detection → skip
│   ├── depersonalize.py # PII removal before RAG ingestion
│   ├── legal_check.py  # Eligibility claims → legal RAG verification
│   └── kg_extract.py   # Inline KG extraction (gpt-4o-mini, auto in ingest)
├── email/
│   ├── processor.py     # Autonomous pipeline: poll → filter → draft → save
│   ├── poller.py        # Graph API inbox polling
│   ├── drafts.py        # Outlook draft creation + final safety gate
│   ├── skip_filter.py   # Deterministic email classification
│   ├── history.py       # Sent items ingest (depersonalized)
│   ├── name_extractor.py # Extract real name from email body signature
│   └── feedback.py      # Draft vs. sent comparison loop + analytics trigger
├── reasoning/
│   ├── style_score.py   # 5-component style matching
│   ├── traces.py        # Reasoning trace storage
│   ├── authority_learner.py  # Dynamic authority adjustments + chunk survival
│   ├── authority_monitor.py  # Authority drift snapshots + Discord alerts
│   ├── feedback_analytics.py # LLM change categorization + chunk survival tracking
│   ├── gap_detector.py      # Missing knowledge detection from human additions
│   └── dspy_optimizer.py    # DSPy MIPROv2 prompt optimization
backend/scripts/
├── eval_live.py         # Live email eval (semantic + style + term overlap)
├── eval_ragas_weekly.py # RAGAS batch evaluation
├── eval_pipeline.py     # Baseline eval with difflib
├── bulk_ingest_sent.py  # Historical email bulk ingest by date range
├── ingest_subfolders.py # Inbox subfolder email ingest
├── scrape_nffku_oetp.py # NFFKU közlemény scraper (inline + accordion + downloads)
├── kg_backfill_new.py   # KG extraction backfill for existing chunks
├── run_dspy_optimization.py  # DSPy prompt optimization CLI
├── build_reranker_training_data.py  # Chunk survival → reranker training pairs
├── finetune_reranker.py     # BGE reranker fine-tuning (MPS GPU)
├── eval_reranker.py         # Base vs fine-tuned reranker comparison
backend/tests/
├── test_deepeval.py     # DeepEval: faithfulness + relevancy (25 golden set entries)
├── promptfoo/promptfooconfig.yaml  # Red-teaming: 12 adversarial tests
backend/db/
└── init_customercare.sql  # DB schema (auto-runs on container creation)
```

## Key Design Decisions

- **Standalone DB** — `cc-db` container with own pgvector volume. Init script ensures DB survives rebuilds.
- **Opus 4.6 primary** — Best instruction following for Hungarian customer service. Fallback: GPT-5.4, then Gemini.
- **Depersonalized RAG** — Email chunks have PII removed (names, OETP IDs, emails, phones → placeholders). Prevents name confusion and OETP ID leakage in drafts.
- **Skip over bad draft** — If CC can't answer properly (echo, irrelevant, insufficient), it skips instead of generating a bad draft. Better no draft than wrong draft.
- **Deterministic greeting + signature** — Never trust LLM for these. Name extracted from email body signature first (most reliable), falls back to Graph API with Hungarian name order + accent fix. Company senders get "Tisztelt Partnerünk!". Full NEÜ signature block — all code-based.
- **Legal risk check** — If draft makes eligibility claims (pályázhat, jogosult, támogatható), legal RAG (:8103) is consulted for contradictions. High risk → confidence=low.
- **LLM retry** — Each provider tried twice (2s backoff) before fallback: Opus → GPT-5.4 → Gemini = 6 attempts total.
- **LLM fail → skip** — If all providers fail, no draft created. Never dumps raw chunks as "response".
- **Internal email skip** — Emails from @neuzrt.hu/@nffku.hu are skipped immediately (Step 0).
- **No self-referencing** — CC replies FROM the configured mailbox, so never asks customers to "write to" that address.
- **Authority hierarchy** — felhívás (1.00) > melléklet (0.95) > közlemény (0.90) > GYIK (0.85) > segédlet (0.80) > dokumentum (0.55) > email (0.40/0.30).

## Learning System (5 levels)

Closed-loop learning from draft-vs-sent email differences:

```
Email → Draft → Outlook → Colleague edits → Sent
                                              ↓
Daily 05:00 → feedback.check_feedback()
  ├→ Match draft↔sent (conv/subject/body)
  ├→ Resolve EXISTING trace (not duplicate)
  ├→ L1: categorize_changes() → feedback_analytics table
  ├→ L1: compute_chunk_survival() → which chunks survived
  └→ L1: export_pair_to_langfuse() → DSPy training dataset
                  ↓
Weekly Mon 06:00 → scheduler
  ├→ L2: authority refresh (traces → per-category adjustments → search)
  ├→ L2: update_chunk_survival_rates (chunks.survival_rate)
  ├→ L2: authority drift report → Discord
  ├→ L4: gap_detector → cluster human additions → suggest missing chunks
  └→ L4: knowledge gap report → Obsidian
                  ↓
Manual (monthly) → run_dspy_optimization.py
  ├→ L3: trainset from reasoning_traces (SENT_AS_IS + SENT_MODIFIED)
  ├→ L3: MIPROv2 optimize → system prompt + few-shot
  └→ L3: push to Langfuse → main.py auto-picks up
                  ↓
Manual (quarterly) → finetune_reranker.py
  ├→ L5: training data from chunk survival (positive/negative pairs)
  ├→ L5: BGE reranker fine-tune (MPS GPU)
  └→ L5: eval_reranker.py → golden set comparison
```

| Level | Component | Trigger | Data Source |
|-------|-----------|---------|-------------|
| L1 | `feedback_analytics.py` | Daily feedback check | LLM categorization + SequenceMatcher |
| L2 | `authority_learner.py` + `authority_monitor.py` | Weekly scheduler | reasoning_traces outcomes |
| L3 | `dspy_optimizer.py` | Manual CLI | 30+ draft-sent pairs from Langfuse |
| L4 | `gap_detector.py` | Weekly scheduler | feedback_analytics.added_content |
| L5 | `finetune_reranker.py` | Manual CLI | chunk survival positive/negative pairs |

## Configuration

All config in `backend/app/config.py` via Pydantic Settings. Key env vars:

- `CC_PG_DSN` — PostgreSQL connection (default: `postgresql://klara:klara_docs_2026@cc-db:5432/customercare`)
- `AUTO_PROCESS_ENABLED` — Autonomous email processing (default: false)
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` — LLM providers
- `GRAPH_TENANT_ID`, `GRAPH_CLIENT_ID`, `GRAPH_CLIENT_SECRET` — MS Graph API
- `DISCORD_BOT_TOKEN`, `DISCORD_CHANNEL_ID` — Monitoring alerts
- `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` — Observability
- `FEEDBACK_ANALYTICS_ENABLED` — Feedback analytics (default: true)
- `DSPY_ENABLED` — DSPy prompt optimization (default: false)
- `RERANKER_MODEL_PATH` — Fine-tuned reranker model path (reranker service env)

## Monitoring

- **Healthcheck** (`scripts/healthcheck_discord.sh`): Every 5min via LaunchAgent, checks backend + DB + embeddings + reranker. Auto-restarts Docker containers, Discord alerts on failure/recovery.
- **NFFKU monitor** (`scripts/monitor_nffku.py`): Daily 06:15 via LaunchAgent, scrapes nffku.hu OETP page, hash-based change detection, auto-triggers Docker scraper+ingest on change.
- **Scheduler Discord**: Every 2h processing run sends summary to Discord (📬 polled, ✅ drafts, 🟢🟡🔴 confidence).
- **Authority drift**: Weekly authority adjustment snapshots + Discord alert on significant drift.
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
| Total chunks | ~1,580 |
| Golden set entries | 25 |

## Language

The codebase, documentation, and domain terminology are primarily in **Hungarian**. Variable names and code structure are in English.
