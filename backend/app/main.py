"""Hanna Backend — FastAPI REST API for RAG + Email."""

import asyncio
import json
import os
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path

import httpx

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .config import settings
from .models import (
    DocumentIngest,
    IngestResult,
    SearchQuery,
    SearchResponse,
    SearchResult,
    DraftRequest,
    DraftResult,
    BatchPollResult,
    HealthResponse,
    AttachmentInfo,
    ImageAnalysisResult,
    AttachmentAnalysisResponse,
)
from .rag import search as rag_search
from .rag import ingest as rag_ingest
from .rag.bm25 import BM25Index
from .rag import reranker
from .email import poller, drafts, history, attachments, feedback, templates, style_learner, draft_context
from .obsidian import pg_ingest as obsidian_ingest
from .obsidian import pg_search as obsidian_search
from .obsidian import kg_extract
from .obsidian import kg_search as obsidian_kg_search
from .obsidian import cross_rag_enrich
from . import analytics
from . import cross_rag_api
from .rag.post_ingest_kg_oetp import post_ingest_kg_oetp


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown."""
    # Verify PostgreSQL connection
    try:
        stats = await rag_search._get_collection_stats_async()
        print(f"[hanna] PostgreSQL connected: {stats}")
    except Exception as e:
        print(f"[hanna] WARNING: PostgreSQL not reachable: {e}")
    
    # Initialize reranker (local service with Cohere fallback)
    reranker_mode = await reranker.initialize()
    print(f"[hanna] Reranker initialized: {reranker_mode}")
    
    yield


app = FastAPI(
    title="Hanna Backend",
    description="RAG pipeline + Email integration for OETP customer service",
    version="0.1.0",
    lifespan=lifespan,
)


# ─── Obsidian ingest job state (non-blocking) ───────────────────────────────
_obsidian_ingest_lock = asyncio.Lock()
_obsidian_ingest_status = {
    "running": False,
    "started_at": None,
    "finished_at": None,
    "last_result": None,
    "last_error": None,
}


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — full (DB + search)."""
    try:
        stats = await asyncio.wait_for(
            rag_search._get_collection_stats_async(), timeout=4.0
        )
        db_status = "connected"
        count = stats.get("total_chunks", 0)
    except (asyncio.TimeoutError, Exception):
        db_status = "disconnected"
        count = 0

    return HealthResponse(
        status="ok" if db_status == "connected" else "degraded",
        chromadb=db_status,  # field name kept for API compat (now PostgreSQL+pgvector)
        collection_count=count,
    )


@app.get("/livez")
async def liveness():
    """Lightweight liveness probe — no DB, no async I/O.
    Returns instantly even if event loop is under load.
    Docker healthcheck should use this endpoint."""
    return {"alive": True}


@app.get("/reranker/status")
async def reranker_status():
    """Get reranker status."""
    return reranker.get_status()


# ─── PDF Ingestion ────────────────────────────────────────────────────────────

@app.post("/ingest/pdfs")
async def ingest_pdfs(pdf_dir: str = "/app/data/pdfs"):
    """Ingest all PDFs from a directory into the knowledge base."""
    try:
        from .ingest_pdfs import ingest_all_pdfs
        import io, contextlib
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            ingest_all_pdfs(pdf_dir)
        return {"status": "ok", "log": f.getvalue()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── RAG: Ingestion ──────────────────────────────────────────────────────────

@app.post("/ingest/text", response_model=IngestResult)
async def ingest_text(doc: DocumentIngest, background_tasks: BackgroundTasks):
    """Ingest a text/markdown document."""
    try:
        is_markdown = doc.source.endswith(".md") or doc.chunk_type == "faq"
        count = rag_ingest.ingest_text(
            text=doc.text,
            source=doc.source,
            category=doc.category,
            chunk_type=doc.chunk_type,
            valid_from=doc.valid_from,
            valid_to=doc.valid_to,
            version=doc.version,
            supersedes=doc.supersedes,
            use_markdown_chunker=is_markdown,
        )
        BM25Index.get().invalidate()
        # KG extraction in background
        background_tasks.add_task(post_ingest_kg_oetp, doc.source)
        return IngestResult(
            chunks_created=count,
            source=doc.source,
            collection="postgresql",  # Legacy field (migrated from ChromaDB)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/pdf", response_model=IngestResult)
async def ingest_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    category: str = Form("general"),
    chunk_type: str = Form("document"),
    valid_from: str | None = Form(None),
    valid_to: str | None = Form(None),
    version: int = Form(1),
    supersedes: str | None = Form(None),
):
    """Upload and ingest a PDF document."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")

    try:
        # Save temp file
        temp_path = Path(f"/tmp/{file.filename}")
        content = await file.read()
        temp_path.write_bytes(content)

        count = rag_ingest.ingest_pdf(
            pdf_path=str(temp_path),
            source=file.filename,
            category=category,
            chunk_type=chunk_type,
            valid_from=valid_from,
            valid_to=valid_to,
            version=version,
            supersedes=supersedes,
        )

        temp_path.unlink(missing_ok=True)

        BM25Index.get().invalidate()
        # KG extraction in background
        background_tasks.add_task(post_ingest_kg_oetp, file.filename)
        return IngestResult(
            chunks_created=count,
            source=file.filename,
            collection="postgresql",  # Legacy field (migrated from ChromaDB)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/email-pair", response_model=IngestResult)
async def ingest_email_pair(
    background_tasks: BackgroundTasks,
    question: str,
    answer: str,
    source: str = "email_archive",
    category: str = "general",
):
    """Ingest a question-answer email pair."""
    try:
        count = rag_ingest.ingest_email_pair(
            question_text=question,
            answer_text=answer,
            source=source,
            category=category,
        )
        BM25Index.get().invalidate()
        # KG extraction in background
        background_tasks.add_task(post_ingest_kg_oetp, source)
        return IngestResult(
            chunks_created=count,
            source=source,
            collection="postgresql",  # Legacy field (migrated from ChromaDB)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/expire")
async def expire_document(source: str, version_below: int | None = None):
    """Mark chunks from a source as expired."""
    count = rag_ingest.expire_chunks(source, version_below)
    BM25Index.get().invalidate()
    return {"expired_chunks": count, "source": source}


# ─── RAG: Search ──────────────────────────────────────────────────────────────

@app.post("/search", response_model=SearchResponse)
async def search(query: SearchQuery):
    """Hybrid search: semantic + BM25 → RRF → rerank → authority → cross-ref resolution."""
    try:
        results = await rag_search.search_async(
            query=query.query,
            top_k=query.top_k,
            category=query.category,
            chunk_type=query.chunk_type,
            only_valid=query.only_valid,
        )
        
        # Cross-reference resolution
        from app.rag.references import resolve_references_in_results
        from app.models import ReferencedChunk
        ref_chunks = []
        try:
            raw_refs = resolve_references_in_results(results, max_total_refs=5)
            ref_chunks = [ReferencedChunk(**r) for r in raw_refs]
            if ref_chunks:
                print(f"[hanna] Resolved {len(ref_chunks)} cross-references")
        except Exception as e:
            print(f"[hanna] Cross-ref resolution failed: {e}")
        
        # Score threshold filtering
        search_results = [SearchResult(**r) for r in results]
        if query.min_score and query.min_score > 0 and search_results:
            search_results = [r for r in search_results if (r.rerank_score or r.score or 0) >= query.min_score]

        # Relevance assessment
        top_score = search_results[0].rerank_score or search_results[0].score if search_results else 0.0
        relevance_threshold = query.min_score if query.min_score and query.min_score > 0 else 0.35
        relevance_sufficient = top_score >= relevance_threshold
        abstain_msg = None if relevance_sufficient else "A rendelkezésre álló dokumentumok alapján erre a kérdésre nem található megbízható válasz."

        return SearchResponse(
            results=search_results,
            referenced_chunks=ref_chunks,
            query=query.query,
            total_found=len(search_results),
            top_score=round(top_score, 4),
            relevance_sufficient=relevance_sufficient,
            abstain_message=abstain_msg,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def stats():
    """Knowledge base statistics."""
    return await rag_search._get_collection_stats_async()


# ─── RAG: Chunk Management ────────────────────────────────────────────────────

@app.post("/rag/find-chunks")
async def find_chunks(search_text: str, limit: int = 50):
    """Find chunks containing specific text (for review/invalidation)."""
    try:
        results = rag_search.find_chunks_by_text(search_text, limit)
        return {"matches": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class InvalidateRequest(BaseModel):
    chunk_ids: list[str]
    reason: str = ""


@app.post("/rag/invalidate")
async def invalidate_chunks_endpoint(req: InvalidateRequest):
    """Mark chunks as invalid (sets valid_to = today).
    
    Chunks are not deleted, just marked as outdated.
    Use only_valid=true in search to exclude them.
    """
    try:
        result = await rag_search.invalidate_chunks(req.chunk_ids, req.reason)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Email: Polling ───────────────────────────────────────────────────────────

@app.post("/emails/process")
async def process_emails(hours: float = 4):
    """Autonomous email processing — poll + filter + draft + save.

    Replaces OpenClaw agent orchestration. No debug blocks in drafts.
    Feature flag: AUTO_PROCESS_ENABLED (default: false).
    """
    from .email.processor import process_new_emails
    try:
        result = await process_new_emails(hours=hours)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/emails/poll", response_model=BatchPollResult)
async def poll_emails(hours: float | None = None):
    """Poll all shared mailboxes for new emails.

    Args:
        hours: if set, fetch emails from the last N hours (overrides saved state).
               Use hours=4 for overlap between 2-hourly cron runs.
    """
    try:
        results = await poller.poll_all_mailboxes(hours=hours)
        total = sum(r.new_emails for r in results)
        return BatchPollResult(results=results, total_new=total)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/emails/thread/{mailbox}/{conversation_id}")
async def get_thread(
    mailbox: str,
    conversation_id: str,
    subject: str | None = None,
    sender_email: str | None = None,
):
    """Get full email thread by conversation ID (with subject+sender fallback).

    If conversationId filter fails on the shared mailbox, falls back to
    subject-based search across Inbox + SentItems + Drafts.
    """
    try:
        messages = await poller.get_email_thread(
            mailbox, conversation_id,
            subject=subject, sender_email=sender_email,
        )
        return {"messages": [m.model_dump() for m in messages], "count": len(messages)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Email: Drafts ────────────────────────────────────────────────────────────

@app.post("/emails/draft", response_model=DraftResult)
async def create_draft(req: DraftRequest):
    """Create a draft reply in the shared mailbox."""
    try:
        result = await drafts.create_reply_draft(
            mailbox=req.mailbox,
            reply_to_message_id=req.reply_to_message_id,
            body_html=req.body_html,
            confidence=req.confidence,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/emails/drafts/{mailbox}")
async def list_drafts_endpoint(mailbox: str, limit: int = 20):
    """List recent drafts in a mailbox."""
    try:
        result = await drafts.list_drafts(mailbox, limit)
        return {"drafts": result, "count": len(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/emails/mark-sent/{mailbox}")
async def mark_sent_endpoint(mailbox: str, hours: int = 4):
    """Check Sent Items and update original emails to 'Hanna - elküldve'.
    
    Only modifies Hanna-prefixed categories, preserves all other categories.
    """
    try:
        result = await drafts.mark_sent_emails(mailbox, hours)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Draft Context ────────────────────────────────────────────────────────────

class DraftContextRequest(BaseModel):
    email_text: str
    email_subject: str = ""
    oetp_ids: list[str] = []
    pod_numbers: list[str] = []
    top_k: int = 5


@app.post("/draft/context")
async def get_draft_context(req: DraftContextRequest):
    """Get full draft context: RAG + style guide + examples + identifiers.
    
    Single call that gives everything needed to write a style-matched draft.
    """
    try:
        return await draft_context.build_draft_context(
            email_text=req.email_text,
            email_subject=req.email_subject,
            oetp_ids=req.oetp_ids,
            pod_numbers=req.pod_numbers,
            top_k=req.top_k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Draft Generate (grounded + coherent) ────────────────────────────────────

DRAFT_GENERATE_SYSTEM = """Te az OETP (Otthoni Energiatároló Program) ügyfélszolgálatának levélíró asszisztense vagy.

KRITIKUS: A program neve OETP = Otthoni Energiatároló Program.
NE keverd össze az "Otthonfelújítási Program"-mal — az egy MÁSIK program!

FELADATOD:
Egy beérkező ügyfél-emailre kell VÁLASZLEVÉL TERVEZETET írnod, KIZÁRÓLAG az alábbi ELLENŐRZÖTT TÉNYEK alapján.

SZIGORÚ SZABÁLYOK:
1. CSAK az [ELLENŐRZÖTT TÉNY] blokkokban szereplő információkat használhatod.
2. Ha a tények NEM fedik le a kérdést → írd meg hogy "kérdésére kollégánk hamarosan válaszol".
3. SOHA ne egészítsd ki saját tudásból, ne találj ki dátumokat, összegeket, határidőket.
4. Az ügyfél kérdésének MINDEN részére reagálj, ami a tényekből megválaszolható.
5. Ha a tények csak RÉSZBEN fedik le → válaszolj ami van, a többire jelezd hogy kollégánk válaszol.
6. Ha a tény KONKRÉT PONTSZÁMOT hivatkozik (pl. "3.3. pont", "4.1. pont"), MINDIG idézd a számot.
7. SOHA ne írd "Otthonfelújítási Program" — a program neve: Otthoni Energiatároló Program (OETP).

STÍLUS:
- Udvarias, hivatalos, de barátságos hangnem
- Tegezés SOHA, magázás/önözés MINDIG
- Használj feltételes módot: "amennyiben", "abban az esetben"
- TÖMÖRSÉG: Ha a válasz egyszerű (pl. "elvégeztük", "nincs lehetőség"), légy RÖVID — 1-3 mondat elég.
  NE fejts ki részletesen amit a kérdés nem kér.
- Ha a kérdés KOMPLEX (több részkérdés, technikai), válaszolj részletesen.
- NE kezdd "Köszönjük megkeresését" sablonnal, hanem rögtön a lényegre térj

VÁLASZ FORMÁTUM (szigorúan JSON):
{
  "body": "A levél szövege HTML formátumban (<p> tagekkel)",
  "confidence": "high|medium|low",
  "used_facts": [1, 2],
  "unanswered_parts": null
}"""

DRAFT_GENERATE_FEWSHOT = [
    {
        "role": "user",
        "content": """Beérkező email: "Mennyi a maximális támogatás és kell-e önerő?"
Tárgy: Támogatás összege

[ELLENŐRZÖTT TÉNY 1] (Felhivas_OETP.pdf)
"A támogatás összege legfeljebb 4.000.000 Ft lehet háztartásonként."

[ELLENŐRZÖTT TÉNY 2] (Felhivas_OETP.pdf)
"A pályázónak legalább 10% önerőt kell biztosítania a beruházás összköltségéhez képest."

Stílus: Tisztelt Pályázó! / Üdvözlettel:""",
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "body": "<p>Tisztelt Pályázó!</p><p>Tájékoztatjuk, hogy az Otthoni Energiatároló Program keretében a támogatás maximális összege háztartásonként 4.000.000 Ft. A pályázónak legalább 10% önerőt kell biztosítania a beruházás összköltségéhez képest.</p><p>Amennyiben további kérdése merülne fel, kérjük, forduljon hozzánk bizalommal.</p><p>Üdvözlettel:<br>Nemzeti Energetikai Ügynökség Zrt.</p>",
            "confidence": "high",
            "used_facts": [1, 2],
            "unanswered_parts": None,
        }, ensure_ascii=False),
    },
]


def _build_greeting(sender_name: str = "", category: str = "") -> str:
    """Build appropriate greeting based on sender name and email category.

    Priority:
    1. Personalized: "Tisztelt Kovács János!" (if name available and not generic)
    2. Category-based: "Tisztelt Érdeklődő!" (for general inquiries)
    3. Default: "Tisztelt Pályázó!" (matches colleague convention)
    """
    # Skip generic/empty names
    skip_names = {"", "null", "none", "info", "admin", "support"}
    name = (sender_name or "").strip()

    if name and name.lower() not in skip_names and len(name) > 2:
        # Check if it looks like a real name (not email prefix)
        if "@" not in name and "." not in name:
            return f"Tisztelt {name}!"

    # Category-based
    if category in ("altalanos", "hatarido"):
        return "Tisztelt Érdeklődő!"

    return "Tisztelt Pályázó!"


class DraftGenerateRequest(BaseModel):
    email_text: str
    email_subject: str = ""
    sender_name: str = ""
    sender_email: str = ""
    oetp_ids: list[str] = []
    pod_numbers: list[str] = []
    top_k: int = 5
    max_context_chunks: int = 3
    model: str | None = None


@app.post("/draft/generate")
async def draft_generate(req: DraftGenerateRequest):
    """Generate a grounded, coherent email draft.

    Pipeline:
    1. draft_context (RAG + skip filter + style guide)
    2. VerbatimRAG extracts verified spans from top chunks
    3. LLM reformulates spans into a coherent, polite email
    4. NLI verification (best-effort)
    """
    # 1. Draft context (includes skip filter)
    ctx = await draft_context.build_draft_context(
        email_text=req.email_text,
        email_subject=req.email_subject,
        oetp_ids=req.oetp_ids,
        pod_numbers=req.pod_numbers,
        top_k=req.top_k,
    )

    if ctx.get("skip"):
        return {
            "skip": True,
            "skip_reason": ctx["skip_reason"],
            "skip_category": ctx["skip_category"],
            "body_html": None,
            "confidence": "skip",
            "sources": [],
        }

    # 1b. Entity processing + reasoning trace (non-blocking)
    trace_id = None
    try:
        if req.sender_email:
            import asyncpg as _apg
            _econn = await _apg.connect(
                "postgresql://klara:klara_docs_2026@host.docker.internal:5433/hanna_oetp"
            )
            try:
                from app.reasoning.person_tracker import process_email_entities
                from app.reasoning.traces import create_trace
                from app.rag.embeddings import embed_query

                await process_email_entities(
                    conn=_econn,
                    sender_name=req.sender_name,
                    sender_email=req.sender_email,
                    oetp_ids=req.oetp_ids,
                    email_subject=req.email_subject,
                    category=ctx.get("category", ""),
                )

                query_emb = await embed_query(req.email_text[:500])
                trace_id = await create_trace(
                    conn=_econn,
                    query_text=req.email_text[:2000],
                    category=ctx.get("category", ""),
                    sender_name=req.sender_name,
                    sender_email=req.sender_email,
                    query_embedding=query_emb if query_emb else None,
                )
            finally:
                await _econn.close()
    except Exception as _e:
        import logging
        logging.getLogger(__name__).warning("Entity/trace processing failed (non-blocking): %s", _e)

    rag_results = ctx.get("rag_results", [])
    style = ctx.get("style_guide", {})

    # Smart greeting: personalized if sender name available, else category-based
    greeting = _build_greeting(req.sender_name, ctx.get("category", ""))

    if not rag_results:
        return {
            "skip": False,
            "body_html": f"<p>{greeting}</p><p>Köszönjük megkeresését. Kérjük türelmét, munkatársunk hamarosan részletes választ ad.</p><p>Üdvözlettel:<br>Nemzeti Energetikai Ügynökség Zrt.</p>",
            "confidence": "low",
            "sources": [],
            "method": "no_results",
        }

    top_chunks = rag_results[:req.max_context_chunks]
    top_score = top_chunks[0].get("score", 0) if top_chunks else 0

    # 2. Try VerbatimRAG for verified fact extraction
    verified_facts = []
    fact_sources = []
    try:
        verbatim_url = os.getenv("VERBATIM_SERVICE_URL", "http://host.docker.internal:8108")
        async with httpx.AsyncClient(timeout=5) as vc:
            vr = await vc.post(f"{verbatim_url}/extract", json={
                "question": req.email_text[:2000],
                "chunks": [r.get("text", "") for r in top_chunks],
            })
            if vr.status_code == 200:
                vdata = vr.json()
                if vdata.get("has_answer") and vdata.get("total_spans", 0) > 0:
                    for sr in vdata["results"]:
                        idx = sr["chunk_index"]
                        r = top_chunks[idx]
                        for span in sr["extracted_spans"]:
                            verified_facts.append({
                                "text": span,
                                "source": r.get("source", "?"),
                                "verified": True,
                            })
                            fact_sources.append({
                                "document": r.get("source", "?"),
                                "quote": span,
                                "verified": True,
                            })
    except Exception:
        pass  # VerbatimRAG unavailable — use raw chunks as fallback

    # Fallback: if VerbatimRAG failed, use raw chunk texts as "facts"
    if not verified_facts:
        for r in top_chunks:
            text = r.get("text", "")
            if text:
                verified_facts.append({
                    "text": text[:500],
                    "source": r.get("source", "?"),
                    "verified": False,
                })
                fact_sources.append({
                    "document": r.get("source", "?"),
                    "quote": text[:200],
                    "verified": False,
                })

    # 3. Build LLM prompt with verified facts
    facts_block = ""
    for i, f in enumerate(verified_facts, 1):
        verified_tag = "✓ ellenőrzött" if f["verified"] else "forrásból"
        facts_block += f'[ELLENŐRZÖTT TÉNY {i}] ({f["source"]}, {verified_tag})\n"{f["text"]}"\n\n'

    style_hint = f"Stílus: {greeting} / Üdvözlettel:"
    if style.get("tone_tips"):
        style_hint += "\nHangnem: " + "; ".join(style["tone_tips"][:2])

    user_msg = f"""Beérkező email: "{req.email_text[:1500]}"
Tárgy: {req.email_subject}

{facts_block}

{style_hint}"""

    # 3b. OETP applicant data (if available and OETP-ID in email)
    try:
        from app.reasoning.radix_client import enrich_draft_context, format_applicant_context
        oetp_data = await enrich_draft_context(oetp_ids=req.oetp_ids, sender_email=req.sender_email)
        if oetp_data:
            for app in oetp_data.get("applications", []):
                user_msg += "\n\n" + format_applicant_context(app)
    except Exception:
        pass

    # 4. LLM generation — reformulate facts into coherent email
    from openai import OpenAI
    model = req.model or "gpt-4o-mini"
    client = OpenAI(api_key=settings.openai_api_key)

    try:
        messages = [
            {"role": "system", "content": DRAFT_GENERATE_SYSTEM},
            *DRAFT_GENERATE_FEWSHOT,
            {"role": "user", "content": user_msg},
        ]
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.15,
            max_tokens=1000,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        draft_data = json.loads(raw)
    except Exception as e:
        # LLM failed — return raw facts as fallback
        facts_html = "".join(f"<p>{f['text']}</p>" for f in verified_facts[:3])
        return {
            "skip": False,
            "body_html": f"<p>{greeting}</p>{facts_html}<p>Üdvözlettel:<br>Nemzeti Energetikai Ügynökség Zrt.</p>",
            "confidence": "low",
            "sources": fact_sources,
            "method": "llm_fallback",
            "error": str(e),
        }

    body_html = draft_data.get("body", "")
    confidence = draft_data.get("confidence", "medium")

    # 5. NLI faithfulness verification (best-effort)
    nli_result = None
    try:
        chunk_texts = " ".join(f["text"] for f in verified_facts)
        nli_url = os.getenv("NLI_SERVICE_URL", "http://host.docker.internal:8107")
        async with httpx.AsyncClient(timeout=10) as nli_client:
            nli_resp = await nli_client.post(f"{nli_url}/verify-answer", json={
                "answer": body_html,
                "context": chunk_texts,
            })
            if nli_resp.status_code == 200:
                nli_result = nli_resp.json()
                if nli_result.get("overall_verdict") == "unfaithful":
                    confidence = "low"
    except Exception:
        pass

    return {
        "skip": False,
        "body_html": body_html,
        "confidence": confidence,
        "sources": fact_sources,
        "method": "verbatim+llm" if any(f["verified"] for f in verified_facts) else "chunks+llm",
        "model_used": model,
        "facts_count": len(verified_facts),
        "nli_verification": nli_result,
        "suggested_confidence": ctx.get("suggested_confidence"),
        "similar_traces": ctx.get("similar_traces", []),
        "trace_id": trace_id,
    }


# ─── Email: Feedback ──────────────────────────────────────────────────────────

def _default_feedback_mailbox() -> str:
    """Return the first configured shared mailbox as the feedback default."""
    mailboxes = [m.strip() for m in settings.shared_mailboxes.split(",") if m.strip()]
    return mailboxes[0] if mailboxes else "info@neuzrt.hu"


@app.post("/emails/feedback/check")
async def feedback_check(mailbox: str | None = None, hours: int = 48):
    """Compare sent emails with stored Hanna drafts.
    
    Returns how many drafts were accepted unchanged vs modified.
    Mailbox defaults to the first SHARED_MAILBOXES entry if not specified.
    """
    if mailbox is None:
        mailbox = _default_feedback_mailbox()
    try:
        result = await feedback.check_feedback(mailbox=mailbox, hours=hours)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reasoning/gaps")
async def knowledge_gaps(days: int = 7):
    """Generate knowledge gap report from reasoning traces.

    Identifies topics where Hanna struggles (REJECTED, low confidence).
    """
    try:
        from app.reasoning.knowledge_gaps import generate_gap_report
        report = await generate_gap_report(days=days)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reasoning/refresh-authority")
async def refresh_authority():
    """Recompute dynamic authority weight adjustments from traces."""
    try:
        from app.reasoning.authority_learner import refresh_adjustments_cache
        adj = await refresh_adjustments_cache(days=30)
        return {"status": "ok", "categories": len(adj), "adjustments": adj}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Email: Historical Ingestion ─────────────────────────────────────────────

@app.post("/emails/history/ingest")
async def ingest_email_history(
    mailbox: str,
    since: str | None = None,
    max_items: int = 200,
    dry_run: bool = True,
):
    """Fetch historical sent items + questions and ingest as Q&A pairs.

    Set dry_run=false to actually ingest into ChromaDB.
    Default: dry_run=true (preview only).
    """
    try:
        result = await history.ingest_historical_emails(
            mailbox=mailbox,
            since=since,
            max_items=max_items,
            dry_run=dry_run,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Email: Attachments & Image Analysis ─────────────────────────────────────

@app.get("/emails/{mailbox}/messages/{message_id}/attachments")
async def list_email_attachments(mailbox: str, message_id: str):
    """List attachments for an email."""
    try:
        atts = await attachments.list_attachments(mailbox, message_id)
        return {
            "attachments": [
                AttachmentInfo(
                    id=a.id,
                    name=a.name,
                    content_type=a.content_type,
                    size=a.size,
                    is_image=a.is_image,
                )
                for a in atts
            ],
            "count": len(atts),
            "images": sum(1 for a in atts if a.is_image),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/emails/{mailbox}/messages/{message_id}/analyze-images",
    response_model=AttachmentAnalysisResponse,
)
async def analyze_email_images(
    mailbox: str,
    message_id: str,
    max_images: int = 5,
):
    """Analyze image attachments in an email using GPT-4o Vision.
    
    Fetches image attachments and returns text descriptions/OCR.
    Useful for understanding screenshots, documents, forms in customer emails.
    
    Args:
        mailbox: Email address of the shared mailbox
        message_id: Graph API message ID
        max_images: Maximum images to analyze (default 5, for cost control)
    """
    try:
        # Get all attachments first for count
        all_atts = await attachments.list_attachments(mailbox, message_id)
        
        # Analyze images
        results = await attachments.analyze_email_attachments(
            mailbox=mailbox,
            message_id=message_id,
            images_only=True,
            max_images=max_images,
        )
        
        return AttachmentAnalysisResponse(
            mailbox=mailbox,
            message_id=message_id,
            total_attachments=len(all_atts),
            images_analyzed=len(results),
            results=[
                ImageAnalysisResult(
                    attachment_id=r.attachment_id,
                    filename=r.filename,
                    description=r.description,
                    error=r.error,
                )
                for r in results
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Style Learning ───────────────────────────────────────────────────────────

@app.get("/style/analyze")
async def analyze_style(
    mailbox: str = "lakossagitarolo@neuzrt.hu",
    hours: int = 168,
    limit: int = 200,
):
    """Analyze colleague sent email style patterns.
    
    Returns word count stats, tone markers, greetings, closings,
    and per-category examples.
    """
    try:
        return await style_learner.analyze_sent_items(
            mailbox=mailbox, hours=hours, limit=limit,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/style/templates")
async def get_style_templates(
    mailbox: str = "lakossagitarolo@neuzrt.hu",
    hours: int = 720,
    min_examples: int = 3,
):
    """Build response templates from colleague sent items per topic category."""
    try:
        return await style_learner.get_category_templates(
            mailbox=mailbox, hours=hours, min_examples=min_examples,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/style/patterns")
async def get_saved_patterns():
    """Get last saved style patterns (from disk cache)."""
    patterns = style_learner.load_patterns()
    if not patterns:
        return {"status": "no_patterns", "message": "Run /style/analyze first"}
    return patterns


# ─── Templates ────────────────────────────────────────────────────────────────

@app.get("/templates")
async def list_templates():
    """List all email templates."""
    return {"templates": templates.list_templates(), "count": len(templates.TEMPLATES)}


class TemplateMatchRequest(BaseModel):
    email_text: str


@app.post("/templates/match")
async def match_template(req: TemplateMatchRequest):
    """Match email text against templates."""
    key, score = templates.match_template(req.email_text)
    if key is None:
        return {"matched": False, "template_key": None, "score": 0.0}
    tmpl = templates.TEMPLATES[key]
    return {
        "matched": True,
        "template_key": key,
        "template_name": tmpl["name"],
        "score": score,
        "confidence": tmpl["confidence"],
        "response_template": tmpl["response_template"],
    }


# ─── Attachment Extraction ────────────────────────────────────────────────────

@app.post("/emails/{mailbox}/messages/{message_id}/extract-attachments")
async def extract_attachments(mailbox: str, message_id: str):
    """Extract text from all attachments (PDF + images)."""
    try:
        results = await attachments.extract_all_attachments(mailbox, message_id)
        return {"attachments": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── BM25 Rebuild ────────────────────────────────────────────────────────────

@app.post("/bm25/rebuild")
async def bm25_rebuild():
    """Refresh BM25 index statistics in PostgreSQL.
    
    The content_tsvector column is auto-generated, so no manual update needed.
    This endpoint reindexes the GIN index and returns stats.
    """
    try:
        import asyncpg
        pool = await asyncpg.create_pool(
            "postgresql://klara:klara_docs_2026@host.docker.internal:5433/hanna_oetp",
            min_size=1, max_size=3,
        )
        async with pool.acquire() as conn:
            # Count total and indexed chunks
            total = await conn.fetchval("SELECT COUNT(*) FROM chunks")
            has_tsv = await conn.fetchval(
                "SELECT COUNT(*) FROM chunks WHERE content_tsvector IS NOT NULL"
            )
            
            # REINDEX the GIN index for performance
            try:
                await conn.execute("REINDEX INDEX CONCURRENTLY idx_chunks_tsvector")
            except Exception:
                # CONCURRENTLY might not work in transaction; try without
                try:
                    await conn.execute("REINDEX INDEX idx_chunks_tsvector")
                except Exception as idx_e:
                    print(f"[bm25] REINDEX warning (non-fatal): {idx_e}")
            
            # ANALYZE for fresh statistics
            await conn.execute("ANALYZE chunks")
        
        await pool.close()
        return {
            "status": "ok",
            "documents": total,
            "indexed": has_tsv,
            "storage": "postgresql_generated_tsvector",
            "note": "content_tsvector is auto-generated; ANALYZE refreshed stats",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Analytics ────────────────────────────────────────────────────────────────

@app.get("/analytics/weekly")
async def analytics_weekly(weeks: int = 1):
    """Weekly thematic analysis of emails."""
    return analytics.analyze_weekly(weeks)


@app.post("/analytics/weekly/report")
async def analytics_weekly_report(weeks: int = 1):
    """Generate and save weekly report to Obsidian."""
    return analytics.generate_weekly_report(weeks)


# ─── Obsidian: Ingestion & Search ────────────────────────────────────────────

async def _run_obsidian_ingest_job(vault_path: str, force: bool):
    """Background ingest runner to avoid blocking API workers."""
    async with _obsidian_ingest_lock:
        _obsidian_ingest_status["running"] = True
        _obsidian_ingest_status["started_at"] = datetime.now().isoformat()
        _obsidian_ingest_status["finished_at"] = None
        _obsidian_ingest_status["last_result"] = None
        _obsidian_ingest_status["last_error"] = None
        try:
            result = await obsidian_ingest.ingest_vault(
                vault_path=vault_path,
                force=force,
                collection_name="obsidian_notes",
            )
            _obsidian_ingest_status["last_result"] = result
        except Exception as e:
            _obsidian_ingest_status["last_error"] = str(e)
        finally:
            _obsidian_ingest_status["running"] = False
            _obsidian_ingest_status["finished_at"] = datetime.now().isoformat()


@app.post("/obsidian/ingest")
async def ingest_obsidian(
    background_tasks: BackgroundTasks,
    vault_path: str = "/app/obsidian-vault",
    force: bool = False,
    wait: bool = False,
):
    """Ingest Obsidian vault. Default non-blocking; set wait=true for synchronous call."""
    # Keep backward compatibility for callers that really want blocking behavior
    if wait:
        try:
            result = await obsidian_ingest.ingest_vault(
                vault_path=vault_path,
                force=force,
                collection_name="obsidian_notes",
            )
            return {"status": "ok", "mode": "sync", **result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    if _obsidian_ingest_status.get("running"):
        return {
            "status": "accepted",
            "mode": "async",
            "message": "Ingest already running",
            **_obsidian_ingest_status,
        }

    background_tasks.add_task(_run_obsidian_ingest_job, vault_path, force)
    return {
        "status": "accepted",
        "mode": "async",
        "message": "Ingest started in background",
        "running": True,
        "started_at": datetime.now().isoformat(),
    }


@app.get("/obsidian/ingest/status")
async def obsidian_ingest_status():
    """Get current/last Obsidian ingest job status."""
    return _obsidian_ingest_status


@app.get("/obsidian/search")
async def search_obsidian(
    q: str,
    limit: int = 10,
    folder: str | None = None,
    caller: str | None = None,
):
    """Search Obsidian notes using semantic similarity.
    
    Args:
        q: Search query
        limit: Maximum number of results (default 10)
        folder: Optional folder filter (inbox, projects, areas, resources, archive, tags, tasknotes)
        caller: Optional identifier for who made the search (bob, max, eve, etc.)
    """
    try:
        results = await obsidian_search.search_obsidian_notes(
            query=q,
            limit=limit,
            folder_filter=folder,
            collection_name="obsidian_notes",
            caller=caller
        )
        return {
            "results": results,
            "query": q,
            "total_found": len(results),
            "folder_filter": folder
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/obsidian/search/hybrid")
async def search_obsidian_hybrid(
    q: str,
    limit: int = 10,
    folder: str | None = None,
    caller: str | None = None,
    rerank: bool = True,
    instruction: str = "",
    compact: bool = False,
    graph: bool = True,
    min_score: float = Query(0.0, ge=0.0, le=2.0, description="Minimum final score (recommended: 0.20 for grounded answers)"),
):
    """Hybrid search: semantic + BM25 + Knowledge Graph → RRF fusion → reranking.
    
    Args:
        q: Search query
        limit: Maximum number of results (default 10)
        folder: Optional folder filter
        caller: Who made the search (bob, max, eve, etc.)
        rerank: Whether to apply reranking (default True)
        instruction: Custom instruction for reranker
        compact: If true, truncate content to 200 chars (saves context tokens)
        graph: Whether to use Knowledge Graph expansion (default True)
    """
    try:
        search_result = await obsidian_search.search_obsidian_hybrid(
            query=q,
            limit=limit,
            folder_filter=folder,
            collection_name="obsidian_notes",
            caller=caller,
            use_reranker=rerank,
            instruction=instruction,
            use_graph=graph,
        )
        results = search_result["results"]
        graph_context = search_result.get("graph_context")

        # Score threshold filtering
        if min_score > 0 and results:
            results = [
                r for r in results
                if float(r.get("rerank_score") or r.get("score") or r.get("rrf_score") or 0) >= min_score
            ]

        if compact:
            for r in results:
                if "content" in r and len(r["content"]) > 200:
                    r["content"] = r["content"][:200] + "…"

        graph_boosted = sum(1 for r in results if r.get("graph_boosted"))
        method = "hybrid"
        if graph:
            method += "+graph"
        if rerank:
            method += "+rerank"

        # Relevance assessment
        top_score = 0.0
        if results:
            top_score = float(results[0].get("rerank_score") or results[0].get("score") or results[0].get("rrf_score") or 0)
        relevance_threshold = min_score if min_score > 0 else 0.20
        relevance_sufficient = top_score >= relevance_threshold

        response = {
            "results": results,
            "query": q,
            "total_found": len(results),
            "folder_filter": folder,
            "method": method,
            "compact": compact,
            "graph_boosted_count": graph_boosted,
            "top_score": round(top_score, 4),
            "relevance_sufficient": relevance_sufficient,
        }
        if not relevance_sufficient:
            response["abstain_message"] = "A keresett téma nem található megfelelő relevanciával az Obsidian vault-ban."
        if graph_context:
            response["graph_context"] = graph_context

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Obsidian Structured Answer Generation (#7) ─────────────────────────────

OBSIDIAN_ANSWER_SYSTEM_PROMPT = """Te egy személyes tudásbázis (Obsidian vault) asszisztens vagy. A felhasználó kérdéseire KIZÁRÓLAG a megadott jegyzet-chunkök alapján válaszolsz.

SZABÁLYOK:
1. CSAK a [FORRÁS] blokkokban szereplő információk alapján válaszolj
2. Hivatkozz a forrás fájlnevére és mappájára
3. Ha a források NEM tartalmaznak választ → confidence: "insufficient"
4. SOHA ne egészítsd ki saját tudásból
5. Ha részben van válasz → válaszolj ami van, jelezd a hiányt

VÁLASZ FORMÁTUM (szigorúan JSON):
{
  "answer": "A válasz szövege",
  "sources": [
    {"document": "fájlnév", "section": "mappa/szekció", "quote": "szó szerinti idézet"}
  ],
  "confidence": "high|medium|low|insufficient",
  "unanswered_parts": null
}"""

OBSIDIAN_ANSWER_FEWSHOT = [
    {
        "role": "user",
        "content": """Kérdés: Mi a Radix Next üzleti modellje?

[FORRÁS 1] §Radix Next.md — 1_projects (relevancia: 91%)
Szöveg: A Radix Next a NEÜ IT fejlesztési partnere. Fő tevékenység: OETP platform fejlesztés és üzemeltetés. Üzleti modell: megbízási szerződés alapú, havi díjas support + projekt alapú fejlesztés.

[FORRÁS 2] Radix Next Kft.md — 3_resources/Companies (relevancia: 76%)
Szöveg: Radix Next Kft. — IT fejlesztő cég, Budapest. Kapcsolat: info@radixnext.hu.""",
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "answer": "A Radix Next a NEÜ IT fejlesztési partnere, megbízási szerződés alapú üzleti modellel: havi díjas support és projekt alapú fejlesztés. Fő tevékenysége az OETP platform fejlesztése és üzemeltetése.",
            "sources": [
                {"document": "§Radix Next.md", "section": "1_projects", "quote": "Üzleti modell: megbízási szerződés alapú, havi díjas support + projekt alapú fejlesztés."}
            ],
            "confidence": "high",
            "unanswered_parts": None,
        }, ensure_ascii=False),
    },
]


class ObsidianAnswerRequest(BaseModel):
    query: str
    limit: int = 10
    folder: str | None = None
    min_score: float = 0.20
    model: str | None = None
    max_context_chunks: int = 3


@app.post("/obsidian/answer")
async def obsidian_answer_endpoint(req: ObsidianAnswerRequest):
    """Obsidian search + LLM structured answer with grounding."""
    # 1. Search
    search_result = await obsidian_search.search_obsidian_hybrid(
        query=req.query, limit=req.limit, folder_filter=req.folder,
        collection_name="obsidian_notes", use_reranker=True, use_graph=True,
    )
    results = search_result.get("results", [])

    # 2. Score threshold
    if req.min_score > 0 and results:
        results = [r for r in results if float(r.get("rerank_score") or r.get("score") or r.get("rrf_score") or 0) >= req.min_score]

    top_score = float(results[0].get("rerank_score") or results[0].get("score") or 0) if results else 0.0

    if not results or top_score < req.min_score:
        return {
            "answer": "A keresett téma nem található megfelelő relevanciával az Obsidian vault-ban.",
            "sources": [], "confidence": "insufficient", "unanswered_parts": req.query,
            "model_used": None, "top_score": round(top_score, 4), "relevance_sufficient": False, "chunks_used": 0,
        }

    # 3. (#11) Format context — tight, structured, top N chunks
    gen_results = results[:req.max_context_chunks]
    context_parts = []
    for i, r in enumerate(gen_results, 1):
        fname = r.get("file_name", r.get("source", "?"))
        folder = r.get("folder", "")
        score = float(r.get("rerank_score") or r.get("score") or 0)
        text = r.get("content", r.get("text", ""))
        context_parts.append(f"[FORRÁS {i}] {fname} — {folder} (relevancia: {int(score * 100)}%)\nSzöveg: {text}\n---")
    context_block = "\n\n".join(context_parts)
    user_msg = f"Kérdés: {req.query}\n\n{context_block}"

    # 4. (#12) Cascade routing + LLM call
    from openai import OpenAI
    query_type = _classify_query_complexity(req.query)
    if req.model:
        model = req.model
    else:
        model = _select_model(query_type, top_score, settings.answer_model)
    client = OpenAI(api_key=settings.openai_api_key)
    try:
        messages = [
            {"role": "system", "content": OBSIDIAN_ANSWER_SYSTEM_PROMPT},
            *OBSIDIAN_ANSWER_FEWSHOT,
            {"role": "user", "content": user_msg},
        ]
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=settings.answer_temperature,
            max_tokens=settings.answer_max_tokens, response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        answer_data = json.loads(raw)
    except Exception as e:
        return {"answer": None, "error": f"LLM generation failed: {e}", "sources": [], "confidence": "error",
                "model_used": model, "top_score": round(top_score, 4), "relevance_sufficient": True, "chunks_used": len(gen_results)}

    # 5. Quote verification
    chunk_texts = " ".join(r.get("content", r.get("text", "")) for r in gen_results)
    for src in answer_data.get("sources", []):
        quote = src.get("quote", "")
        src["verified"] = bool(quote and quote in chunk_texts)

    # 6. NLI faithfulness verification (best-effort, non-blocking)
    nli_result = None
    try:
        nli_url = os.getenv("NLI_SERVICE_URL", "http://host.docker.internal:8107")
        async with httpx.AsyncClient(timeout=10) as nli_client:
            nli_resp = await nli_client.post(f"{nli_url}/verify-answer", json={
                "answer": answer_data.get("answer", ""),
                "context": chunk_texts,
            })
            if nli_resp.status_code == 200:
                nli_result = nli_resp.json()
                if nli_result.get("overall_verdict") == "unfaithful":
                    answer_data["confidence"] = "low"
                    answer_data["nli_warning"] = "NLI verifikáció ellentmondást talált egyes állításokban."
    except Exception:
        pass

    return {**answer_data, "model_used": model, "query_type": query_type, "top_score": round(top_score, 4),
            "relevance_sufficient": True, "chunks_used": len(gen_results), "nli_verification": nli_result}


@app.get("/obsidian/stats")
async def obsidian_stats():
    """Obsidian notes collection statistics."""
    try:
        return await obsidian_search.get_obsidian_stats("obsidian_notes")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/obsidian/search-stats")
async def obsidian_search_stats(hours: int = 24):
    """Get search statistics for Obsidian notes.
    
    Args:
        hours: Look back this many hours (default 24)
    """
    try:
        return obsidian_search.get_search_stats(hours)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/obsidian/last-sync")
async def obsidian_last_sync():
    """Get information about the last Obsidian vault sync."""
    try:
        return await obsidian_ingest.get_last_sync_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Knowledge Graph endpoints ────────────────────────────────────────

@app.post("/obsidian/graph/extract")
async def kg_extract_vault(
    vault_path: str = "/app/obsidian-vault",
    use_llm: bool = False,
    limit: int | None = None,
):
    """Run KG entity/relation extraction across the vault."""
    try:
        stats = await kg_extract.extract_vault_kg(vault_path, use_llm=use_llm, limit=limit)
        return {"status": "ok", **stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/obsidian/graph/entity")
async def kg_search_entity(q: str, type: str | None = None, limit: int = 20):
    """Search entities by name (trigram + ILIKE + alias matching)."""
    pool = await kg_extract._get_pool()
    type_clause = "AND e.type = $2" if type else ""
    params = [q]
    if type:
        params.append(type)
    params.append(limit)
    limit_param = f"${len(params)}"

    # Strategy 1: Trigram similarity
    rows = await pool.fetch(
        f"""SELECT e.id, e.name, e.type, e.aliases, e.metadata, e.source_file,
                   similarity(e.name, $1) AS sim,
                   (SELECT COUNT(*) FROM kg_relations r WHERE r.source_id = e.id OR r.target_id = e.id) AS relation_count,
                   (SELECT COUNT(*) FROM kg_entity_chunks ec WHERE ec.entity_id = e.id) AS chunk_count
            FROM kg_entities e
            WHERE similarity(e.name, $1) > 0.15 {type_clause}
            ORDER BY similarity(e.name, $1) DESC
            LIMIT {limit_param}""",
        *params,
    )
    
    seen = {}
    for r in rows:
        seen[r["id"]] = r

    # Strategy 2: ILIKE fallback (short names like Kbt., NEÜ, OETP)
    if len(q) >= 3:
        ilike_params = [f"%{q}%"]
        ilike_type_clause = "AND e.type = $2" if type else ""
        if type:
            ilike_params.append(type)
        ilike_params.append(limit)
        ilike_limit = f"${len(ilike_params)}"

        ilike_rows = await pool.fetch(
            f"""SELECT e.id, e.name, e.type, e.aliases, e.metadata, e.source_file,
                       0.6::float AS sim,
                       (SELECT COUNT(*) FROM kg_relations r WHERE r.source_id = e.id OR r.target_id = e.id) AS relation_count,
                       (SELECT COUNT(*) FROM kg_entity_chunks ec WHERE ec.entity_id = e.id) AS chunk_count
                FROM kg_entities e
                WHERE e.name ILIKE $1 {ilike_type_clause}
                LIMIT {ilike_limit}""",
            *ilike_params,
        )
        for r in ilike_rows:
            if r["id"] not in seen:
                seen[r["id"]] = r

    # Strategy 3: Alias matching
    if len(q) >= 3:
        alias_params = [f"%{q}%"]
        alias_type_clause = "AND e.type = $2" if type else ""
        if type:
            alias_params.append(type)
        alias_params.append(limit)
        alias_limit = f"${len(alias_params)}"

        alias_rows = await pool.fetch(
            f"""SELECT e.id, e.name, e.type, e.aliases, e.metadata, e.source_file,
                       0.85::float AS sim,
                       (SELECT COUNT(*) FROM kg_relations r WHERE r.source_id = e.id OR r.target_id = e.id) AS relation_count,
                       (SELECT COUNT(*) FROM kg_entity_chunks ec WHERE ec.entity_id = e.id) AS chunk_count
                FROM kg_entities e
                WHERE EXISTS (SELECT 1 FROM unnest(e.aliases) a WHERE a ILIKE $1)
                {alias_type_clause}
                LIMIT {alias_limit}""",
            *alias_params,
        )
        for r in alias_rows:
            if r["id"] not in seen:
                seen[r["id"]] = r

    # Sort by similarity, limit
    results = sorted(seen.values(), key=lambda r: float(r["sim"]), reverse=True)[:limit]

    return {
        "query": q,
        "results": [
            {
                "id": r["id"],
                "name": r["name"],
                "type": r["type"],
                "aliases": r["aliases"],
                "metadata": json.loads(r["metadata"]) if r["metadata"] else {},
                "source_file": r["source_file"],
                "similarity": round(float(r["sim"]), 3),
                "relation_count": r["relation_count"],
                "chunk_count": r["chunk_count"],
            }
            for r in results
        ],
    }


@app.get("/obsidian/graph/relations")
async def kg_get_relations(
    entity_id: int | None = None,
    entity_name: str | None = None,
    relation_type: str | None = None,
    limit: int = 50,
):
    """Get relations for an entity (by ID or name)."""
    pool = await kg_extract._get_pool()

    if entity_id:
        rows = await pool.fetch(
            """SELECT r.id, r.relation_type, r.confidence, r.source_file,
                      s.name AS source_name, s.type AS source_type,
                      t.name AS target_name, t.type AS target_type
               FROM kg_relations r
               JOIN kg_entities s ON s.id = r.source_id
               JOIN kg_entities t ON t.id = r.target_id
               WHERE r.source_id = $1 OR r.target_id = $1
               ORDER BY r.confidence DESC
               LIMIT $2""",
            entity_id, limit,
        )
    elif entity_name:
        rows = await pool.fetch(
            """SELECT r.id, r.relation_type, r.confidence, r.source_file,
                      s.name AS source_name, s.type AS source_type,
                      t.name AS target_name, t.type AS target_type
               FROM kg_relations r
               JOIN kg_entities s ON s.id = r.source_id
               JOIN kg_entities t ON t.id = r.target_id
               WHERE s.name ILIKE $1 OR t.name ILIKE $1
               ORDER BY r.confidence DESC
               LIMIT $2""",
            f"%{entity_name}%", limit,
        )
    else:
        raise HTTPException(status_code=400, detail="Provide entity_id or entity_name")

    type_filter = relation_type
    results = []
    for r in rows:
        if type_filter and r["relation_type"] != type_filter:
            continue
        results.append({
            "id": r["id"],
            "source": {"name": r["source_name"], "type": r["source_type"]},
            "target": {"name": r["target_name"], "type": r["target_type"]},
            "relation_type": r["relation_type"],
            "confidence": round(float(r["confidence"]), 2),
            "source_file": r["source_file"],
        })

    return {"results": results, "total": len(results)}


@app.get("/obsidian/graph/stats")
async def kg_stats():
    """Get Knowledge Graph statistics."""
    pool = await kg_extract._get_pool()
    entities = await pool.fetchval("SELECT COUNT(*) FROM kg_entities")
    relations = await pool.fetchval("SELECT COUNT(*) FROM kg_relations")
    entity_chunks = await pool.fetchval("SELECT COUNT(*) FROM kg_entity_chunks")

    type_counts = await pool.fetch(
        "SELECT type, COUNT(*) AS cnt FROM kg_entities GROUP BY type ORDER BY cnt DESC"
    )
    relation_counts = await pool.fetch(
        "SELECT relation_type, COUNT(*) AS cnt FROM kg_relations GROUP BY relation_type ORDER BY cnt DESC"
    )

    return {
        "entities": entities,
        "relations": relations,
        "entity_chunk_links": entity_chunks,
        "entity_types": {r["type"]: r["cnt"] for r in type_counts},
        "relation_types": {r["relation_type"]: r["cnt"] for r in relation_counts},
    }


@app.get("/obsidian/graph/search")
async def kg_graph_search(
    q: str,
    entity_threshold: float = 0.35,
    max_entities: int = 5,
    max_related: int = 15,
    max_chunks: int = 8,
):
    """Graph-only search: detect entities → expand graph → return related chunks.

    Useful for relationship queries like "ki dolgozik az ENAIRGY projekten?"
    """
    try:
        result = await obsidian_kg_search.graph_augmented_search(
            query=q,
            entity_threshold=entity_threshold,
            max_entities=max_entities,
            max_related=max_related,
            max_graph_chunks=max_chunks,
        )
        return {
            "query": q,
            **result,
            "total_chunks": len(result["graph_chunks"]),
            "total_entities": len(result["matched_entities"]) + len(result["related_entities"]),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Graph: Neighbors (depth-aware) ────────────────────────────────────────
@app.get("/obsidian/graph/neighbors/{entity_id}")
async def kg_neighbors(entity_id: int, depth: int = 1, type: str | None = None, limit: int = 50):
    """Get neighboring entities via relations, with configurable depth (max 2)."""
    if depth < 1 or depth > 2:
        raise HTTPException(status_code=400, detail="depth must be 1 or 2")
    pool = await kg_extract._get_pool()

    # Verify entity exists
    entity = await pool.fetchrow("SELECT id, name, type, aliases FROM kg_entities WHERE id = $1", entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")

    type_clause = "AND e.type = $3" if type else ""
    params_base = [entity_id, limit] + ([type] if type else [])

    # Depth 1: direct neighbors
    rows = await pool.fetch(
        f"""WITH neighbors AS (
            SELECT DISTINCT ON (e.id)
                e.id, e.name, e.type, e.aliases,
                r.relation_type,
                CASE WHEN r.source_id = $1 THEN 'outgoing' ELSE 'incoming' END AS direction,
                1 AS depth
            FROM kg_relations r
            JOIN kg_entities e ON e.id = CASE WHEN r.source_id = $1 THEN r.target_id ELSE r.source_id END
            WHERE (r.source_id = $1 OR r.target_id = $1)
            {type_clause}
            ORDER BY e.id
            LIMIT $2
        )
        SELECT * FROM neighbors""",
        *params_base,
    )
    results = [dict(r) for r in rows]

    if depth == 2 and results:
        depth1_ids = [r["id"] for r in results]
        # Depth 2: neighbors of neighbors (exclude original entity and depth-1)
        exclude_ids = [entity_id] + depth1_ids
        d2_rows = await pool.fetch(
            f"""SELECT DISTINCT ON (e.id)
                e.id, e.name, e.type, e.aliases,
                r.relation_type,
                CASE WHEN r.source_id = ANY($1) THEN 'outgoing' ELSE 'incoming' END AS direction,
                2 AS depth
            FROM kg_relations r
            JOIN kg_entities e ON e.id = CASE WHEN r.source_id = ANY($1) THEN r.target_id ELSE r.source_id END
            WHERE (r.source_id = ANY($1) OR r.target_id = ANY($1))
            AND e.id != ALL($2)
            {type_clause.replace('$3', '$4') if type else ''}
            ORDER BY e.id
            LIMIT $3""",
            depth1_ids, exclude_ids, limit, *([type] if type else []),
        )
        results.extend(dict(r) for r in d2_rows)

    return {
        "entity": {"id": entity["id"], "name": entity["name"], "type": entity["type"]},
        "neighbors": [
            {
                "id": r["id"],
                "name": r["name"],
                "type": r["type"],
                "aliases": r["aliases"],
                "relation_type": r["relation_type"],
                "direction": r["direction"],
                "depth": r["depth"],
            }
            for r in results
        ],
        "total": len(results),
    }


# ── Graph: Shortest Path ─────────────────────────────────────────────────
@app.get("/obsidian/graph/path")
async def kg_shortest_path(from_id: int, to_id: int, max_depth: int = 4):
    """Find shortest path between two entities via BFS (max depth 4)."""
    if max_depth < 1 or max_depth > 4:
        raise HTTPException(status_code=400, detail="max_depth must be 1-4")
    pool = await kg_extract._get_pool()

    # Verify both entities exist
    from_ent = await pool.fetchrow("SELECT id, name, type FROM kg_entities WHERE id = $1", from_id)
    to_ent = await pool.fetchrow("SELECT id, name, type FROM kg_entities WHERE id = $1", to_id)
    if not from_ent or not to_ent:
        raise HTTPException(status_code=404, detail="Entity not found")

    if from_id == to_id:
        return {"path": [{"id": from_ent["id"], "name": from_ent["name"], "type": from_ent["type"]}], "edges": [], "depth": 0}

    # BFS
    from collections import deque
    visited = {from_id: None}  # entity_id → (prev_entity_id, relation_type, direction)
    queue = deque([from_id])
    found = False

    for current_depth in range(max_depth):
        if found:
            break
        next_queue = deque()
        while queue:
            node = queue.popleft()
            rels = await pool.fetch(
                """SELECT source_id, target_id, relation_type
                   FROM kg_relations WHERE source_id = $1 OR target_id = $1""",
                node,
            )
            for r in rels:
                neighbor = r["target_id"] if r["source_id"] == node else r["source_id"]
                if neighbor not in visited:
                    direction = "outgoing" if r["source_id"] == node else "incoming"
                    visited[neighbor] = (node, r["relation_type"], direction)
                    if neighbor == to_id:
                        found = True
                        break
                    next_queue.append(neighbor)
            if found:
                break
        queue = next_queue

    if not found:
        return {"path": [], "edges": [], "depth": -1, "message": "No path found"}

    # Reconstruct path
    path_ids = []
    edges = []
    current = to_id
    while current is not None:
        path_ids.append(current)
        info = visited[current]
        if info is not None:
            prev, rel_type, direction = info
            edges.append({"relation_type": rel_type, "direction": direction})
            current = prev
        else:
            current = None
    path_ids.reverse()
    edges.reverse()

    # Fetch entity details
    entities = await pool.fetch(
        "SELECT id, name, type FROM kg_entities WHERE id = ANY($1)",
        path_ids,
    )
    ent_map = {e["id"]: {"id": e["id"], "name": e["name"], "type": e["type"]} for e in entities}

    return {
        "path": [ent_map.get(eid, {"id": eid}) for eid in path_ids],
        "edges": edges,
        "depth": len(edges),
    }


# ── Graph: Entity Chunks ─────────────────────────────────────────────────
@app.get("/obsidian/graph/entity/{entity_id}/chunks")
async def kg_entity_chunks(entity_id: int, limit: int = 20):
    """Get chunks linked to an entity via kg_entity_chunks."""
    pool = await kg_extract._get_pool()

    entity = await pool.fetchrow("SELECT id, name, type FROM kg_entities WHERE id = $1", entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")

    rows = await pool.fetch(
        """SELECT c.chunk_id, c.file_path, c.file_name,
                  c.folder, ec.mention_type, c.context_prefix
           FROM kg_entity_chunks ec
           JOIN obsidian_chunks c ON c.chunk_id = ec.chunk_id
           WHERE ec.entity_id = $1
           ORDER BY ec.mention_type, c.file_path
           LIMIT $2""",
        entity_id, limit,
    )

    # Fetch content snippets separately with error handling
    chunk_ids = [r["chunk_id"] for r in rows]
    snippets = {}
    if chunk_ids:
        try:
            snippet_rows = await pool.fetch(
                """SELECT chunk_id, encode(substring(content::bytea from 1 for 300), 'escape') AS snippet
                   FROM obsidian_chunks WHERE chunk_id = ANY($1::text[])""",
                chunk_ids,
            )
            snippets = {r["chunk_id"]: r["snippet"] for r in snippet_rows}
        except Exception:
            # Fallback: no snippets
            pass

    return {
        "entity": {"id": entity["id"], "name": entity["name"], "type": entity["type"]},
        "chunks": [
            {
                "chunk_id": r["chunk_id"],
                "snippet": snippets.get(r["chunk_id"], ""),
                "file_path": r["file_path"],
                "file_name": r["file_name"],
                "folder": r["folder"],
                "mention_type": r["mention_type"],
                "context_prefix": r["context_prefix"],
            }
            for r in rows
        ],
        "total": len(rows),
    }


# ── Cross-RAG Entity Enrichment ──────────────────────────────────────────
@app.post("/obsidian/enrich-from-klara")
async def enrich_from_klara(
    since_hours: int = 0,
    min_relations: int = 2,
    dry_run: bool = False,
):
    """Cross-RAG enrichment: sync Klára NEÜ docs KG entities → Obsidian People/Companies.

    Args:
        since_hours: Look at entities from last N hours (0=all)
        min_relations: Minimum relations to create new files
        dry_run: Preview changes without writing
    """
    try:
        result = await cross_rag_enrich.enrich_from_klara(
            since_hours=since_hours,
            min_relations=min_relations,
            dry_run=dry_run,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Cross-RAG endpoints ─────────────────────────────────────────────


@app.get("/cross-rag/search")
async def cross_rag_search(q: str, entity_type: str = None, limit: int = 20):
    """Search entities across all RAG databases."""
    try:
        return await cross_rag_api.search_canonical(q, entity_type, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cross-rag/entity/{canonical_id}")
async def cross_rag_entity(canonical_id: int):
    """Get a canonical entity with all its source links and chunk counts."""
    result = await cross_rag_api.get_canonical_entity(canonical_id)
    if not result:
        raise HTTPException(status_code=404, detail="Canonical entity not found")
    return result


# ─── Structured Answer Generation (#7) ──────────────────────────────────────

HANNA_ANSWER_SYSTEM_PROMPT = """Te az OETP (Otthoni Energiatároló Program) ügyfélszolgálati asszisztens vagy. A pályázók kérdéseire KIZÁRÓLAG a megadott forrás-chunkök alapján válaszolsz.

SZABÁLYOK:
1. CSAK a [FORRÁS] blokkokban szereplő információk alapján válaszolj
2. Minden állításhoz kötelező szó szerinti idézet a forrásból (quote mező)
3. Ha a források NEM tartalmaznak választ → confidence: "insufficient"
4. SOHA ne egészítsd ki saját tudásból — inkább mondd hogy "nem tudom"
5. Összegeket, határidőket, feltételeket PONTOSAN idézd — ne kerekíts, ne általánosíts
6. Ha részben van válasz → válaszolj ami van, és jelezd a hiányt unanswered_parts-ban
7. Használj kedves, segítőkész ügyfélszolgálati hangnemet

VÁLASZ FORMÁTUM (szigorúan JSON):
{
  "answer": "A válasz szövege, közérthetően megfogalmazva",
  "sources": [
    {"document": "forrás neve", "section": "pont/kérdés szám", "quote": "szó szerinti idézet"}
  ],
  "confidence": "high|medium|low|insufficient",
  "unanswered_parts": null
}"""

HANNA_ANSWER_FEWSHOT = [
    {
        "role": "user",
        "content": """Kérdés: Mekkora a támogatás maximális összege?

[FORRÁS 1] OETP Felhívás — 3.1. pont (relevancia: 95%)
Szöveg: A támogatás összege legfeljebb 4.000.000 Ft lehet háztartásonként.

[FORRÁS 2] OETP GYIK — 12. kérdés (relevancia: 82%)
Szöveg: A pályázó háztartásonként legfeljebb 4 millió forint vissza nem térítendő támogatást igényelhet.""",
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "answer": "A támogatás maximális összege háztartásonként 4.000.000 Ft (vissza nem térítendő).",
            "sources": [
                {"document": "OETP Felhívás", "section": "3.1. pont", "quote": "A támogatás összege legfeljebb 4.000.000 Ft lehet háztartásonként."}
            ],
            "confidence": "high",
            "unanswered_parts": None,
        }, ensure_ascii=False),
    },
    {
        "role": "user",
        "content": """Kérdés: Lehet-e Dubajból pályázni?

[FORRÁS 1] OETP Felhívás — 2.1. pont (relevancia: 35%)
Szöveg: Pályázatot nyújthat be természetes személy, aki Magyarország területén állandó lakóhellyel rendelkezik.

[FORRÁS 2] OETP GYIK — 3. kérdés (relevancia: 30%)
Szöveg: A pályázó csak magyarországi lakóingatlana vonatkozásában nyújthat be pályázatot.""",
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "answer": "A program kizárólag Magyarország területén állandó lakóhellyel rendelkező természetes személyek számára érhető el, így külföldi (pl. dubaji) lakóhelyről sajnos nem lehet pályázni.",
            "sources": [
                {"document": "OETP Felhívás", "section": "2.1. pont", "quote": "Pályázatot nyújthat be természetes személy, aki Magyarország területén állandó lakóhellyel rendelkezik."}
            ],
            "confidence": "high",
            "unanswered_parts": None,
        }, ensure_ascii=False),
    },
]


def _classify_query_complexity(query: str) -> str:
    """#12 Cascade routing: classify query as 'simple' or 'complex'."""
    import re
    q = query.lower().strip()
    simple_patterns = [
        r'^(mennyi|mekkora|mikor|mikortól|meddig|hány|ki a|mi a|milyen)\b',
        r'^(kell-e|lehet-e|szabad-e|van-e|köteles-e|jogosult-e)\b',
        r'\b(mértéke|összege|határideje|határidő|feltétele|díja|bírság)\b',
        r'\b(hány nap|hány hónap|hány év|hány százalék)\b',
    ]
    for pattern in simple_patterns:
        if re.search(pattern, q):
            return "simple"
    complex_patterns = [
        r'\b(hasonlítsd|összehasonlít|különbség|eltérés)\b',
        r'\b(magyarázd|elemezd|fejts ki|részletezd)\b',
        r'\b(milyen esetben|milyen feltételekkel.*és.*és)\b',
        r'\b(hogyan változott|hogyan alakult)\b',
    ]
    for pattern in complex_patterns:
        if re.search(pattern, q):
            return "complex"
    return "simple" if len(q) < 50 else "complex"


def _select_model(query_type: str, top_score: float, default_model: str) -> str:
    """#12 Cascade routing: select LLM based on query type + retrieval confidence."""
    if query_type == "simple" and top_score >= 0.60:
        return "gpt-4o-mini"
    if top_score < 0.45 or query_type == "complex":
        return "gpt-4o"
    return default_model


class HannaAnswerRequest(BaseModel):
    query: str
    top_k: int = 5
    category: str | None = None
    min_score: float = 0.35
    model: str | None = None
    max_context_chunks: int = 3


@app.post("/answer")
async def answer_endpoint(req: HannaAnswerRequest):
    """Search + LLM structured answer with grounding and hallucination control."""
    # 1. Search
    results = await rag_search.search_async(
        query=req.query,
        top_k=req.top_k,
        category=req.category,
        only_valid=True,
    )

    # 2. Score threshold
    if req.min_score > 0 and results:
        results = [
            r for r in results
            if float(r.get("rerank_score") or r.get("score") or 0) >= req.min_score
        ]

    # 3. Relevance check
    top_score = float(results[0].get("rerank_score") or results[0].get("score") or 0) if results else 0.0

    if not results or top_score < req.min_score:
        return {
            "answer": "A rendelkezésre álló dokumentumok alapján erre a kérdésre nem található megbízható válasz. Kérem forduljon az ügyfélszolgálathoz.",
            "sources": [],
            "confidence": "insufficient",
            "unanswered_parts": req.query,
            "model_used": None,
            "top_score": round(top_score, 4),
            "relevance_sufficient": False,
            "chunks_used": 0,
        }

    # 4. Format context (#11 context preparation)
    gen_results = results[:req.max_context_chunks]
    context_parts = []
    for i, r in enumerate(gen_results, 1):
        source = r.get("source", "?")
        category = r.get("category", "")
        score = float(r.get("rerank_score") or r.get("score") or 0)
        text = r.get("text", "")
        context_parts.append(
            f"[FORRÁS {i}] {source} — {category} (relevancia: {int(score * 100)}%)\nSzöveg: {text}\n---"
        )
    context_block = "\n\n".join(context_parts)
    user_msg = f"Kérdés: {req.query}\n\n{context_block}"

    # 5. (#12) Cascade routing + (#10) VerbatimRAG for simple fact questions
    from openai import OpenAI
    query_type = _classify_query_complexity(req.query)

    # Try extractive answer first for simple queries with high confidence
    if query_type == "simple" and top_score >= 0.50:
        try:
            verbatim_url = os.getenv("VERBATIM_SERVICE_URL", "http://host.docker.internal:8108")
            async with httpx.AsyncClient(timeout=5) as vc:
                vr = await vc.post(f"{verbatim_url}/extract", json={
                    "question": req.query,
                    "chunks": [r.get("text", "") for r in gen_results],
                })
                if vr.status_code == 200:
                    vdata = vr.json()
                    if vdata.get("has_answer") and vdata.get("total_spans", 0) > 0:
                        all_spans = []
                        sources = []
                        for sr in vdata["results"]:
                            idx = sr["chunk_index"]
                            r = gen_results[idx]
                            source_name = r.get("source", "?")
                            category = r.get("category", "")
                            for span in sr["extracted_spans"]:
                                all_spans.append(span)
                                sources.append({"document": source_name, "section": category, "quote": span, "verified": True})
                        return {
                            "answer": " ".join(all_spans),
                            "sources": sources,
                            "confidence": "high",
                            "unanswered_parts": None,
                            "model_used": "verbatim-rag",
                            "query_type": query_type,
                            "top_score": round(top_score, 4),
                            "relevance_sufficient": True,
                            "chunks_used": len(gen_results),
                            "extraction_method": "verbatim",
                            "nli_verification": None,
                        }
        except Exception:
            pass  # VerbatimRAG unavailable — fall through to generative

    if req.model:
        model = req.model
    else:
        model = _select_model(query_type, top_score, settings.answer_model)
    client = OpenAI(api_key=settings.openai_api_key)
    try:
        messages = [
            {"role": "system", "content": HANNA_ANSWER_SYSTEM_PROMPT},
            *HANNA_ANSWER_FEWSHOT,
            {"role": "user", "content": user_msg},
        ]
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=settings.answer_temperature,
            max_tokens=settings.answer_max_tokens,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        answer_data = json.loads(raw)
    except Exception as e:
        return {
            "answer": None,
            "error": f"LLM generation failed: {e}",
            "sources": [],
            "confidence": "error",
            "model_used": model,
            "top_score": round(top_score, 4),
            "relevance_sufficient": True,
            "chunks_used": len(gen_results),
        }

    # 6. Quote verification
    chunk_texts = " ".join(r.get("text", "") for r in gen_results)
    for src in answer_data.get("sources", []):
        quote = src.get("quote", "")
        src["verified"] = bool(quote and quote in chunk_texts)

    # 7. NLI faithfulness verification (best-effort, non-blocking)
    nli_result = None
    try:
        nli_url = os.getenv("NLI_SERVICE_URL", "http://host.docker.internal:8107")
        async with httpx.AsyncClient(timeout=10) as nli_client:
            nli_resp = await nli_client.post(f"{nli_url}/verify-answer", json={
                "answer": answer_data.get("answer", ""),
                "context": chunk_texts,
            })
            if nli_resp.status_code == 200:
                nli_result = nli_resp.json()
                if nli_result.get("overall_verdict") == "unfaithful":
                    answer_data["confidence"] = "low"
                    answer_data["nli_warning"] = "NLI verifikáció ellentmondást talált egyes állításokban."
    except Exception:
        pass

    return {
        **answer_data,
        "model_used": model,
        "query_type": query_type,
        "top_score": round(top_score, 4),
        "relevance_sufficient": True,
        "chunks_used": len(gen_results),
        "nli_verification": nli_result,
    }


@app.get("/cross-rag/stats")
async def cross_rag_stats():
    """Get cross-rag database statistics."""
    return await cross_rag_api.get_stats()


@app.get("/cross-rag/multi-db")
async def cross_rag_multi_db(min_dbs: int = 2, entity_type: str = None, limit: int = 50):
    """Get entities present in multiple databases."""
    return await cross_rag_api.get_multi_db_entities(min_dbs, entity_type, limit)
