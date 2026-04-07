"""Hanna Backend — FastAPI REST API for RAG + Email."""

import asyncio
import json
import os
import re
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
# Obsidian RAG moved to standalone service (obsidian-rag :8115)
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

    # Start autonomous scheduler (if enabled)
    from .scheduler import start_scheduler, stop_scheduler
    start_scheduler()

    yield

    stop_scheduler()


app = FastAPI(
    title="Hanna Backend",
    description="RAG pipeline + Email integration for OETP customer service",
    version="0.1.0",
    lifespan=lifespan,
)



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

@app.get("/scheduler/status")
async def scheduler_status():
    """Return scheduler status: last run time, result, errors."""
    from .scheduler import get_scheduler_status
    return get_scheduler_status()


@app.get("/llm/health")
async def llm_health():
    """Test all LLM providers and return status.

    Returns provider availability + response time.
    ALERT: if no provider is available, returns 503.
    """
    from .llm_client import health_check
    results = await health_check()
    any_ok = any(r.get("status") == "ok" for r in results.values())
    if not any_ok:
        raise HTTPException(status_code=503, detail={"error": "ALL LLM providers down", "providers": results})
    return {"status": "ok" if any_ok else "critical", "providers": results}


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

DRAFT_GENERATE_SYSTEM = """Te a Nemzeti Energetikai Ügynökség ügyfélszolgálati munkatársa vagy. Pályázóknak válaszolsz az Otthoni Energiatároló Programmal (OETP) kapcsolatos kérdésekre.

Úgy írj, mint egy tapasztalt, kedves kolléga — NEM mint egy AI vagy chatbot.

SZABÁLYOK:

1. CSAK a megadott tények alapján válaszolj.
2. SOHA ne találj ki dátumot, összeget, határidőt, feltételt ami nincs a tényekben.
3. Ha magabiztosan megválaszolható → confidence: "high".
4. Ha a tényekből RÉSZBEN megválaszolható → válaszolj ami van, confidence: "medium".
5. Ha a tények EGYÁLTALÁN nem relevánsak a kérdéshez → confidence: "skip".
6. NE írj kitérő válaszokat ("kollégánk válaszol", "nincs elég info", "nem áll rendelkezésre").
7. Ha a tényekből LOGIKUSAN következtethető a válasz → válaszolj. Pl. ha a szabály "max 3 kivitelező" és az ügyfél kérdezi "kereshetek-e 4.-et?" → a válasz NEM skip, hanem "a rendszerben legfeljebb 3 kivitelezőnek küldhet űrlapot" + válasz a konkrét kérdésre.

KRITIKUS — NE MONDD VISSZA A KÉRDÉST:
- Ha az ügyfél mond valamit (pl. "a 10.2.4 pont törlésre került") és azt kérdezi "mi a teendő" →
  NE írd vissza hogy "igen, törlésre került". Ez NEM válasz. Ha nem tudod MI a teendő → skip.
- A válasznak a KÉRDÉSRE kell felelnie, nem a kérdésben szereplő tényeket visszaigazolnia.
- Ha a tények megerősítik amit az ügyfél már tud, de nem válaszolják meg amit kérdez → skip.

STÍLUS — ÍGY ÍRNAK A KOLLÉGÁID:
- Egyszerű, közvetlen, barátságos
- NAGYON RÖVID — a kollégák átlagosan 2-3 mondatot írnak (100-200 karakter). TE IS ÍGY ÍRJ.
- Maximum 3-4 mondat, kivéve ha a kérdés komplex (3+ részkérdés).
- "Tájékoztatjuk, hogy..." vagy rögtön a lényeg
- NE magyarázd el a hátteret ha nem kérdezték — csak válaszolj a kérdésre
- TILTOTT FORDULATOK (ezeket SOHA ne használd):
  * "a rendelkezésre álló tények alapján"
  * "a megadott tények alapján"
  * "ellenőrzött tények"
  * "forrás-chunkök"
  * "nem szerepel információ" / "nem tartalmaz információt"
  * "a tények nem adnak választ"
  * "kérdésére kollégánk hamarosan válaszol"
  * Bármilyen utalás arra, hogy "tények" vagy "források" alapján dolgozol
- SOHA ne használj [1], [2] hivatkozásokat a szövegben — a pályázó nem tudja mi az
- Ha Felhívás pontszámra hivatkozol, természetesen írd be: "a Pályázati felhívás 4.2. pontja szerint"
- Magázás mindig, tegezés soha
- Program neve: Otthoni Energiatároló Program. SOHA ne írd: "Otthonfelújítási Program"
- Ha eldöntendő kérdést tesz fel az ügyfél (pl. "megkereshetem-e?", "kell-e?", "lehet-e?") → válaszolj IGEN vagy NEM, ne kerülgesd

KONTEXTUS — TE EZT AZ EMAILCÍMET HASZNÁLOD:
- TE a lakossagitarolo@neuzrt.hu címről válaszolsz. SOHA ne kérd az ügyfelet, hogy "írjon a lakossagitarolo@neuzrt.hu-ra" — hiszen IDE írt és INNEN válaszolsz.
- Ha a tényekben szerepel ez a cím mint elérhetőség, NE idézd — az ügyfél már IDE írt.
- Ha más emailcímre kell irányítani (pl. oetpkivitelezo@neuzrt.hu kivitelezői kérdéseknél), azt írd meg.

GREETING ÉS ALÁÍRÁS:
- NE írj megszólítást — azt a rendszer adja.
- NE írj aláírást — azt is a rendszer adja.
- CSAK a tartalmi válasz legyen a body-ban.

ÉKEZETEK:
- MINDIG helyes magyar ékezetekkel írj. Ékezet nélküli szöveg ELFOGADHATATLAN.

PÉLDÁK HOGY HOGYAN ÍRNAK A KOLLÉGÁID:
- "Tájékoztatjuk, hogy a kért módosítást elvégeztük."
- "A pályázati felületet a meghatalmazottja éri el az ügyfélkapuján keresztül."
- "Az igazolási szakasz megnyitásáról a honlapon tájékoztatást adunk."
- "Kérjük, küldje meg a meghatalmazás aláírt példányát a lakossagitarolo@neuzrt.hu címre."

VÁLASZ FORMÁTUM (JSON):
{
  "body": "A válasz HTML-ben (<p> tagek). Természetes magyar szöveg, hivatkozások nélkül.",
  "confidence": "high|medium|skip",
  "used_facts": [1, 2],
  "citations": {"1": "dokumentum neve"},
  "unanswered_parts": null
}

A "skip" confidence azt jelenti: NINCS elég információ válaszolni → a rendszer nem ment draft-ot.
Inkább legyen skip, mint rossz vagy kitérő válasz."""

DRAFT_GENERATE_FEWSHOT = [
    {
        "role": "user",
        "content": """Beérkező email: "Mennyi a maximális támogatás és kell-e önerő?"
Tárgy: Támogatás összege

[TÉNY 1] (Felhivas_OETP.pdf)
"A támogatás összege pályázatonként legfeljebb 2.500.000 Ft."

[TÉNY 2] (Felhivas_OETP.pdf)
"A pályázónak a támogatáson felüli részt önerőből kell finanszíroznia."

Stílus: Tisztelt Pályázó! / Üdvözlettel:""",
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "body": "<p>Tájékoztatjuk, hogy a támogatás összege pályázatonként legfeljebb 2.500.000 Ft. A támogatáson felüli részt önerőből szükséges finanszírozni.</p>",
            "confidence": "high",
            "used_facts": [1, 2],
            "citations": {"1": "Felhivas_OETP.pdf", "2": "Felhivas_OETP.pdf"},
            "unanswered_parts": None,
        }, ensure_ascii=False),
    },
    {
        "role": "user",
        "content": """Beérkező email: "Mikor lesz az elbírálás vége?"
Tárgy: Elbírálás

[TÉNY 1] (kozlemeny:2026-03-16)
"Az Otthoni Energiatároló Program 2. pályázati ütem felfüggesztésre került."

Stílus: Tisztelt Pályázó! / Üdvözlettel:""",
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "body": "",
            "confidence": "skip",
            "used_facts": [],
            "citations": {},
            "unanswered_parts": "elbírálás vége — nincs adat a tényekben",
        }, ensure_ascii=False),
    },
]


def _strip_enrichment_prefix(text: str) -> str:
    """Remove contextual enrichment prefix from chunk text.

    The enrichment prefix looks like:
    "Ez az OETP (Otthonfelújítási Program) hivatalos pályázati felhívásának részlete. Forrás: X.pdf. ..."
    "Ez egy gyakran ismételt kérdés (GYIK) és válasz az OETP programról. Forrás: ..."
    "Ez egy korábbi ügyfélszolgálati email válasz..."

    This prefix is for embedding quality, NOT for the LLM or customer.
    """
    if not text:
        return text
    # Match known enrichment patterns at start
    patterns = [
        r"^Ez az OETP[^.]*\.\s*Forrás:[^.]*\.\s*[^.]*\.\s*",
        r"^Ez egy gyakran ismételt kérdés[^.]*\.\s*Forrás:[^.]*\.\s*",
        r"^Ez egy hivatalos közlemény[^.]*\.\s*Forrás:[^.]*\.\s*",
        r"^Ez egy korábbi ügyfélszolgálati[^.]*\.\s*Forrás:[^.]*\.\s*[^.]*\.\s*",
        r"^Ez az OETP pályázati felhívás hivatalos mellékletének részlete\.\s*Forrás:[^.]*\.\s*[^.]*\.\s*",
    ]
    for p in patterns:
        text = re.sub(p, "", text, count=1)
    return text.strip()


def _build_greeting(sender_name: str = "", category: str = "") -> str:
    """Build appropriate greeting based on sender name and email category.

    Priority:
    1. Personalized: "Tisztelt Kovács János!" (Hungarian name order: family first)
    2. Category-based: "Tisztelt Érdeklődő!" (for general inquiries)
    3. Default: "Tisztelt Pályázó!" (matches colleague convention)
    """
    # Skip generic/empty names
    skip_names = {"", "null", "none", "info", "admin", "support"}
    name = (sender_name or "").strip()

    if name and name.lower() not in skip_names and len(name) > 2:
        # Check if it looks like a real name (not email prefix)
        # Allow "." for prefixes like "Dr." but reject email-like patterns
        if "@" not in name and not re.match(r"^[\w.]+$", name):
            name = _normalize_hungarian_name(name)
            return f"Tisztelt {name}!"
        # Also allow names with Dr./Prof. prefix
        if re.match(r"^(Dr\.?|Prof\.?)\s+\w", name, re.IGNORECASE):
            name = _normalize_hungarian_name(name)
            return f"Tisztelt {name}!"

    # Category-based
    if category in ("altalanos", "hatarido"):
        return "Tisztelt Érdeklődő!"

    return "Tisztelt Pályázó!"


def _normalize_hungarian_name(name: str) -> str:
    """Normalize name to Hungarian convention: Family Given (not Given Family).

    Strategy: use a comprehensive Hungarian given name set to detect order.
    If first part is a known given name and second is not → swap to Family Given.
    """
    import re

    name = name.strip()
    if not name:
        return name

    # Extract prefix (Dr., Prof., stb.)
    prefix = ""
    prefix_match = re.match(r"^(Dr\.?|Prof\.?|Ifj\.?|Id\.?|Özv\.?)\s+", name, re.IGNORECASE)
    if prefix_match:
        prefix = prefix_match.group(0)
        name = name[len(prefix):].strip()

    # Fix ALL CAPS: "KOVÁCS JÁNOS" → "Kovács János"
    parts = name.split()
    if all(p.isupper() for p in parts if len(p) > 1):
        parts = [p.capitalize() for p in parts]

    # Contains -né suffix (e.g. Kovácsné, Győriné) → already Hungarian order
    # Must end with "né" — not just contain it (e.g. "Németh" should NOT match)
    if any(p.lower().endswith("né") for p in parts):
        return f"{prefix}{' '.join(parts)}"

    # 3+ parts → likely already Hungarian (Kovácsné Nagy Anna, or "Nagy Kiss Anna")
    if len(parts) >= 3:
        return f"{prefix}{' '.join(parts)}"

    # 2 parts: detect order using given name database
    # Check BOTH accented and unaccented forms (Graph API may strip accents)
    if len(parts) == 2:
        first_is_given = _is_given_name(parts[0])
        second_is_given = _is_given_name(parts[1])

        if first_is_given and not second_is_given:
            # English order: "Tamás Szegvári" → "Szegvári Tamás"
            parts = [parts[1], parts[0]]
        # If both are given names (rare) or neither → keep as-is

    # Fix accents on known given names (after reorder)
    parts = [_fix_given_name_accent(p) for p in parts]

    return f"{prefix}{' '.join(parts)}"


def _is_given_name(word: str) -> bool:
    """Check if word is a known Hungarian given name (accented or unaccented)."""
    lower = word.lower().rstrip('.')
    # Direct match (accented given names)
    if lower in _HUNGARIAN_GIVEN_NAMES:
        return True
    # Unaccented match — check if accent-fixed version is a given name
    fixed = _ACCENT_FIX_MAP.get(lower, "")
    if fixed and fixed in _HUNGARIAN_GIVEN_NAMES:
        return True
    return False


def _fix_given_name_accent(word: str) -> str:
    """Fix missing accents on known Hungarian given names.

    Graph API often strips accents: "Tamas" → "Tamás", "Jozsef" → "József".
    """
    lower = word.lower()
    accented = _ACCENT_FIX_MAP.get(lower)
    if not accented:
        return word
    if word[0].isupper():
        return accented[0].upper() + accented[1:]
    return accented


_ACCENT_FIX_MAP = {
    "adam": "ádám", "andras": "andrás", "arpad": "árpád", "balint": "bálint",
    "barnabas": "barnabás", "bela": "béla", "balazs": "balázs",
    "daniel": "dániel", "david": "dávid", "denes": "dénes", "dezso": "dezső",
    "erno": "ernő", "gabor": "gábor", "geza": "géza", "gergo": "gergő",
    "gyozo": "győző", "gyorgy": "györgy", "istvan": "istván", "ivan": "iván",
    "janos": "jános", "jozsef": "józsef", "karoly": "károly",
    "krisztian": "krisztián", "kristof": "kristóf", "laszlo": "lászló",
    "lorant": "lóránt", "marton": "márton", "mate": "máté", "matyas": "mátyás",
    "mihaly": "mihály", "miklos": "miklós", "milan": "milán", "oliver": "olivér",
    "oszkar": "oszkár", "otto": "ottó", "pal": "pál", "peter": "péter",
    "rene": "rené", "richard": "richárd", "robert": "róbert", "sandor": "sándor",
    "szilard": "szilárd", "tamas": "tamás", "zoltan": "zoltán",
    "kalman": "kálmán", "lajos": "lajos",
    # Female
    "agnes": "ágnes", "aniko": "anikó", "eniko": "enikő", "erzsebet": "erzsébet",
    "eva": "éva", "ildiko": "ildikó", "maria": "mária", "marta": "márta",
    "monika": "mónika", "nora": "nóra", "noemi": "noémi", "reka": "réka",
    "renata": "renáta", "rozalia": "rozália", "terez": "teréz",
    "terezia": "terézia", "timea": "tímea", "tunde": "tünde",
    "valeria": "valéria", "viktoria": "viktória", "diana": "diána",
    "dora": "dóra", "julia": "júlia",
    # Common family names with accents (top ~50 ékezetes családnév)
    "kovacs": "kovács", "szabo": "szabó", "toth": "tóth", "horvath": "horváth",
    "varga": "varga", "nemeth": "németh", "farkas": "farkas", "balogh": "balogh",
    "papp": "papp", "takacs": "takács", "juhasz": "juhász", "kis": "kis",
    "szucs": "szűcs", "hajdu": "hajdú", "lukacs": "lukács", "gulyas": "gulyás",
    "biro": "bíró", "kiraly": "király", "lazar": "lázár", "bognar": "bognár",
    "orban": "orbán", "fulop": "fülöp", "vincze": "vincze", "hegedus": "hegedűs",
    "szekely": "székely", "szalai": "szalai", "feher": "fehér", "torok": "török",
    "lengyel": "lengyel", "fazekas": "fazekas", "mate": "máté",
    "bernat": "bernát", "csaszar": "császár", "deak": "deák",
    "erdelyi": "erdélyi", "foldi": "földi", "galfi": "gálfi",
    "halasz": "halász", "illes": "illés", "jakab": "jakab",
    "kelemen": "kelemen", "lokos": "lőkös", "meszaros": "mészáros",
    "nadasdi": "nádasdi", "olah": "oláh", "palfi": "pálfi",
    "racz": "rácz", "sarkozy": "sárközy", "tatai": "tatai",
    "ujvari": "újvári", "vasarhelyi": "vásárhelyi", "zelei": "zelei",
    "puskas": "puskás", "szegvari": "szegvári", "csiszar": "csiszár",
    "fabian": "fábián", "csanyi": "csányi", "becsey": "becsey",
}


# Common Hungarian given names (for name order detection)
_HUNGARIAN_GIVEN_NAMES = {
    # Male
    "ádám", "andrás", "antal", "attila", "balázs", "bálint", "barnabás",
    "béla", "bence", "benedek", "benjamin", "botond", "csaba", "dániel",
    "dávid", "dénes", "dezső", "dominik", "endre", "erik", "ernő",
    "ferenc", "gábor", "géza", "gergő", "gergely", "gli", "gustav",
    "gusztáv", "győző", "gyula", "györgy", "henrik", "hunor", "ignác",
    "imre", "istván", "iván", "jakab", "jános", "jenő", "józsef",
    "károly", "krisztián", "kristóf", "lajos", "lászló", "levente",
    "lóránt", "márk", "márton", "máté", "mátyás", "mihály", "miklós",
    "milán", "norbert", "olivér", "oszkár", "ottó", "pál", "patrik",
    "péter", "rené", "richárd", "róbert", "roland", "sándor", "sebestyén",
    "szabolcs", "szilárd", "tamás", "tibor", "tivadar", "viktor",
    "vilmos", "vince", "zoltán", "zsolt", "zsombor",
    # Female
    "ágnes", "andrea", "angéla", "anikó", "anna", "annamária", "anett",
    "barbara", "beatrix", "bernadett", "boglárka", "brigitta",
    "csilla", "diána", "dóra", "edit", "edina", "emese", "enikő",
    "erika", "erzsébet", "eszter", "éva", "fruzsina",
    "gabriella", "gitta", "györgyi", "hajnalka", "helga",
    "ildikó", "ilona", "irén", "ágota", "judit", "julianna", "júlia",
    "katalin", "kinga", "klára", "krisztina", "laura", "lilla",
    "mária", "margit", "marianna", "márta", "melinda", "mónika",
    "nóra", "noémi", "nikolett", "orsolya", "piroska",
    "réka", "renáta", "rita", "rozália", "sarolta", "szilvia",
    "teréz", "terézia", "tímea", "tünde", "valéria", "vera",
    "veronika", "viktória", "virág", "vivien", "zita", "zsuzsanna",
    "evelin", "petra", "bianka", "fanni",
}


NEU_SIGNATURE_HTML = (
    '<p>Üdvözlettel:<br>'
    'Nemzeti Energetikai Ügynökség<br>'
    'Zártkörűen Működő Részvénytársaság<br>'
    '1037- Budapest, Montevideo u. 14.</p>'
)


def _fix_greeting_and_signature(body_html: str, correct_greeting: str) -> str:
    """Replace LLM-generated greeting and signature with deterministic versions.

    Problems this fixes:
    1. LLM writes names in English order (Given Family instead of Family Given)
    2. LLM drops accents from names (Meszaros instead of Mészáros)
    3. LLM uses short signature ("Üdvözlettel:") instead of full NEÜ block
    4. LLM sometimes adds extra greeting variations

    Strategy: strip LLM greeting + closing, wrap content with correct ones.
    """
    import re

    if not body_html:
        return f"<p>{correct_greeting}</p>{NEU_SIGNATURE_HTML}"

    # 1. Remove ALL LLM greetings ("Tisztelt ...!" patterns)
    # The LLM sometimes adds greeting despite being told not to
    body_html = re.sub(
        r'<p>\s*Tisztelt\s+[^<]*?!\s*</p>',
        '',
        body_html,
    )
    # Also catch greeting at start of a paragraph (not in its own <p>)
    body_html = re.sub(
        r'(<p>)\s*Tisztelt\s+[^!]*!\s*',
        r'\1',
        body_html,
        count=1,
    )

    # 2. Remove LLM signature variations
    # "Üdvözlettel:" or "Üdvözlettel:\nNEÜ Zrt." etc. — everything from last "Üdvözlettel"
    body_html = re.sub(
        r'<p>\s*Üdvözlettel\s*:?\s*(?:<br\s*/?>.*?)?</p>\s*(?:<p>.*?(?:Nemzeti|NEÜ|Zrt|Montevideo|1037).*?</p>\s*)*$',
        '',
        body_html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Also handle simple "Üdvözlettel:" at end without <p>
    body_html = re.sub(
        r'\s*Üdvözlettel\s*:?\s*$',
        '',
        body_html,
        flags=re.IGNORECASE,
    )

    # 3. Remove inline [N] citations — these are internal, not for the customer
    body_html = re.sub(r'\s*\[\d+\]', '', body_html)

    # 4. Clean up empty paragraphs
    body_html = re.sub(r'<p>\s*</p>', '', body_html)
    body_html = body_html.strip()

    # 5. Reassemble: greeting + LLM content + NEÜ signature
    return f"<p>{correct_greeting}</p>{body_html}{NEU_SIGNATURE_HTML}"


def _validate_citations(body_html: str, facts: list[dict], citations: dict) -> dict:
    """Validate that inline citations [1], [2] in the draft body are legitimate.

    Checks:
    1. Every [N] reference maps to a real fact
    2. Counts claims without any citation
    """
    plain = re.sub(r"<[^>]+>", "", body_html)

    # Find all [N] references in the text
    cited_nums = set(int(m) for m in re.findall(r"\[(\d+)\]", plain))

    # Check that cited numbers are valid fact indices
    invalid_refs = [n for n in cited_nums if n < 1 or n > len(facts)]

    # Count sentences without citations (rough heuristic)
    # Split on sentence boundaries, skip greeting/closing
    sentences = re.split(r"[.!?]\s+", plain)
    # Filter out very short segments, greeting, closing
    content_sentences = [
        s for s in sentences
        if len(s.strip()) > 20
        and "tisztelt" not in s.lower()
        and "üdvözlettel" not in s.lower()
        and "kollégánk" not in s.lower()  # standard "no info" phrase is OK uncited
        and "kérdésére" not in s.lower()
    ]
    uncited = [s for s in content_sentences if not re.search(r"\[\d+\]", s)]

    return {
        "total_citations": len(cited_nums),
        "invalid_refs": invalid_refs,
        "uncited_claims": len(uncited),
        "total_content_sentences": len(content_sentences),
    }


def _check_numerical_consistency(body_html: str, facts: list[dict]) -> list[str]:
    """Check that numbers in the draft match numbers in the source facts.

    Extracts monetary amounts (Ft), percentages, and specific numbers
    from both draft and facts, then flags mismatches.
    """
    plain = re.sub(r"<[^>]+>", "", body_html).lower()
    facts_text = " ".join(f["text"] for f in facts).lower()

    warnings = []

    # 1. Monetary amounts: "4.000.000 ft", "4 millió ft", "2,5 millió"
    def _extract_amounts(text: str) -> set[int]:
        amounts = set()
        # "X.XXX.XXX Ft" pattern
        for m in re.finditer(r"([\d.]+)\s*(?:ft|forint)", text):
            try:
                amounts.add(int(m.group(1).replace(".", "")))
            except ValueError:
                pass
        # "X millió" pattern
        for m in re.finditer(r"([\d,]+)\s*milli[oó]", text):
            try:
                val = float(m.group(1).replace(",", "."))
                amounts.add(int(val * 1_000_000))
            except ValueError:
                pass
        return amounts

    draft_amounts = _extract_amounts(plain)
    fact_amounts = _extract_amounts(facts_text)

    if draft_amounts and fact_amounts:
        novel_amounts = draft_amounts - fact_amounts
        if novel_amounts:
            warnings.append(
                f"Ft összeg a draftban ({novel_amounts}) nem található a forrásokban ({fact_amounts})"
            )

    # 2. Percentages: "10%", "10 százalék"
    draft_pcts = set(re.findall(r"(\d+)\s*(?:%|százalék)", plain))
    fact_pcts = set(re.findall(r"(\d+)\s*(?:%|százalék)", facts_text))
    if draft_pcts and fact_pcts:
        novel_pcts = draft_pcts - fact_pcts
        if novel_pcts:
            warnings.append(f"Százalék a draftban ({novel_pcts}) nem található a forrásokban ({fact_pcts})")

    # 3. Dates: "2026.03.16", "2026. március 16"
    draft_dates = set(re.findall(r"\d{4}[\.\s]+\d{2}[\.\s]+\d{2}", plain))
    fact_dates = set(re.findall(r"\d{4}[\.\s]+\d{2}[\.\s]+\d{2}", facts_text))
    if draft_dates:
        novel_dates = draft_dates - fact_dates
        if novel_dates:
            warnings.append(f"Dátum a draftban ({novel_dates}) nem található a forrásokban ({fact_dates})")

    # 4. Specific kW values
    draft_kw = set(re.findall(r"(\d+)\s*kw", plain))
    fact_kw = set(re.findall(r"(\d+)\s*kw", facts_text))
    if draft_kw and fact_kw:
        novel_kw = draft_kw - fact_kw
        if novel_kw:
            warnings.append(f"kW érték a draftban ({novel_kw}) nem található a forrásokban ({fact_kw})")

    return warnings


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
    ref_check = None
    try:
        if req.sender_email:
            import asyncpg as _apg
            _econn = await _apg.connect(
                os.environ.get("HANNA_PG_DSN", "postgresql://klara:klara_docs_2026@hanna-db:5432/hanna_oetp")
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

    # 1c. Temporal staleness check — warn if chunks are outdated
    from datetime import datetime as _dt, timedelta as _td
    _now = _dt.now()
    _stale_threshold = _now - _td(days=90)
    stale_chunks = []
    for _tc in top_chunks:
        _meta = _tc.get("metadata", {})
        _valid_to = _meta.get("valid_to", "")
        _source_date_str = _meta.get("source_date") or _meta.get("valid_from", "")
        if _valid_to and _valid_to.strip():
            stale_chunks.append(f"{_tc.get('source','?')} (lejárt: {_valid_to})")
        elif _source_date_str:
            try:
                _sd = _dt.fromisoformat(str(_source_date_str)[:10])
                if _sd < _stale_threshold:
                    stale_chunks.append(f"{_tc.get('source','?')} (régi: {_source_date_str})")
            except (ValueError, TypeError):
                pass
    if stale_chunks:
        print(f"[hanna] Stale chunks in top results: {stale_chunks}")

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
    except Exception as e:
        print(f"[hanna] VerbatimRAG unavailable: {e}")

    # Fallback: if VerbatimRAG failed, use raw chunk texts — but mark clearly as unverified
    verbatim_available = bool(verified_facts)
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

    # 2b. Program consistency check — ensure chunks are about OETP, not NPP/other
    program_counts = {}
    for r in top_chunks:
        prog = r.get("category", r.get("metadata", {}).get("program", "OETP"))
        if isinstance(r.get("metadata"), dict):
            prog = r["metadata"].get("program", prog)
        program_counts[prog] = program_counts.get(prog, 0) + 1
    dominant_program = max(program_counts, key=program_counts.get) if program_counts else "OETP"
    if dominant_program != "OETP" and program_counts.get(dominant_program, 0) > len(top_chunks) // 2:
        print(f"[hanna] WARNING: chunks dominated by {dominant_program}, not OETP")

    # 3. Build LLM prompt with verified facts
    facts_block = ""
    for i, f in enumerate(verified_facts, 1):
        # Strip enrichment prefix — it's for embedding, not for LLM/customer
        fact_text = _strip_enrichment_prefix(f["text"])
        facts_block += f'[TÉNY {i}] ({f["source"]})\n"{fact_text}"\n\n'

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

    # 4. LLM generation — multi-provider with automatic fallback
    from .llm_client import chat_completion

    messages = [
        {"role": "system", "content": DRAFT_GENERATE_SYSTEM},
        *DRAFT_GENERATE_FEWSHOT,
        {"role": "user", "content": user_msg},
    ]

    try:
        llm_result = await chat_completion(
            messages=messages,
            temperature=0.15,
            max_tokens=1000,
            json_mode=True,
        )
        raw = llm_result["content"]
        llm_provider = llm_result["provider"]
        llm_model = llm_result["model"]
        draft_data = json.loads(raw)
    except Exception as e:
        # LLM failed — SKIP, never dump raw chunks as a "response"
        print(f"[hanna] LLM FAILED: {e} — skipping draft (no raw chunk dump)")
        return {
            "skip": True,
            "skip_reason": f"llm_failed: {e}",
            "body_html": None,
            "confidence": "skip",
            "sources": fact_sources,
            "method": "skip_llm_error",
            "error": str(e),
        }

    raw_body = draft_data.get("body", "")
    confidence = draft_data.get("confidence", "medium")
    citations = draft_data.get("citations", {})

    # If LLM says "skip" — not enough info to answer, don't create a draft
    if confidence == "skip" or not raw_body or not raw_body.strip():
        return {
            "skip": True,
            "skip_reason": "insufficient_facts",
            "body_html": None,
            "confidence": "skip",
            "sources": fact_sources,
            "method": "skip_no_answer",
            "unanswered": draft_data.get("unanswered_parts"),
        }

    # 4b. Deterministic greeting + signature (never trust LLM for these)
    body_html = _fix_greeting_and_signature(raw_body, greeting)

    # 4c. Citation validation — check that inline [N] refs exist and map to real facts
    citation_warnings = _validate_citations(body_html, verified_facts, citations)
    if citation_warnings.get("uncited_claims"):
        print(f"[hanna] Citation warning: {citation_warnings['uncited_claims']} uncited claims")
        if confidence == "high":
            confidence = "medium"

    # 4d. Domain guardrails — numerical, eligibility, contradictions, forbidden phrases
    from .rag.guardrails import run_all_guardrails
    guardrails_result = run_all_guardrails(
        body_html=body_html,
        verified_facts=verified_facts,
        top_chunks=top_chunks,
        email_oetp_ids=req.oetp_ids or None,
        email_text=req.email_text,
        citations=citations,
    )
    if not guardrails_result["pass"]:
        for w in guardrails_result["warnings"]:
            print(f"[hanna] Guardrail {w['rule']}: {w['detail']}")
        if guardrails_result["suggested_confidence"] == "low":
            confidence = "low"
        elif guardrails_result["suggested_confidence"] == "medium" and confidence == "high":
            confidence = "medium"

    # 4e. Accent safety check — reject accent-free Hungarian drafts
    _accent_chars = set("áéíóöőúüűÁÉÍÓÖŐÚÜŰ")
    _plain_text = re.sub(r"<[^>]+>", "", body_html)
    if len(_plain_text) > 80 and not any(c in _accent_chars for c in _plain_text):
        # LLM generated accent-free text — this should never happen but guard against it
        print(f"[hanna] WARNING: accent-free draft detected, forcing low confidence")
        confidence = "low"

    # 5. NLI faithfulness verification
    nli_result = None
    nli_failed = False
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
                    print(f"[hanna] NLI: unfaithful → confidence=low")
            else:
                nli_failed = True
    except Exception as e:
        nli_failed = True
        print(f"[hanna] NLI service unavailable: {e}")

    # NLI fail + no VerbatimRAG = double-blind — downgrade confidence
    if nli_failed and not verbatim_available:
        if confidence == "high":
            confidence = "medium"
            print(f"[hanna] No NLI + no VerbatimRAG → confidence downgraded to medium")

    # 6. CoVe (Chain of Verification) — runs for non-high confidence drafts
    cove_result = None
    if confidence != "high":
        try:
            from .rag.cove import verify_draft
            _draft_plain = re.sub(r"<[^>]+>", "", body_html)
            cove_result = await verify_draft(_draft_plain, verified_facts)
            if cove_result.get("issue_count", 0) > 0:
                issues = cove_result["issues"]
                contradicted = [i for i in issues if i.get("verdict") == "contradicted"]
                if contradicted:
                    confidence = "low"
                    print(f"[hanna] CoVe: {len(contradicted)} CONTRADICTED claims → confidence=low")
                else:
                    print(f"[hanna] CoVe: {len(issues)} unsupported claims (no contradiction)")
        except Exception as e:
            print(f"[hanna] CoVe failed (non-blocking): {e}")

    # 7. Answer-Question Alignment — does the draft actually answer the question?
    alignment_result = None
    if confidence != "skip":
        try:
            from .rag.answer_alignment import check_alignment
            _draft_plain = re.sub(r"<[^>]+>", "", body_html)
            alignment_result = await check_alignment(req.email_text, _draft_plain)
            if not alignment_result.get("aligned", True):
                verdict = alignment_result.get("verdict", "?")
                reason = alignment_result.get("reason", "")
                if verdict == "echoes":
                    # Draft just repeats what customer said — useless, skip it
                    print(f"[hanna] Alignment: ECHOES ({reason}) → skip")
                    return {
                        "skip": True,
                        "skip_reason": f"echo_detected: {reason}",
                        "body_html": None,
                        "confidence": "skip",
                        "sources": fact_sources,
                        "method": "skip_echo",
                    }
                elif verdict == "irrelevant":
                    print(f"[hanna] Alignment: IRRELEVANT ({reason}) → skip")
                    return {
                        "skip": True,
                        "skip_reason": f"irrelevant_answer: {reason}",
                        "body_html": None,
                        "confidence": "skip",
                        "sources": fact_sources,
                        "method": "skip_irrelevant",
                    }
        except Exception as e:
            print(f"[hanna] Alignment check failed (non-blocking): {e}")

    # 8. SelfCheck — multi-sample consistency (only for medium, cost control)
    selfcheck_result = None
    if confidence == "medium":
        try:
            from .rag.selfcheck import selfcheck
            selfcheck_result = await selfcheck(
                messages=messages,
                original_response=raw_body,
                n_samples=2,
            )
            if not selfcheck_result.get("consistent", True):
                confidence = "low"
                print(f"[hanna] SelfCheck: INCONSISTENT (min_sim={selfcheck_result['min_similarity']}) → confidence=low")
        except Exception as e:
            print(f"[hanna] SelfCheck failed (non-blocking): {e}")

    # 8. Final confidence gate — "low" → route to human review
    draft_category = "Hanna - draft kész"
    if confidence == "low":
        draft_category = "Hanna - emberi válasz kell"

    return {
        "skip": False,
        "body_html": body_html,
        "confidence": confidence,
        "draft_category": draft_category,
        "sources": fact_sources,
        "method": "verbatim+llm" if verbatim_available else "chunks+llm",
        "model_used": llm_model,
        "llm_provider": llm_provider,
        "facts_count": len(verified_facts),
        "nli_verification": nli_result,
        "nli_failed": nli_failed,
        "verbatim_available": verbatim_available,
        "citations": citations,
        "citation_warnings": citation_warnings if citation_warnings.get("uncited_claims") else None,
        "guardrails": guardrails_result if not guardrails_result["pass"] else None,
        "cove_verification": cove_result,
        "selfcheck": selfcheck_result,
        "alignment": alignment_result,
        "reference_check": ref_check,
        "radix_data": ctx.get("radix_data") or None,
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
            os.environ.get("HANNA_PG_DSN", "postgresql://klara:klara_docs_2026@hanna-db:5432/hanna_oetp"),
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


# ─── Obsidian RAG: moved to standalone service (obsidian-rag :8115) ──────────
# All /obsidian/* endpoints removed. Use http://localhost:8115/ instead.
# See: ~/DEV/obsidian-rag/


_placeholder_removed_obsidian = True  # marker


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
    from .llm_client import chat_completion
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

    try:
        messages = [
            {"role": "system", "content": HANNA_ANSWER_SYSTEM_PROMPT},
            *HANNA_ANSWER_FEWSHOT,
            {"role": "user", "content": user_msg},
        ]
        llm_result = await chat_completion(
            messages=messages,
            temperature=settings.answer_temperature,
            max_tokens=settings.answer_max_tokens,
            json_mode=True,
        )
        raw = llm_result["content"]
        model = llm_result["model"]
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


# ─── Evaluation ─────────────────────────────────────────────────────────────

@app.post("/eval/live")
async def eval_live(
    limit: int = 50,
    mailbox: str = "lakossagitarolo@neuzrt.hu",
    report: bool = False,
    background_tasks: BackgroundTasks = None,
):
    """Run live email evaluation (compares Hanna drafts vs actual colleague answers).

    This is the end-to-end quality check: fetches recent sent emails, extracts
    the customer question, generates a Hanna draft, and compares it against
    the actual answer using semantic + style + term overlap metrics.

    Use limit=20 for a quick check, limit=250 for full eval.
    Set report=true to generate an Obsidian report.
    """
    from scripts.eval_live import run_eval
    stats = await run_eval(
        mailbox=mailbox,
        max_items=limit,
        dry_run=False,
        generate_report=report,
    )
    return stats
