"""Hanna Backend — FastAPI REST API for RAG + Email."""

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
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
async def ingest_text(doc: DocumentIngest):
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
        return IngestResult(
            chunks_created=count,
            source=doc.source,
            collection="postgresql",  # Legacy field (migrated from ChromaDB)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/pdf", response_model=IngestResult)
async def ingest_pdf(
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
        return IngestResult(
            chunks_created=count,
            source=file.filename,
            collection="postgresql",  # Legacy field (migrated from ChromaDB)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/email-pair", response_model=IngestResult)
async def ingest_email_pair(
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
        
        return SearchResponse(
            results=[SearchResult(**r) for r in results],
            referenced_chunks=ref_chunks,
            query=query.query,
            total_found=len(results),
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
async def invalidate_chunks(req: InvalidateRequest):
    """Mark chunks as invalid (sets valid_to = today).
    
    Chunks are not deleted, just marked as outdated.
    Use only_valid=true in search to exclude them.
    """
    try:
        result = rag_search.invalidate_chunks(req.chunk_ids, req.reason)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Email: Polling ───────────────────────────────────────────────────────────

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


# ─── Email: Feedback ──────────────────────────────────────────────────────────

@app.post("/emails/feedback/check")
async def feedback_check(mailbox: str = "info@neuzrt.hu", hours: int = 48):
    """Compare sent emails with stored Hanna drafts.
    
    Returns how many drafts were accepted unchanged vs modified.
    """
    try:
        result = await feedback.check_feedback(mailbox=mailbox, hours=hours)
        return result
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

@app.post("/obsidian/ingest")
async def ingest_obsidian(
    vault_path: str = "/app/obsidian-vault",
    force: bool = False,
):
    """Ingest Obsidian vault with incremental sync.
    
    Args:
        vault_path: Path to Obsidian vault
        force: If True, re-process all files regardless of hash
    """
    try:
        result = await obsidian_ingest.ingest_vault(
            vault_path=vault_path,
            force=force,
            collection_name="obsidian_notes"
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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

        response = {
            "results": results,
            "query": q,
            "total_found": len(results),
            "folder_filter": folder,
            "method": method,
            "compact": compact,
            "graph_boosted_count": graph_boosted,
        }
        if graph_context:
            response["graph_context"] = graph_context

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
