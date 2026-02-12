"""Hanna Backend — FastAPI REST API for RAG + Email."""

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
from .email import poller, drafts, history, attachments
from .obsidian import ingest as obsidian_ingest
from .obsidian import search as obsidian_search


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown."""
    # Verify ChromaDB connection
    try:
        stats = rag_search.get_collection_stats()
        print(f"[hanna] ChromaDB connected: {stats}")
    except Exception as e:
        print(f"[hanna] WARNING: ChromaDB not reachable: {e}")
    
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
    """Health check."""
    try:
        stats = rag_search.get_collection_stats()
        chroma_status = "connected"
        count = stats.get("total_chunks", 0)
    except Exception:
        chroma_status = "disconnected"
        count = 0

    return HealthResponse(
        status="ok",
        chromadb=chroma_status,
        collection_count=count,
    )


@app.get("/reranker/status")
async def reranker_status():
    """Get reranker status."""
    return reranker.get_status()


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
            collection=settings.chroma_collection,
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
            collection=settings.chroma_collection,
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
            collection=settings.chroma_collection,
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
    """Hybrid search: semantic + BM25 → RRF → Cohere rerank."""
    try:
        results = await rag_search.search_async(
            query=query.query,
            top_k=query.top_k,
            category=query.category,
            chunk_type=query.chunk_type,
            only_valid=query.only_valid,
        )
        return SearchResponse(
            results=[SearchResult(**r) for r in results],
            query=query.query,
            total_found=len(results),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def stats():
    """Knowledge base statistics."""
    return rag_search.get_collection_stats()


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
async def poll_emails():
    """Poll all shared mailboxes for new emails."""
    try:
        results = await poller.poll_all_mailboxes()
        total = sum(r.new_emails for r in results)
        return BatchPollResult(results=results, total_new=total)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/emails/thread/{mailbox}/{conversation_id}")
async def get_thread(mailbox: str, conversation_id: str):
    """Get full email thread by conversation ID."""
    try:
        messages = await poller.get_email_thread(mailbox, conversation_id)
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
        result = obsidian_ingest.ingest_vault(
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
        results = obsidian_search.search_obsidian_notes(
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
):
    """Hybrid search: semantic + BM25 → RRF fusion → reranking.
    
    Args:
        q: Search query
        limit: Maximum number of results (default 10)
        folder: Optional folder filter
        caller: Who made the search (bob, max, eve, etc.)
        rerank: Whether to apply reranking (default True)
        instruction: Custom instruction for reranker
    """
    try:
        results = await obsidian_search.search_obsidian_hybrid(
            query=q,
            limit=limit,
            folder_filter=folder,
            collection_name="obsidian_notes",
            caller=caller,
            use_reranker=rerank,
            instruction=instruction,
        )
        return {
            "results": results,
            "query": q,
            "total_found": len(results),
            "folder_filter": folder,
            "method": "hybrid+rerank" if rerank else "hybrid"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/obsidian/stats")
async def obsidian_stats():
    """Obsidian notes collection statistics."""
    try:
        return obsidian_search.get_obsidian_stats("obsidian_notes")
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
        return obsidian_ingest.get_last_sync_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
