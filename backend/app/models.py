"""Pydantic models for Hanna API."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# --- Ingestion ---

class DocumentIngest(BaseModel):
    """Ingest a document (text/markdown) into the knowledge base."""
    text: str
    source: str  # filename or URL
    category: str = "general"
    chunk_type: str = "document"  # document | email_reply | faq
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    version: int = 1
    supersedes: Optional[str] = None


class IngestResult(BaseModel):
    chunks_created: int
    source: str
    collection: str


# --- Search ---

class SearchQuery(BaseModel):
    """Search the knowledge base."""
    query: str
    top_k: int = 5
    category: Optional[str] = None
    chunk_type: Optional[str] = None
    only_valid: bool = True  # filter out expired chunks


class SearchResult(BaseModel):
    id: Optional[str] = None  # Chunk ID for updates/invalidation
    text: str
    source: str
    category: str
    chunk_type: str
    score: float
    metadata: dict = {}
    rerank_score: Optional[float] = None
    semantic_score: Optional[float] = None
    rrf_score: Optional[float] = None
    authority_weight: Optional[float] = None
    pre_authority_score: Optional[float] = None


class ReferencedChunk(BaseModel):
    """A chunk resolved from a cross-reference in the main results."""
    id: Optional[str] = None
    text: str
    source: str
    category: str
    chunk_type: str
    score: float
    ref_type: str           # "felhivas", "gyik", "melleklet", "segedlet"
    ref_section: str        # "4.2", "12", "*"
    ref_text: str           # Original reference text found


class SearchResponse(BaseModel):
    results: list[SearchResult]
    referenced_chunks: list[ReferencedChunk] = []
    query: str
    total_found: int


# --- Email ---

class EmailMessage(BaseModel):
    """Representation of an email from Graph API."""
    id: str
    subject: str = "(Nincs tárgy)"
    sender: str
    sender_email: str
    body_text: str
    body_html: Optional[str] = None
    received_at: str
    conversation_id: Optional[str] = None
    in_reply_to: Optional[str] = None
    mailbox: str  # which shared mailbox
    has_attachments: bool = False
    importance: str = "normal"
    # Auto-extracted identifiers
    oetp_ids: list[str] = Field(default_factory=list)  # e.g. ["OETP-2026-123456"]
    pod_numbers: list[str] = Field(default_factory=list)  # e.g. ["HU-ELMU-xxx"]
    # Outlook categories (for dedup: skip if Hanna already processed)
    categories: list[str] = Field(default_factory=list)


class EmailThread(BaseModel):
    """A thread of related emails."""
    conversation_id: str
    messages: list[EmailMessage]
    subject: str
    mailbox: str


class DraftRequest(BaseModel):
    """Request to save a draft reply."""
    mailbox: str  # shared mailbox to create draft in
    reply_to_message_id: str  # Graph API message ID to reply to
    body_html: str  # HTML body of draft
    confidence: str = "medium"  # high | medium | low


class DraftResult(BaseModel):
    draft_id: str
    mailbox: str
    subject: str
    confidence: str


# --- Email Poll ---

class PollResult(BaseModel):
    """Result of polling for new emails."""
    new_emails: int
    mailbox: str
    messages: list[EmailMessage]


class BatchPollResult(BaseModel):
    """Result of polling all mailboxes."""
    results: list[PollResult]
    total_new: int


# --- Attachments ---

class AttachmentInfo(BaseModel):
    """Attachment metadata."""
    id: str
    name: str
    content_type: str
    size: int
    is_image: bool


class ImageAnalysisResult(BaseModel):
    """Result of analyzing an image attachment."""
    attachment_id: str
    filename: str
    description: str
    error: Optional[str] = None


class AttachmentAnalysisResponse(BaseModel):
    """Response from attachment analysis endpoint."""
    mailbox: str
    message_id: str
    total_attachments: int
    images_analyzed: int
    results: list[ImageAnalysisResult]


# --- Health ---

class HealthResponse(BaseModel):
    status: str
    chromadb: str
    collection_count: int
    version: str = "0.1.0"
