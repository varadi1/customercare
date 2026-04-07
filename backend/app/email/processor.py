"""
Autonomous email processor — replaces OpenClaw agent orchestration.

Complete pipeline: poll → filter → RAG → draft → save → notify.
No LLM for decision-making — deterministic rules only.
LLM used only for: draft text generation, image analysis, legal context.

Usage:
  Scheduled: APScheduler or cron calls POST /emails/process
  Manual: POST /emails/process?hours=4
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any

import httpx

from ..config import settings
from . import poller, drafts, draft_context, feedback
from .skip_filter import check_skip
from .attachments import list_attachments, analyze_email_attachments

logger = logging.getLogger(__name__)


async def process_new_emails(hours: float = 4) -> dict[str, Any]:
    """Complete autonomous email processing pipeline.

    1. Poll all mailboxes for new emails
    2. For each email: skip filter → RAG → draft → save
    3. Return processing summary

    No OpenClaw agent needed — fully self-contained.
    """
    stats = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "emails_polled": 0,
        "drafts_created": 0,
        "skipped": 0,
        "skipped_reasons": {},
        "errors": 0,
        "high_confidence": 0,
        "medium_confidence": 0,
        "low_confidence": 0,
        "legal_queries": 0,
        "image_analyses": 0,
        "oetp_db_lookups": 0,
        "details": [],
    }

    # 1. Poll all mailboxes
    try:
        poll_results = await poller.poll_all_mailboxes(hours=hours)
    except Exception as e:
        logger.error("Poll failed: %s", e)
        stats["errors"] += 1
        return stats

    all_messages = []
    for pr in poll_results:
        all_messages.extend(pr.messages)
    stats["emails_polled"] = len(all_messages)

    if not all_messages:
        logger.info("No new emails to process")
        return stats

    # 2. Process each email
    for msg in all_messages:
        try:
            result = await _process_single_email(msg)
            stats["details"].append(result)

            if result["status"] == "skipped":
                stats["skipped"] += 1
                reason = result.get("skip_reason", "unknown")
                stats["skipped_reasons"][reason] = stats["skipped_reasons"].get(reason, 0) + 1
            elif result["status"] == "draft_created":
                stats["drafts_created"] += 1
                conf = result.get("confidence", "medium")
                if conf == "high":
                    stats["high_confidence"] += 1
                elif conf == "medium":
                    stats["medium_confidence"] += 1
                else:
                    stats["low_confidence"] += 1
            elif result["status"] == "error":
                stats["errors"] += 1

            if result.get("legal_query"):
                stats["legal_queries"] += 1
            if result.get("image_analysis"):
                stats["image_analyses"] += 1
            if result.get("oetp_lookup"):
                stats["oetp_db_lookups"] += 1

        except Exception as e:
            logger.error("Error processing email %s: %s", msg.subject[:50], e)
            stats["errors"] += 1
            stats["details"].append({
                "subject": msg.subject[:80],
                "status": "error",
                "error": str(e),
            })

    stats["finished_at"] = datetime.now(timezone.utc).isoformat()

    # 3. Discord notification (if configured)
    if settings.discord_webhook_url:
        try:
            await _send_discord_summary(stats)
        except Exception as e:
            logger.warning("Discord notification failed: %s", e)

    logger.info(
        "Processing complete: %d polled, %d drafts, %d skipped, %d errors",
        stats["emails_polled"], stats["drafts_created"],
        stats["skipped"], stats["errors"],
    )

    return stats


async def _process_single_email(msg) -> dict[str, Any]:
    """Process a single email: filter → context → draft → save."""
    result = {
        "subject": msg.subject[:80],
        "sender": msg.sender,
        "sender_email": msg.sender_email,
        "status": "pending",
    }

    # Step 0: Skip internal emails — no draft needed for colleague-to-colleague
    INTERNAL_DOMAINS = {"neuzrt.hu", "nffku.hu", "nffku.onmicrosoft.com", "norvegalap.hu"}
    sender_domain = (msg.sender_email or "").lower().split("@")[-1]
    if sender_domain in INTERNAL_DOMAINS:
        result["status"] = "skipped"
        result["skip_reason"] = "internal_email"
        return result

    # Step 1: Skip filter (deterministic — no LLM)
    skip_info = check_skip(msg.body_text, msg.subject)
    if skip_info["skip"]:
        result["status"] = "skipped"
        result["skip_reason"] = skip_info["reason"]
        return result

    # Step 2: Check if already has Hanna category (dedup)
    if any(cat.startswith("Hanna -") for cat in (msg.categories or [])):
        result["status"] = "skipped"
        result["skip_reason"] = "already_processed"
        return result

    # Step 3: Build draft context (RAG + style + similar traces)
    ctx = await draft_context.build_draft_context(
        email_text=msg.body_text,
        email_subject=msg.subject,
        oetp_ids=msg.oetp_ids,
        pod_numbers=msg.pod_numbers,
    )

    if ctx.get("skip"):
        result["status"] = "skipped"
        result["skip_reason"] = ctx.get("skip_reason", "context_skip")
        return result

    # Step 4: Image analysis (if attachments)
    image_context = ""
    if msg.has_attachments:
        try:
            image_context = await _analyze_images(msg.mailbox, msg.id)
            if image_context:
                result["image_analysis"] = True
        except Exception as e:
            logger.debug("Image analysis failed: %s", e)

    # Step 5: Legal context (if needed — deterministic trigger, LLM query)
    legal_context = ""
    if ctx.get("needs_legal_context", {}).get("should_consult_reka"):
        try:
            legal_context = await _get_legal_context(msg.body_text)
            if legal_context:
                result["legal_query"] = True
        except Exception as e:
            logger.debug("Legal context failed: %s", e)

    # Step 6: Generate draft (LLM call — the main one)
    from ..main import _build_greeting
    import json as _json

    # Build the draft via internal function call (not HTTP)
    draft_result = await _generate_draft_internal(
        email_text=msg.body_text,
        email_subject=msg.subject,
        sender_name=msg.sender,
        sender_email=msg.sender_email,
        oetp_ids=msg.oetp_ids,
        ctx=ctx,
        image_context=image_context,
        legal_context=legal_context,
    )

    if not draft_result or not draft_result.get("body_html"):
        result["status"] = "error"
        result["error"] = "empty_draft"
        return result

    confidence = draft_result.get("confidence", "medium")

    # Step 6b: Identity guard — check OETP IDs in draft match the email
    import re as _re
    draft_oetp_ids = set(_re.findall(r"OETP-\d{4}-\d+", draft_result.get("body_html", "")))
    email_oetp_ids = set(msg.oetp_ids) if msg.oetp_ids else set()
    if draft_oetp_ids and email_oetp_ids and not (draft_oetp_ids & email_oetp_ids):
        # Draft mentions different OETP IDs than the email — identity confusion!
        logger.warning("Identity mismatch: draft has %s but email has %s", draft_oetp_ids, email_oetp_ids)
        confidence = "low"

    result["confidence"] = confidence

    # Step 7: OETP DB lookup tracking
    if draft_result.get("radix_data"):
        result["oetp_lookup"] = True

    # Step 8: Save draft to Outlook (clean — NO debug block)
    try:
        draft_saved = await drafts.create_reply_draft(
            mailbox=msg.mailbox,
            reply_to_message_id=msg.id,
            body_html=draft_result["body_html"],
            confidence=confidence,
        )
        result["status"] = "draft_created"
        result["draft_id"] = draft_saved.draft_id if draft_saved else None
    except Exception as e:
        logger.error("Draft save failed for %s: %s", msg.subject[:40], e)
        result["status"] = "error"
        result["error"] = f"draft_save: {e}"

    # Step 9: Person tracking + reasoning trace (non-blocking)
    try:
        import asyncpg
        conn = await asyncpg.connect(
            os.environ.get("HANNA_PG_DSN", "postgresql://klara:klara_docs_2026@hanna-db:5432/hanna_oetp")
        )
        try:
            from ..reasoning.person_tracker import process_email_entities
            from ..reasoning.traces import create_trace
            from ..rag.embeddings import embed_query

            await process_email_entities(
                conn=conn,
                sender_name=msg.sender,
                sender_email=msg.sender_email,
                oetp_ids=msg.oetp_ids,
                email_subject=msg.subject,
                category=ctx.get("category", ""),
            )

            query_emb = await embed_query(msg.body_text[:500])
            await create_trace(
                conn=conn,
                query_text=msg.body_text[:2000],
                category=ctx.get("category", ""),
                email_message_id=msg.id,
                sender_name=msg.sender,
                sender_email=msg.sender_email,
                confidence=confidence,
                draft_text=draft_result.get("body_html", "")[:2000],
                query_embedding=query_emb if query_emb else None,
                top_chunks=[
                    {"id": r.get("id", ""), "score": r.get("score", 0), "chunk_type": r.get("chunk_type", "")}
                    for r in ctx.get("rag_results", [])[:5]
                ],
            )
        finally:
            await conn.close()
    except Exception as e:
        logger.debug("Person/trace tracking failed (non-blocking): %s", e)

    return result


async def _generate_draft_internal(
    email_text: str,
    email_subject: str,
    sender_name: str,
    sender_email: str,
    oetp_ids: list[str],
    ctx: dict,
    image_context: str = "",
    legal_context: str = "",
) -> dict:
    """Generate draft using internal logic (same as /draft/generate but no HTTP)."""
    # Use the Hanna API internally
    async with httpx.AsyncClient(timeout=120) as client:
        payload = {
            "email_text": email_text[:3000],
            "email_subject": email_subject,
            "sender_name": sender_name,
            "sender_email": sender_email,
            "oetp_ids": oetp_ids,
            "top_k": 5,
            "max_context_chunks": 3,
        }

        # Add image context to email text if available
        if image_context:
            payload["email_text"] += f"\n\n[CSATOLMÁNY ELEMZÉS]\n{image_context}"

        # Add legal context
        if legal_context:
            payload["email_text"] += f"\n\n[JOGI KONTEXTUS]\n{legal_context}"

        resp = await client.post("http://localhost:8000/draft/generate", json=payload)
        if resp.status_code == 200:
            return resp.json()
        logger.error("Draft generate failed: %d", resp.status_code)
        return {}


async def _analyze_images(mailbox: str, message_id: str) -> str:
    """Analyze image attachments using GPT-4o Vision."""
    try:
        results = await analyze_email_attachments(mailbox, message_id)
        if not results:
            return ""
        descriptions = []
        for r in results:
            desc = r.get("description") or r.get("analysis") or r.get("text", "")
            if desc:
                descriptions.append(desc)
        return "\n".join(descriptions) if descriptions else ""
    except Exception:
        return ""


async def _get_legal_context(question: str) -> str:
    """Query Jogszabály RAG for relevant legal context."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get("http://host.docker.internal:8103/search", params={
                "q": question[:500],
                "top_k": 3,
            })
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                if results:
                    lines = ["Releváns jogszabályi háttér:"]
                    for r in results[:3]:
                        text = r.get("text", r.get("content", ""))[:300]
                        source = r.get("source", r.get("metadata", {}).get("source", ""))
                        lines.append(f"- ({source}) {text}")
                    return "\n".join(lines)
            return ""
    except Exception:
        return ""


async def _send_discord_summary(stats: dict) -> None:
    """Send processing summary to Discord webhook."""
    if not settings.discord_webhook_url:
        return

    emoji_map = {"high": "🟢", "medium": "🟡", "low": "🔴"}
    msg = (
        f"📋 **Hanna OETP — Email feldolgozás**\n"
        f"📬 Új: {stats['emails_polled']} | "
        f"✅ Draft: {stats['drafts_created']} | "
        f"🟢 {stats['high_confidence']} | "
        f"🟡 {stats['medium_confidence']} | "
        f"🔴 {stats['low_confidence']} | "
        f"⏭️ Skip: {stats['skipped']} | "
        f"❌ Error: {stats['errors']}"
    )

    if stats.get("legal_queries"):
        msg += f"\n⚖️ Jogi lekérdezés: {stats['legal_queries']}"
    if stats.get("oetp_db_lookups"):
        msg += f"\n🗄️ OETP DB: {stats['oetp_db_lookups']}"

    async with httpx.AsyncClient(timeout=10) as client:
        await client.post(settings.discord_webhook_url, json={"content": msg})
