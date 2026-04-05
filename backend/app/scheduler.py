"""
Built-in scheduler for autonomous email processing.

Runs inside the FastAPI process — no external cron needed.
Feature flag: AUTO_PROCESS_ENABLED (default: false)

Schedule:
  - Every 2h: process new emails (poll + filter + draft)
  - Every 24h (05:00): feedback check + reasoning trace sync
  - Weekly (Mon 06:00): knowledge gap report + authority refresh
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from .config import settings

logger = logging.getLogger(__name__)

_task: asyncio.Task | None = None


def start_scheduler():
    """Start the background scheduler. Call from FastAPI startup."""
    global _task
    if not settings.auto_process_enabled:
        logger.info("Autonomous processing disabled (AUTO_PROCESS_ENABLED=false)")
        return

    _task = asyncio.create_task(_scheduler_loop())
    logger.info("Scheduler started: email processing every 2h")


def stop_scheduler():
    """Stop the scheduler. Call from FastAPI shutdown."""
    global _task
    if _task:
        _task.cancel()
        _task = None


async def _scheduler_loop():
    """Main scheduler loop."""
    logger.info("Scheduler loop starting...")

    # Wait 30s after startup to let services stabilize
    await asyncio.sleep(30)

    while True:
        now = datetime.now(timezone.utc)
        hour = now.hour
        weekday = now.weekday()  # 0=Monday

        try:
            # Every iteration: process emails (2h polling window)
            await _run_email_processing()

            # At 05:00 UTC: feedback check
            if hour == 5:
                await _run_feedback_check()

            # Monday 06:00 UTC: weekly report + authority refresh
            if weekday == 0 and hour == 6:
                await _run_weekly_report()

        except Exception as e:
            logger.error("Scheduler error: %s", e)

        # Sleep 2 hours
        await asyncio.sleep(7200)


async def _run_email_processing():
    """Process new emails — poll + filter + draft."""
    logger.info("Scheduler: starting email processing...")
    try:
        from .email.processor import process_new_emails
        stats = await process_new_emails(hours=4)
        logger.info(
            "Scheduler: processed %d emails, %d drafts, %d skipped",
            stats.get("emails_polled", 0),
            stats.get("drafts_created", 0),
            stats.get("skipped", 0),
        )
    except Exception as e:
        logger.error("Scheduler: email processing failed: %s", e)


async def _run_feedback_check():
    """Daily feedback check — compare drafts vs sent."""
    logger.info("Scheduler: running feedback check...")
    try:
        from .email import feedback
        mailboxes = [m.strip() for m in settings.shared_mailboxes.split(",") if m.strip()]
        for mb in mailboxes:
            result = await feedback.check_feedback(mailbox=mb, hours=48)
            logger.info("Scheduler: feedback for %s: %s", mb, result.get("status"))
    except Exception as e:
        logger.error("Scheduler: feedback check failed: %s", e)


async def _run_weekly_report():
    """Weekly knowledge gap report + authority refresh."""
    logger.info("Scheduler: running weekly report...")
    try:
        from .reasoning.knowledge_gaps import generate_gap_report, format_obsidian_report
        from .reasoning.authority_learner import refresh_adjustments_cache
        from pathlib import Path

        # Gap report
        report = await generate_gap_report(days=7)
        if report.get("status") != "no_data":
            md = format_obsidian_report(report)
            date_str = datetime.now(timezone.utc).strftime("%y%m%d")
            reports_dir = Path("/app/data/reports")
            reports_dir.mkdir(exist_ok=True)
            (reports_dir / f"{date_str}-knowledge-gaps.md").write_text(md, encoding="utf-8")
            logger.info("Scheduler: gap report saved")

        # Authority refresh
        adj = await refresh_adjustments_cache(days=30)
        logger.info("Scheduler: authority refreshed (%d categories)", len(adj))

    except Exception as e:
        logger.error("Scheduler: weekly report failed: %s", e)
