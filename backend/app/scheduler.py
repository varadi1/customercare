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
_last_run: dict = {"time": None, "status": None, "stats": None, "error": None}


def get_scheduler_status() -> dict:
    """Return scheduler status for health check."""
    return {
        "enabled": settings.auto_process_enabled,
        "running": _task is not None and not _task.done(),
        "last_run": _last_run,
    }


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
            stats = await _run_email_processing()
            _last_run["time"] = now.isoformat()
            _last_run["status"] = "ok"
            _last_run["stats"] = stats

            # Discord notification after each run
            await _notify_discord_run(stats)

            # At 05:00 UTC: feedback check
            if hour == 5:
                await _run_feedback_check()

            # At 06:00 UTC: style patterns refresh
            if hour == 6:
                await _run_style_refresh()

            # Monday 06:00 UTC: weekly report + authority refresh
            if weekday == 0 and hour == 6:
                await _run_weekly_report()

        except Exception as e:
            logger.error("Scheduler error: %s", e)
            _last_run["time"] = now.isoformat()
            _last_run["status"] = "error"
            _last_run["error"] = str(e)
            await _notify_discord_error(str(e))

        # Sleep 2 hours
        await asyncio.sleep(7200)


async def _run_email_processing() -> dict:
    """Process new emails — poll + filter + draft."""
    logger.info("Scheduler: starting email processing...")
    from .email.processor import process_new_emails
    stats = await process_new_emails(hours=4)
    logger.info(
        "Scheduler: processed %d emails, %d drafts, %d skipped, %d errors",
        stats.get("emails_polled", 0),
        stats.get("drafts_created", 0),
        stats.get("skipped", 0),
        stats.get("errors", 0),
    )
    return stats


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


async def _run_style_refresh():
    """Daily style patterns refresh from colleague sent emails."""
    logger.info("Scheduler: refreshing style patterns...")
    try:
        from .email.style_learner import analyze_sent_items
        mailboxes = [m.strip() for m in settings.shared_mailboxes.split(",") if m.strip()]
        for mb in mailboxes:
            result = await analyze_sent_items(mailbox=mb, hours=168)  # 1 week
            logger.info("Scheduler: style refresh for %s: %d emails analyzed",
                       mb, result.get("total_analyzed", 0))
    except Exception as e:
        logger.error("Scheduler: style refresh failed: %s", e)


async def _notify_discord_run(stats: dict) -> None:
    """Send run summary to Discord (only if drafts created or errors)."""
    if not settings.discord_webhook_url:
        return

    drafts = stats.get("drafts_created", 0)
    errors = stats.get("errors", 0)
    polled = stats.get("emails_polled", 0)

    # Only notify if something happened
    if drafts == 0 and errors == 0 and polled == 0:
        return

    msg = (
        f"📋 **Hanna** | "
        f"📬 {polled} email | "
        f"✅ {drafts} draft | "
        f"🟢 {stats.get('high_confidence', 0)} "
        f"🟡 {stats.get('medium_confidence', 0)} "
        f"🔴 {stats.get('low_confidence', 0)} | "
        f"⏭️ {stats.get('skipped', 0)} skip"
    )
    if errors > 0:
        msg += f" | ❌ **{errors} ERROR**"

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(settings.discord_webhook_url, json={"content": msg})
    except Exception as e:
        logger.debug("Discord notify failed: %s", e)


async def _notify_discord_error(error: str) -> None:
    """Send error alert to Discord."""
    if not settings.discord_webhook_url:
        return
    msg = f"🚨 **Hanna HIBA** | Scheduler error: {error[:200]}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(settings.discord_webhook_url, json={"content": msg})
    except Exception:
        pass


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
