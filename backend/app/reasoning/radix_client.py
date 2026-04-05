"""
RADIX API client — fetches applicant data from the pályázati rendszer.

Enabled via .env:
  RADIX_API_URL=https://radix.neuzrt.hu/api/v1
  RADIX_API_KEY=your-api-key
  RADIX_ENABLED=true

All functions return None/empty if RADIX is not configured (non-blocking).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from ..config import settings

logger = logging.getLogger(__name__)

_TIMEOUT = 10  # seconds


async def is_available() -> bool:
    """Check if RADIX API is configured and reachable."""
    if not settings.radix_enabled or not settings.radix_api_url:
        return False
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(
                f"{settings.radix_api_url}/health",
                headers=_auth_headers(),
            )
            return resp.status_code == 200
    except Exception:
        return False


async def get_application(oetp_id: str) -> Optional[dict[str, Any]]:
    """Fetch application data by OETP-ID.

    Returns None if RADIX is not configured or application not found.
    """
    if not settings.radix_enabled or not settings.radix_api_url:
        return None

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                f"{settings.radix_api_url}/applications/{oetp_id}",
                headers=_auth_headers(),
            )
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 404:
                logger.debug("Application %s not found in RADIX", oetp_id)
                return None
            else:
                logger.warning("RADIX API error for %s: %d", oetp_id, resp.status_code)
                return None
    except Exception as e:
        logger.warning("RADIX API connection failed: %s", e)
        return None


async def get_applications_by_email(email: str) -> list[dict[str, Any]]:
    """Find applications linked to an email address."""
    if not settings.radix_enabled or not settings.radix_api_url:
        return []

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                f"{settings.radix_api_url}/applications",
                params={"email": email},
                headers=_auth_headers(),
            )
            if resp.status_code == 200:
                data = resp.json()
                return data if isinstance(data, list) else data.get("results", [])
            return []
    except Exception as e:
        logger.warning("RADIX API search failed: %s", e)
        return []


async def enrich_draft_context(
    oetp_ids: list[str],
    sender_email: str = "",
) -> dict[str, Any]:
    """Fetch all relevant RADIX data for draft generation.

    Returns structured dict for LLM context, or empty dict if unavailable.
    """
    if not settings.radix_enabled:
        return {}

    result = {"applications": [], "sender_applications": []}

    # Fetch by OETP-ID
    for oetp_id in oetp_ids[:5]:  # max 5 to avoid rate limiting
        app_data = await get_application(oetp_id)
        if app_data:
            result["applications"].append({
                "oetp_id": oetp_id,
                "status": app_data.get("status"),
                "applicant_name": app_data.get("applicant_name"),
                "submission_date": app_data.get("submission_date"),
                "target_area": app_data.get("target_area"),
                "decision": app_data.get("decision"),
                "phase": app_data.get("phase"),
                "representative": app_data.get("representative_name"),
            })

    # Fetch by sender email (if no OETP-ID provided)
    if not oetp_ids and sender_email:
        apps = await get_applications_by_email(sender_email)
        for app in apps[:3]:
            result["sender_applications"].append({
                "oetp_id": app.get("oetp_id"),
                "status": app.get("status"),
                "applicant_name": app.get("applicant_name"),
            })

    return result if result["applications"] or result["sender_applications"] else {}


def format_radix_context(data: dict[str, Any]) -> str:
    """Format RADIX data as LLM-readable context block."""
    if not data:
        return ""

    lines = ["[PÁLYÁZÓI ADATOK — RADIX rendszerből]"]

    for app in data.get("applications", []):
        lines.append(f"- Pályázat: {app.get('oetp_id', '?')}")
        if app.get("applicant_name"):
            lines.append(f"  Pályázó: {app['applicant_name']}")
        if app.get("status"):
            lines.append(f"  Státusz: {app['status']}")
        if app.get("decision"):
            lines.append(f"  Döntés: {app['decision']}")
        if app.get("phase"):
            lines.append(f"  Szakasz: {app['phase']}")
        if app.get("representative"):
            lines.append(f"  Meghatalmazott: {app['representative']}")

    for app in data.get("sender_applications", []):
        lines.append(f"- Pályázat: {app.get('oetp_id', '?')} ({app.get('status', '?')})")

    return "\n".join(lines)


def _auth_headers() -> dict[str, str]:
    """Build authentication headers for RADIX API."""
    headers = {"Accept": "application/json"}
    if settings.radix_api_key:
        headers["X-API-Key"] = settings.radix_api_key
    return headers
