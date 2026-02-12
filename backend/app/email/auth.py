"""Microsoft Graph API authentication via MSAL."""

from __future__ import annotations

import json
import os
from pathlib import Path

from msal import ConfidentialClientApplication
from ..config import settings

TOKEN_CACHE_PATH = Path("/app/data/token_cache.json")
GRAPH_SCOPES = [
    "https://graph.microsoft.com/Mail.Read.Shared",
    "https://graph.microsoft.com/Mail.ReadWrite.Shared",
]

_app: ConfidentialClientApplication | None = None
_token_cache: dict | None = None


def _get_msal_app() -> ConfidentialClientApplication:
    """Get or create MSAL confidential client app."""
    global _app
    if _app is None:
        _app = ConfidentialClientApplication(
            client_id=settings.graph_client_id,
            client_credential=settings.graph_client_secret,
            authority=f"https://login.microsoftonline.com/{settings.graph_tenant_id}",
        )
    return _app


def get_access_token() -> str | None:
    """Get a valid access token, using cached refresh token if available.

    Returns the access token string or None if auth fails.
    """
    app = _get_msal_app()

    # Try client credentials flow (app-only, no user context)
    # This requires Application permissions, not Delegated
    result = app.acquire_token_for_client(
        scopes=["https://graph.microsoft.com/.default"]
    )

    if result and "access_token" in result:
        return result["access_token"]

    # Log error for debugging
    if result:
        print(f"[auth] Token acquisition failed: {result.get('error_description', 'unknown error')}")

    return None


def get_auth_headers() -> dict:
    """Get Authorization headers for Graph API calls."""
    token = get_access_token()
    if not token:
        raise RuntimeError("Failed to acquire Graph API access token")
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
