"""
Generic program database client — fetches applicant/project data.

Schema-driven: all table names, column mappings, status maps, and display
labels come from the `database:` section in program.yaml.

Supports two drivers:
  - mysql:  pymysql  (e.g. OETP MySQL)
  - mssql:  pymssql  (e.g. Miniradix Azure SQL)

Config via .env: PROGRAM_DB_DRIVER, PROGRAM_DB_HOST, PROGRAM_DB_PORT,
  PROGRAM_DB_USER, PROGRAM_DB_PASSWORD, PROGRAM_DB_NAME
Feature flag: PROGRAM_DB_ENABLED (default: false)

All functions return None/empty if DB is not configured (non-blocking).
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

from ..config import settings, get_db_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def _get_connection():
    """Get database connection based on configured driver."""
    if not settings.program_db_enabled:
        return None

    driver = settings.program_db_driver.lower()

    try:
        if driver == "mssql":
            import pymssql
            return pymssql.connect(
                server=settings.program_db_host,
                port=settings.program_db_port,
                user=settings.program_db_user,
                password=settings.program_db_password,
                database=settings.program_db_name,
                login_timeout=5,
                timeout=10,
                as_dict=True,
                tds_version="7.3",
            )
        else:  # mysql (default)
            import pymysql
            return pymysql.connect(
                host=settings.program_db_host,
                port=settings.program_db_port,
                user=settings.program_db_user,
                password=settings.program_db_password,
                database=settings.program_db_name,
                connect_timeout=5,
                read_timeout=10,
                charset="utf8mb4",
                cursorclass=pymysql.cursors.DictCursor,
            )
    except Exception as e:
        logger.warning("Program DB connection failed (%s): %s", driver, e)
        return None


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------

def _build_select(db_cfg: dict) -> tuple[str, list[str]]:
    """Build a SELECT statement from the database config.

    Returns (sql_prefix, logical_column_names) where sql_prefix is everything
    up to but not including WHERE.

    Example output:
      "SELECT c.palyazati_kodszam, c.status, ... FROM competitions c
       LEFT JOIN dealers d ON c.dealer_id = d.id"
    """
    columns = db_cfg.get("columns", {})
    table = db_cfg["primary_table"]
    join_cfg = db_cfg.get("join")

    # Build SELECT columns: c.col AS logical_name
    select_parts = []
    logical_names = []
    for logical, db_col in columns.items():
        select_parts.append(f"c.{db_col} AS {logical}")
        logical_names.append(logical)

    # Add join columns
    if join_cfg and join_cfg.get("columns"):
        for logical, db_col in join_cfg["columns"].items():
            # db_col may already have alias prefix (e.g. "d.name")
            if "." in db_col:
                select_parts.append(f"{db_col} AS {logical}")
            else:
                select_parts.append(f"d.{db_col} AS {logical}")
            logical_names.append(logical)

    select_clause = ", ".join(select_parts)
    from_clause = f"FROM {table} c"

    # JOIN
    if join_cfg and join_cfg.get("table"):
        join_table = join_cfg["table"]
        join_on = join_cfg.get("on", "")
        from_clause += f" LEFT JOIN {join_table} d ON {join_on}"

    return f"SELECT {select_clause} {from_clause}", logical_names


def _where_clause(db_cfg: dict, extra: str = "") -> str:
    """Build WHERE clause from soft_delete + optional extra conditions."""
    parts = []
    soft_delete = db_cfg.get("soft_delete")
    if soft_delete:
        parts.append(soft_delete)

    join_filter = (db_cfg.get("join") or {}).get("filter")
    if join_filter:
        parts.append(join_filter)

    if extra:
        parts.append(extra)

    return " WHERE " + " AND ".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def get_application(app_id: str) -> Optional[dict[str, Any]]:
    """Fetch application data by ID. Returns None if not found."""
    db_cfg = get_db_config()
    if not db_cfg or not db_cfg.get("columns", {}).get("app_id"):
        return None

    conn = _get_connection()
    if not conn:
        return None

    app_id_col = db_cfg["columns"]["app_id"]
    select, _ = _build_select(db_cfg)
    where = _where_clause(db_cfg, f"c.{app_id_col} = %s")

    try:
        with conn.cursor() as cursor:
            cursor.execute(f"{select}{where}", (app_id,))
            return cursor.fetchone()
    except Exception as e:
        logger.warning("Program DB query failed for %s: %s", app_id, e)
        return None
    finally:
        conn.close()


def get_applications_by_email(email: str) -> list[dict[str, Any]]:
    """Find applications linked to an email address."""
    db_cfg = get_db_config()
    if not db_cfg:
        return []

    identity_fields = db_cfg.get("identity_fields", [])
    if not identity_fields:
        return []

    columns = db_cfg.get("columns", {})
    email_cols = [columns[f] for f in identity_fields if f in columns]
    if not email_cols:
        return []

    conn = _get_connection()
    if not conn:
        return []

    # Build OR condition for all email columns
    or_parts = " OR ".join(f"c.{col} = %s" for col in email_cols)
    email_extra = f"({or_parts})"

    select, _ = _build_select(db_cfg)
    where = _where_clause(db_cfg, email_extra)

    # Order by declaration date if available
    order = ""
    if "declaration_date" in columns:
        order = f" ORDER BY c.{columns['declaration_date']} DESC"

    try:
        with conn.cursor() as cursor:
            params = tuple(email for _ in email_cols)
            cursor.execute(f"{select}{where}{order}", params)
            return cursor.fetchall()
    except Exception as e:
        logger.warning("Program DB email search failed: %s", e)
        return []
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _resolve_status(db_cfg: dict, raw_value: Any) -> str:
    """Resolve status code/text using status_map. Falls back to raw value."""
    status_map = db_cfg.get("status_map", {})
    if status_map and raw_value in status_map:
        return status_map[raw_value]
    # Try int conversion (YAML may load keys as int)
    try:
        int_val = int(raw_value)
        if int_val in status_map:
            return status_map[int_val]
    except (TypeError, ValueError):
        pass
    return str(raw_value) if raw_value is not None else "Ismeretlen"


def _resolve_category(db_cfg: dict, logical_col: str, raw_value: Any) -> str:
    """Resolve a category value using category_maps."""
    cat_maps = db_cfg.get("category_maps", {})
    col_map = cat_maps.get(logical_col, {})
    if col_map and raw_value in col_map:
        return col_map[raw_value]
    try:
        int_val = int(raw_value)
        if int_val in col_map:
            return col_map[int_val]
    except (TypeError, ValueError):
        pass
    return str(raw_value) if raw_value is not None else ""


def format_applicant_context(data: dict[str, Any], sender_email: str = "") -> str:
    """Format applicant data as LLM-readable context block.

    Uses display_labels, status_map, category_maps, identity_fields, and
    result_statuses from program.yaml database config.
    """
    if not data:
        return ""

    db_cfg = get_db_config()
    if not db_cfg:
        return ""

    columns = db_cfg.get("columns", {})
    labels = db_cfg.get("display_labels", {})
    identity_fields = db_cfg.get("identity_fields", [])
    result_statuses = db_cfg.get("result_statuses", [])
    category_maps = db_cfg.get("category_maps", {})

    # --- Identity check ---
    sender_lower = (sender_email or "").strip().lower()
    is_identified = False
    if sender_lower and identity_fields:
        for field in identity_fields:
            val = (data.get(field) or "").strip().lower()
            if val and sender_lower == val:
                is_identified = True
                break

    # --- Status ---
    raw_status = data.get("status")
    status_text = _resolve_status(db_cfg, raw_status)

    # Result masking: hide outcome for unidentified senders
    can_share_result = is_identified and raw_status in result_statuses
    try:
        can_share_result = can_share_result or (is_identified and int(raw_status) in result_statuses)
    except (TypeError, ValueError):
        pass

    if identity_fields:
        id_tag = "[AZONOSÍTOTT PÁLYÁZÓ]" if is_identified else "[NEM AZONOSÍTOTT — eredmény nem közölhető]"
    else:
        id_tag = ""

    # --- Build output lines ---
    program_cfg = get_db_config()
    header = f"[PÁLYÁZÓI ADATOK — program adatbázisból] {id_tag}".strip()
    lines = [header]

    # Always show app_id and applicant_name first, then status
    for key in ["app_id", "applicant_name"]:
        val = data.get(key)
        if val and key in labels:
            lines.append(f"- {labels[key]}: {val}")

    # Status (with masking)
    if "status" in labels:
        if result_statuses and raw_status in result_statuses and not can_share_result:
            lines.append(f"- {labels['status']}: Elbírálás megtörtént (eredmény nem közölhető)")
        else:
            lines.append(f"- {labels['status']}: {status_text}")

    # Remaining fields in label order
    shown = {"app_id", "applicant_name", "status"}
    for key, label in labels.items():
        if key in shown:
            continue
        shown.add(key)

        val = data.get(key)
        if not val:
            continue

        # Category resolution
        if key in category_maps:
            val = _resolve_category(db_cfg, key, val)

        # Funding formatting (numeric → thousand separator + Ft)
        if "funding" in key:
            try:
                val = f"{int(val):,} Ft".replace(",", " ")
            except (TypeError, ValueError):
                pass

        # Location + region combo
        if key == "location" and data.get("region"):
            val = f"{val}, {data['region']}"

        lines.append(f"- {label}: {val}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# App ID extraction from email text
# ---------------------------------------------------------------------------

def extract_app_ids(text: str) -> list[str]:
    """Extract application IDs from text using the configured regex pattern."""
    db_cfg = get_db_config()
    pattern = db_cfg.get("app_id_pattern")
    if not pattern:
        return []
    return list(set(re.findall(pattern, text, re.IGNORECASE)))


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

async def enrich_draft_context(
    app_ids: list[str],
    sender_email: str = "",
) -> dict[str, Any]:
    """Fetch all relevant applicant data for draft generation."""
    if not settings.program_db_enabled:
        return {}

    db_cfg = get_db_config()
    if not db_cfg:
        return {}

    result = {"applications": [], "sender_applications": []}

    for app_id in app_ids[:5]:
        app_data = get_application(app_id)
        if app_data:
            result["applications"].append(app_data)

    if not app_ids and sender_email:
        apps = get_applications_by_email(sender_email)
        result["sender_applications"] = apps

    return result if result["applications"] or result["sender_applications"] else {}
