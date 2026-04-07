"""
OETP pályázati adatbázis client — fetches applicant data from MySQL.

Connection: MySQL readonly (tarolo_neuzrt_hu_db)
Config via .env: OETP_DB_HOST, OETP_DB_PORT, OETP_DB_USER, OETP_DB_PASSWORD, OETP_DB_NAME
Feature flag: OETP_DB_ENABLED (default: false)

All functions return None/empty if DB is not configured (non-blocking).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..config import settings

logger = logging.getLogger(__name__)


def _get_connection():
    """Get MySQL connection (sync — pymysql)."""
    if not settings.oetp_db_enabled:
        return None
    try:
        import pymysql
        return pymysql.connect(
            host=settings.oetp_db_host,
            port=settings.oetp_db_port,
            user=settings.oetp_db_user,
            password=settings.oetp_db_password,
            database=settings.oetp_db_name,
            connect_timeout=5,
            read_timeout=10,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
        )
    except Exception as e:
        logger.warning("OETP DB connection failed: %s", e)
        return None


def get_application(oetp_id: str) -> Optional[dict[str, Any]]:
    """Fetch application data by OETP pályázati kódszám.

    Returns None if not found or DB not configured.
    """
    conn = _get_connection()
    if not conn:
        return None

    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT
                    c.palyazati_kodszam,
                    c.status,
                    c.step,
                    c.palyazo_neve,
                    c.palyazo_email,
                    c.meghatalmazott_neve,
                    c.megbizott_email,
                    c.palyazo_telepules,
                    c.palyazo_megye,
                    c.megvalositasi_telepules,
                    c.megvalositasi_megye,
                    c.celterulet,
                    c.igenyelt_tamogatas,
                    c.jovahagyott_tamogatas,
                    c.megitelt_tamogatas,
                    c.nyilatkozat,
                    c.veglegesites,
                    c.szaldo_status,
                    c.pod,
                    c.dealer_id,
                    d.name AS dealer_name
                FROM competitions c
                LEFT JOIN dealers d ON d.id = c.dealer_id
                WHERE c.palyazati_kodszam = %s
                  AND c.deleted = 0
                LIMIT 1
            """, (oetp_id,))
            row = cursor.fetchone()
            return row
    except Exception as e:
        logger.warning("OETP DB query failed for %s: %s", oetp_id, e)
        return None
    finally:
        conn.close()


def get_applications_by_email(email: str) -> list[dict[str, Any]]:
    """Find applications linked to an email address."""
    conn = _get_connection()
    if not conn:
        return []

    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT palyazati_kodszam, status, palyazo_neve,
                       igenyelt_tamogatas, celterulet
                FROM competitions
                WHERE (palyazo_email = %s OR megbizott_email = %s)
                  AND deleted = 0
                ORDER BY nyilatkozat DESC
                LIMIT 10
            """, (email, email))
            return cursor.fetchall()
    except Exception as e:
        logger.warning("OETP DB email search failed: %s", e)
        return []
    finally:
        conn.close()


# Status code mapping
STATUS_MAP = {
    0: "Piszkozat",
    1: "Benyújtva",
    2: "Formai ellenőrzés alatt",
    3: "Értékelés alatt",
    4: "Hiánypótlás",
    5: "Döntésre vár",
    6: "Nyertes",
    7: "Nem nyertes",
    8: "Elutasított",
    9: "Visszavont",
    10: "Szerződéskötés",
    11: "Megvalósítás",
    12: "Elszámolás",
    13: "Lezárt",
}

CELTERULET_MAP = {
    1: "1. célterület: Új napelem + tároló",
    2: "2. célterület: Meglévő rendszer bővítése tárolóval",
}


def format_applicant_context(data: dict[str, Any], sender_email: str = "") -> str:
    """Format applicant data as LLM-readable context block.

    If sender_email matches the applicant or representative email,
    marks the data as "[AZONOSÍTOTT PÁLYÁZÓ]" — eligible to see all data
    including winning status. Otherwise marks as "[NEM AZONOSÍTOTT]".
    """
    if not data:
        return ""

    status_code = data.get("status")
    status_text = STATUS_MAP.get(status_code, f"Ismeretlen ({status_code})")
    celterulet_text = CELTERULET_MAP.get(data.get("celterulet"), "")

    # Identity check: does sender email match applicant or representative?
    sender_lower = (sender_email or "").strip().lower()
    applicant_email = (data.get("palyazo_email") or "").strip().lower()
    representative_email = (data.get("megbizott_email") or "").strip().lower()

    is_identified = sender_lower and (
        sender_lower == applicant_email or sender_lower == representative_email
    )

    # Result status — only share if identified AND already notified (status >= 6)
    RESULT_STATUSES = {6, 7, 8}  # Nyertes, Nem nyertes, Elutasított
    can_share_result = is_identified and status_code in RESULT_STATUSES

    if is_identified:
        id_tag = "[AZONOSÍTOTT PÁLYÁZÓ]"
    else:
        id_tag = "[NEM AZONOSÍTOTT — eredmény nem közölhető]"

    lines = [f"[PÁLYÁZÓI ADATOK — OETP adatbázisból] {id_tag}"]
    lines.append(f"- Pályázati kódszám: {data.get('palyazati_kodszam', '?')}")
    lines.append(f"- Pályázó neve: {data.get('palyazo_neve', '?')}")

    # For result statuses of unidentified senders, mask the winning status
    if status_code in RESULT_STATUSES and not can_share_result:
        lines.append(f"- Státusz: Elbírálás megtörtént (eredmény nem közölhető)")
    else:
        lines.append(f"- Státusz: {status_text}")

    if data.get("meghatalmazott_neve"):
        lines.append(f"- Meghatalmazott: {data['meghatalmazott_neve']}")
    if celterulet_text:
        lines.append(f"- Célterület: {celterulet_text}")
    if data.get("megvalositasi_telepules"):
        lines.append(f"- Megvalósítás helye: {data['megvalositasi_telepules']}, {data.get('megvalositasi_megye', '')}")
    if data.get("igenyelt_tamogatas"):
        lines.append(f"- Igényelt támogatás: {data['igenyelt_tamogatas']:,} Ft".replace(",", " "))
    if data.get("jovahagyott_tamogatas"):
        lines.append(f"- Jóváhagyott támogatás: {data['jovahagyott_tamogatas']:,} Ft".replace(",", " "))
    if data.get("dealer_name"):
        lines.append(f"- Kivitelező: {data['dealer_name']}")
    if data.get("pod"):
        lines.append(f"- POD szám: {data['pod']}")

    return "\n".join(lines)


async def enrich_draft_context(
    oetp_ids: list[str],
    sender_email: str = "",
) -> dict[str, Any]:
    """Fetch all relevant OETP data for draft generation."""
    if not settings.oetp_db_enabled:
        return {}

    result = {"applications": [], "sender_applications": []}

    for oetp_id in oetp_ids[:5]:
        app_data = get_application(oetp_id)
        if app_data:
            result["applications"].append(app_data)

    if not oetp_ids and sender_email:
        apps = get_applications_by_email(sender_email)
        result["sender_applications"] = apps

    return result if result["applications"] or result["sender_applications"] else {}
