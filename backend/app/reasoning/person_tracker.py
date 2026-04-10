"""
Person & Organization tracker — builds KG entities from email interactions.

Creates person entities from email senders, organization entities from
email domains, and links them to OETP applications via OETP-ID extraction.

Entity types: person, organization, application
Relation types: ASKED_ABOUT, APPLICANT_OF, REPRESENTATIVE_OF, BELONGS_TO
"""
from __future__ import annotations

import logging
import re
from typing import Optional

import asyncpg

logger = logging.getLogger(__name__)


async def register_sender(
    conn: asyncpg.Connection,
    sender_name: str,
    sender_email: str,
) -> int:
    """Register an email sender as a person entity in kg_entities.

    Upserts: if email already exists in aliases, returns existing ID.
    Returns the entity ID (integer).
    """
    existing = await conn.fetchrow(
        """
        SELECT id, name, aliases
        FROM kg_entities
        WHERE type = 'person'
          AND $1 = ANY(aliases)
        """,
        sender_email,
    )

    if existing:
        entity_id = existing["id"]
        logger.debug("Person entity exists: %s (id=%d)", sender_name, entity_id)
        return entity_id

    row = await conn.fetchrow(
        """
        INSERT INTO kg_entities (name, type, aliases, metadata)
        VALUES ($1, 'person', ARRAY[$2]::text[], '{"source": "email"}'::jsonb)
        RETURNING id
        """,
        sender_name,
        sender_email,
    )

    entity_id = row["id"]
    logger.info("Registered person entity: %s <%s> → id=%d", sender_name, sender_email, entity_id)
    return entity_id


async def register_organization(
    conn: asyncpg.Connection,
    org_name: str,
    domain: Optional[str] = None,
) -> int:
    """Register an organization entity. Dedup by name (case-insensitive)."""
    existing = await conn.fetchrow(
        """
        SELECT id FROM kg_entities
        WHERE type = 'organization'
          AND LOWER(name) = LOWER($1)
        """,
        org_name,
    )

    if existing:
        return existing["id"]

    aliases = [domain] if domain else []
    row = await conn.fetchrow(
        """
        INSERT INTO kg_entities (name, type, aliases, metadata)
        VALUES ($1, 'organization', $2::text[], '{"source": "email"}'::jsonb)
        RETURNING id
        """,
        org_name,
        aliases,
    )

    entity_id = row["id"]
    logger.info("Registered organization: %s → id=%d", org_name, entity_id)
    return entity_id


async def get_or_create_application(
    conn: asyncpg.Connection,
    app_id: str,
) -> int:
    """Get or create an application entity from OETP-ID."""
    existing = await conn.fetchrow(
        """
        SELECT id FROM kg_entities
        WHERE type = 'application'
          AND $1 = ANY(aliases)
        """,
        app_id,
    )

    if existing:
        return existing["id"]

    row = await conn.fetchrow(
        """
        INSERT INTO kg_entities (name, type, aliases, metadata)
        VALUES ($1, 'application', ARRAY[$1]::text[], '{"source": "email", "program": "OETP"}'::jsonb)
        RETURNING id
        """,
        app_id,
    )

    entity_id = row["id"]
    logger.info("Created application entity: %s → id=%d", app_id, entity_id)
    return entity_id


async def link_entities(
    conn: asyncpg.Connection,
    source_id: int,
    target_id: int,
    relation_type: str,
    metadata: Optional[dict] = None,
) -> None:
    """Create a relation between two entities (idempotent)."""
    existing = await conn.fetchval(
        """
        SELECT COUNT(*) FROM kg_relations
        WHERE source_id = $1 AND target_id = $2 AND relation_type = $3
        """,
        source_id, target_id, relation_type,
    )

    if existing > 0:
        return

    import json
    meta_json = json.dumps(metadata or {}, ensure_ascii=False)

    await conn.execute(
        """
        INSERT INTO kg_relations (source_id, target_id, relation_type, weight, metadata)
        VALUES ($1, $2, $3, 1.0, $4::jsonb)
        """,
        source_id, target_id, relation_type, meta_json,
    )
    logger.debug("Linked entity %d →[%s]→ %d", source_id, relation_type, target_id)


async def process_email_entities(
    conn: asyncpg.Connection,
    sender_name: str,
    sender_email: str,
    app_ids: list[str],
    email_subject: str = "",
    category: str = "",
) -> dict:
    """Full entity processing for an incoming email.

    1. Register sender as person
    2. Extract organization from email domain
    3. Link person to OETP applications
    4. Track what they asked about

    Returns summary of created/linked entities.
    """
    result = {"person_id": None, "org_id": None, "application_ids": [], "relations_created": 0}

    # 1. Person entity
    person_id = await register_sender(conn, sender_name, sender_email)
    result["person_id"] = person_id

    # 2. Organization from domain (skip common providers)
    domain = sender_email.split("@")[-1] if "@" in sender_email else None
    skip_domains = {
        "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "freemail.hu",
        "citromail.hu", "t-online.hu", "indamail.hu", "vipmail.hu",
    }
    if domain and domain.lower() not in skip_domains:
        org_name = _domain_to_org_name(domain)
        org_id = await register_organization(conn, org_name, domain)
        result["org_id"] = org_id

        # Link person → org
        await link_entities(conn, person_id, org_id, "BELONGS_TO",
                          {"detected_from": "email_domain"})
        result["relations_created"] += 1

    # 3. OETP application entities + links
    for app_id in app_ids:
        app_id = await get_or_create_application(conn, app_id)
        result["application_ids"].append(app_id)

        # Person ASKED_ABOUT application
        await link_entities(conn, person_id, app_id, "ASKED_ABOUT",
                          {"subject": email_subject[:100], "category": category})
        result["relations_created"] += 1

    return result


def extract_app_ids(text: str) -> list[str]:
    """Extract application IDs from text using program.yaml pattern."""
    from .radix_client import extract_app_ids as _extract
    return _extract(text)


def _domain_to_org_name(domain: str) -> str:
    """Convert email domain to readable organization name."""
    # Remove common TLDs
    parts = domain.split(".")
    if len(parts) >= 2:
        name = parts[-2]  # e.g., "solarpro" from "solarpro.hu"
    else:
        name = parts[0]
    return name.replace("-", " ").title()
