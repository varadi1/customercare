"""Obsidian ingestion — PostgreSQL + pgvector storage (replaces ChromaDB).

Hash-based incremental sync, chunking, pgvector embedding storage.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import asyncpg

from ..rag.chunker import chunk_markdown, chunk_text
from ..rag.embeddings import embed_texts

PG_DSN = "postgresql://klara:klara_docs_2026@host.docker.internal:5433/neu_docs"

_pool: asyncpg.Pool | None = None


async def _get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(PG_DSN, min_size=2, max_size=10)
    return _pool


def _generate_chunk_id(file_path: str, chunk_index: int, file_hash: str) -> str:
    raw = f"{file_path}::{chunk_index}::{file_hash[:8]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _calculate_file_hash(file_path: Path) -> str:
    try:
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return ""


def _extract_folder_type(file_path: Path, vault_path: Path) -> str:
    relative = file_path.relative_to(vault_path)
    parts = relative.parts
    if not parts:
        return "root"

    folder = parts[0].lower()
    if folder.startswith("!"):
        return "inbox"
    elif folder.startswith("1_"):
        return "projects"
    elif folder.startswith("2_"):
        return "areas"
    elif folder.startswith("3_"):
        return "resources"
    elif folder.startswith("4_"):
        return "archive"
    elif folder == "tags":
        return "tags"
    elif folder == "tasknotes":
        return "tasknotes"
    else:
        return "other"


def _chunk_file_content(content: str, file_path: str) -> list[str]:
    if len(content) < 1000:
        return [content]
    if file_path.endswith(".md"):
        chunks = chunk_markdown(content)
    else:
        chunks = chunk_text(content, chunk_size=1000, chunk_overlap=200)
    return chunks if chunks else [content]


def scan_vault_files(vault_path: str) -> list[Path]:
    vault = Path(vault_path)
    if not vault.exists():
        raise ValueError(f"Vault path does not exist: {vault_path}")

    files = []
    include_folders = {
        "!inbox", "1_projects", "2_areas", "3_resources",
        "4_archive", "Tags", "TaskNotes",
    }
    exclude_patterns = {".obsidian", ".trash"}
    exclude_extensions = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".mp4", ".mov"}

    for folder in vault.iterdir():
        if not folder.is_dir() or folder.name in exclude_patterns:
            continue
        if folder.name in include_folders:
            for file_path in folder.rglob("*"):
                if (
                    file_path.is_file()
                    and not any(exc in str(file_path) for exc in exclude_patterns)
                    and file_path.suffix.lower() not in exclude_extensions
                ):
                    files.append(file_path)

    return files


async def _get_stored_hashes() -> dict[str, str]:
    """Get stored file hashes from PostgreSQL."""
    pool = await _get_pool()
    rows = await pool.fetch("SELECT file_path, file_hash FROM obsidian_sync_state")
    return {r["file_path"]: r["file_hash"] for r in rows}


async def _update_hash(file_path: str, file_hash: str, chunk_count: int):
    pool = await _get_pool()
    await pool.execute(
        """INSERT INTO obsidian_sync_state (file_path, file_hash, chunk_count, last_synced)
           VALUES ($1, $2, $3, NOW())
           ON CONFLICT (file_path) DO UPDATE
           SET file_hash = $2, chunk_count = $3, last_synced = NOW()""",
        file_path, file_hash, chunk_count,
    )


async def _delete_file_chunks(file_path: str) -> int:
    pool = await _get_pool()
    result = await pool.execute("DELETE FROM obsidian_chunks WHERE file_path = $1", file_path)
    return int(result.split()[-1])  # "DELETE N"


async def _ingest_file(
    file_path: Path,
    vault_path: Path,
    file_hash: str,
) -> int:
    """Ingest a single file into PostgreSQL."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except Exception as e:
        print(f"[obsidian-pg] Failed to read {file_path}: {e}")
        return 0

    if not content:
        return 0

    chunks = _chunk_file_content(content, str(file_path))
    if not chunks:
        return 0

    # Truncate chunks to 6000 chars max (BGE-M3 limit)
    truncated = [c[:6000] for c in chunks]

    # Generate embeddings
    try:
        embeddings = embed_texts(truncated)
    except Exception as e:
        print(f"[obsidian-pg] Embedding failed for {file_path}: {e}")
        return 0

    relative_path = str(file_path.relative_to(vault_path))
    folder_type = _extract_folder_type(file_path, vault_path)
    modified_time = datetime.fromtimestamp(file_path.stat().st_mtime)

    pool = await _get_pool()

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_id = _generate_chunk_id(relative_path, i, file_hash)
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

        await pool.execute(
            """INSERT INTO obsidian_chunks
               (chunk_id, file_path, file_name, folder, chunk_index, content, embedding,
                file_hash, modified_at, metadata)
               VALUES ($1, $2, $3, $4, $5, $6, $7::vector, $8, $9, $10)
               ON CONFLICT (chunk_id) DO UPDATE
               SET content = EXCLUDED.content, embedding = EXCLUDED.embedding,
                   file_hash = EXCLUDED.file_hash, modified_at = EXCLUDED.modified_at,
                   updated_at = NOW()""",
            chunk_id, relative_path, file_path.name, folder_type, i,
            chunk, embedding_str, file_hash, modified_time, "{}",
        )

    return len(chunks)


async def ingest_vault(
    vault_path: str,
    force: bool = False,
    collection_name: str = "obsidian_notes",  # ignored, kept for compat
) -> dict[str, Any]:
    """Ingest entire Obsidian vault with incremental sync into PostgreSQL."""
    vault = Path(vault_path)

    stored_hashes = await _get_stored_hashes()
    files = scan_vault_files(vault_path)
    print(f"[obsidian-pg] Found {len(files)} files in vault")

    stats = {
        "total_files": len(files),
        "processed_files": 0,
        "skipped_files": 0,
        "total_chunks": 0,
        "errors": [],
        "changed_files": [],
        "new_files": [],
    }

    for file_path in files:
        relative_path = str(file_path.relative_to(vault))
        current_hash = _calculate_file_hash(file_path)

        if not current_hash:
            stats["errors"].append(f"Failed to hash: {relative_path}")
            continue

        if not force and stored_hashes.get(relative_path) == current_hash:
            stats["skipped_files"] += 1
            continue

        is_update = relative_path in stored_hashes

        if is_update:
            deleted = await _delete_file_chunks(relative_path)
            print(f"[obsidian-pg] Deleted {deleted} old chunks for {relative_path}")
            stats["changed_files"].append(relative_path)
        else:
            stats["new_files"].append(relative_path)

        try:
            chunk_count = await _ingest_file(file_path, vault, current_hash)
            await _update_hash(relative_path, current_hash, chunk_count)
            stats["processed_files"] += 1
            stats["total_chunks"] += chunk_count
            if stats["processed_files"] % 50 == 0:
                print(f"[obsidian-pg] Progress: {stats['processed_files']} files, {stats['total_chunks']} chunks")
        except Exception as e:
            error_msg = f"Failed to ingest {relative_path}: {str(e)}"
            stats["errors"].append(error_msg)
            print(f"[obsidian-pg] {error_msg}")

    # Log sync
    pool = await _get_pool()
    await pool.execute(
        """INSERT INTO obsidian_sync_log
           (finished_at, total_files, processed_files, skipped_files, total_chunks,
            new_files, changed_files, errors)
           VALUES (NOW(), $1, $2, $3, $4, $5, $6, $7)""",
        stats["total_files"], stats["processed_files"], stats["skipped_files"],
        stats["total_chunks"], stats["new_files"][:100], stats["changed_files"][:100],
        stats["errors"][:50],
    )

    print(f"[obsidian-pg] Ingest complete: {stats['processed_files']} files, "
          f"{stats['total_chunks']} chunks, {len(stats['errors'])} errors")

    return stats


async def get_last_sync_info() -> dict[str, Any]:
    """Get info about the last sync."""
    try:
        pool = await _get_pool()
        row = await pool.fetchrow(
            "SELECT * FROM obsidian_sync_log ORDER BY id DESC LIMIT 1"
        )
        if row:
            return {
                "timestamp": row["finished_at"].isoformat() if row["finished_at"] else None,
                "processed_files": row["processed_files"],
                "new_files": row["new_files"] or [],
                "changed_files": row["changed_files"] or [],
                "total_chunks": row["total_chunks"],
                "errors": row["errors"] or [],
            }
    except Exception:
        pass
    return {"timestamp": None, "processed_files": 0, "new_files": [], "changed_files": []}
