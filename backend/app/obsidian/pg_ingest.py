"""Obsidian ingestion — PostgreSQL + pgvector storage (replaces ChromaDB).

Hash-based incremental sync, chunking, pgvector embedding storage.
"""

from __future__ import annotations

# --- AGENT ZERO MODOSITAS START (Unstructured) ---
try:
    from langchain_community.document_loaders import UnstructuredFileLoader
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
# --- AGENT ZERO MODOSITAS END ---

import asyncio
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import asyncpg

from ..rag.chunker import chunk_markdown, chunk_text
from ..rag.embeddings import embed_texts_ingest as embed_texts
from .enrichment import enrich_chunks_batch, ENRICHMENT_ENABLED
from . import kg_extract

PG_DSN = "postgresql://klara:klara_docs_2026@host.docker.internal:5433/obsidian_rag"

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
    if len(content) < 10:
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
    exclude_extensions = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".mp4", ".mov", ".csv", ".xlsx", ".xls"}
    # Skip files larger than 500KB to avoid blocking ingest with huge CSVs/logs
    MAX_FILE_SIZE = 512_000

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
                    try:
                        if file_path.stat().st_size > MAX_FILE_SIZE:
                            continue
                    except OSError:
                        continue
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

    # --- AGENT ZERO PATCH START (Smart Loader) ---
    def _read_file_sync():
        ext = file_path.suffix.lower()
        if UNSTRUCTURED_AVAILABLE and ext in ['.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls', '.jpg', '.jpeg', '.png', '.eml', '.msg']:
            print(f"[INGEST] Loading {file_path.name} with Unstructured...", flush=True)
            loader = UnstructuredFileLoader(str(file_path), mode="single", strategy="fast")
            docs = loader.load()
            return "\n\n".join([d.page_content for d in docs]).strip()
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read().strip()

    content = ""
    try:
        content = await asyncio.to_thread(_read_file_sync)
    except Exception as e:
        print(f"[obsidian-pg] Failed to read {file_path}: {e}")
        return 0
    # --- AGENT ZERO PATCH END ---


    if not content:
        return 0

    chunks = _chunk_file_content(content, str(file_path))
    if not chunks:
        return 0

    relative_path = str(file_path.relative_to(vault_path))
    folder_type = _extract_folder_type(file_path, vault_path)
    modified_time = datetime.fromtimestamp(file_path.stat().st_mtime)

    # Contextual enrichment (LLM-generated context prefix per chunk)
    chunk_dicts = [{"content": c} for c in chunks]
    if ENRICHMENT_ENABLED:
        try:
            chunk_dicts = await asyncio.to_thread(
                enrich_chunks_batch,
                chunk_dicts,
                file_path.name,
                folder_type,
            )
        except Exception as e:
            print(f"[obsidian-pg] Enrichment failed for {file_path}, proceeding without: {e}")

    # Use enriched content for embedding (includes context prefix)
    texts_for_embedding = [
        cd.get("enriched_content", cd["content"])[:6000]
        for cd in chunk_dicts
    ]

    # Generate embeddings (sync HTTP call — run in thread to avoid blocking event loop)
    try:
        embeddings = await asyncio.to_thread(embed_texts, texts_for_embedding)
    except Exception as e:
        print(f"[obsidian-pg] Embedding failed for {file_path}: {e}")
        return 0

    pool = await _get_pool()

    for i, (cd, embedding) in enumerate(zip(chunk_dicts, embeddings)):
        chunk_id = _generate_chunk_id(relative_path, i, file_hash)
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
        context_prefix = cd.get("context_prefix")
        original_content = cd.get("original_content")
        # Store the original content in 'content' column, context separately
        chunk_content = cd["content"]

        await pool.execute(
            """INSERT INTO obsidian_chunks
               (chunk_id, file_path, file_name, folder, chunk_index, content, embedding,
                file_hash, modified_at, metadata, context_prefix, original_content)
               VALUES ($1, $2, $3, $4, $5, $6, $7::vector, $8, $9, $10, $11, $12)
               ON CONFLICT (chunk_id) DO UPDATE
               SET content = EXCLUDED.content, embedding = EXCLUDED.embedding,
                   file_hash = EXCLUDED.file_hash, modified_at = EXCLUDED.modified_at,
                   context_prefix = EXCLUDED.context_prefix,
                   original_content = EXCLUDED.original_content,
                   updated_at = NOW()""",
            chunk_id, relative_path, file_path.name, folder_type, i,
            chunk_content, embedding_str, file_hash, modified_time, "{}",
            context_prefix, original_content,
        )

    # ── KG Extraction: wikilinks + YAML ────────────────────────────
    try:
        from .kg_extract import extract_wikilinks, extract_wikilink_relations
        wl_entities = extract_wikilinks(content, relative_path)
        wl_relations = extract_wikilink_relations(content, relative_path)

        if wl_entities or wl_relations:
            chunk_ids_for_kg = [
                _generate_chunk_id(relative_path, i, file_hash) for i in range(len(chunks))
            ]

            for ent in wl_entities:
                # Skip noisy entities (file extensions, paths, very short)
                name = ent["name"]
                if len(name) < 2 or "." in name and any(name.endswith(ext) for ext in (".jpg", ".png", ".pdf", ".docx", ".md")):
                    continue

                await pool.execute("""
                    INSERT INTO kg_entities (name, type, source_file)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (name, type) DO UPDATE SET updated_at = NOW()
                """, name, ent["type"], ent.get("source_file", ""))

                entity_id = await pool.fetchval(
                    "SELECT id FROM kg_entities WHERE name = $1 AND type = $2", name, ent["type"]
                )

                if entity_id:
                    # Link to first chunk of this file
                    for cid in chunk_ids_for_kg[:1]:
                        await pool.execute("""
                            INSERT INTO kg_entity_chunks (entity_id, chunk_id)
                            VALUES ($1, $2)
                            ON CONFLICT (entity_id, chunk_id) DO NOTHING
                        """, entity_id, cid)

            for rel in wl_relations:
                src_id = await pool.fetchval(
                    "SELECT id FROM kg_entities WHERE name = $1 LIMIT 1", rel["source_name"]
                )
                tgt_id = await pool.fetchval(
                    "SELECT id FROM kg_entities WHERE name = $1 LIMIT 1", rel["target_name"]
                )
                if src_id and tgt_id and src_id != tgt_id:
                    await pool.execute("""
                        INSERT INTO kg_relations (source_id, target_id, relation_type, source_file)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT DO NOTHING
                    """, src_id, tgt_id, rel.get("type", "LINKS_TO"), relative_path)

    except Exception as e:
        print(f"[obsidian-kg] KG extraction failed for {file_path.name}: {e}")

    # ── Cross-RAG sync ────────────────────────────────────────────
    try:
        await _crossrag_sync_file(pool, relative_path)
    except Exception as e:
        print(f"[obsidian-kg] cross-rag sync failed for {file_path.name} (non-fatal): {e}")

    return len(chunks)


async def _crossrag_sync_file(pool: asyncpg.Pool, file_path: str):
    """Sync KG entities from a single file to cross_rag database."""
    import sys, os
    sys.path.insert(0, os.environ.get("CROSSRAG_SCRIPTS", "/app/scripts"))
    os.environ.setdefault("CROSSRAG_DSN", "postgresql://klara:klara_docs_2026@host.docker.internal:5433/cross_rag")

    from cross_rag_sync import sync_entities_to_crossrag, get_crossrag_pool, close_crossrag_pool

    # Get entities associated with this file (via source_file or entity_chunks)
    entities_rows = await pool.fetch("""
        SELECT id, name, type FROM kg_entities
        WHERE source_file = $1
    """, file_path)

    if not entities_rows:
        return

    entities = [{"id": str(e["id"]), "name": e["name"], "type": e["type"]} for e in entities_rows]

    crossrag_pool = await get_crossrag_pool()
    stats = await sync_entities_to_crossrag("obsidian_rag", entities, crossrag_pool)
    if stats.get("created") or stats.get("linked_exact") or stats.get("linked_fuzzy"):
        print(f"[obsidian-kg] cross-rag: {stats}")
    await close_crossrag_pool()


async def ingest_vault(
    vault_path: str,
    force: bool = False,
    collection_name: str = "obsidian_notes",  # ignored, kept for compat
) -> dict[str, Any]:
    """Ingest entire Obsidian vault with incremental sync into PostgreSQL."""
    vault = Path(vault_path)

    stored_hashes = await _get_stored_hashes()

    # Run file scanning and hashing in a single thread to avoid per-file overhead
    def _scan_and_hash():
        files = scan_vault_files(vault_path)
        print(f"[obsidian-pg] Found {len(files)} files in vault", flush=True)
        result = []
        for f in files:
            rel = str(f.relative_to(vault))
            h = _calculate_file_hash(f)
            result.append((f, rel, h))
        return result

    file_entries = await asyncio.to_thread(_scan_and_hash)
    print(f"[obsidian-pg] Hashed {len(file_entries)} files", flush=True)

    stats = {
        "total_files": len(file_entries),
        "processed_files": 0,
        "skipped_files": 0,
        "total_chunks": 0,
        "errors": [],
        "changed_files": [],
        "new_files": [],
        "kg_stats": {"entities": 0, "relations": 0, "files_synced": 0, "deleted_cleanup": 0},
    }

    # Build list of files that need processing
    to_process = []
    for file_path, relative_path, current_hash in file_entries:
        if not current_hash:
            stats["errors"].append(f"Failed to hash: {relative_path}")
            continue

        if not force and stored_hashes.get(relative_path) == current_hash:
            stats["skipped_files"] += 1
            continue

        to_process.append((file_path, relative_path, current_hash))

    print(f"[obsidian-pg] {len(to_process)} files to process, {stats['skipped_files']} skipped", flush=True)

    for idx, (file_path, relative_path, current_hash) in enumerate(to_process):
        # Yield control periodically so the event loop can serve healthchecks
        if idx % 5 == 0:
            await asyncio.sleep(0)

        is_update = relative_path in stored_hashes

        if is_update:
            deleted = await _delete_file_chunks(relative_path)
            print(f"[obsidian-pg] Deleted {deleted} old chunks for {relative_path}", flush=True)
            stats["changed_files"].append(relative_path)
        else:
            stats["new_files"].append(relative_path)

        try:
            chunk_count = await _ingest_file(file_path, vault, current_hash)
            await _update_hash(relative_path, current_hash, chunk_count)
            stats["processed_files"] += 1
            stats["total_chunks"] += chunk_count

            # Incremental KG sync for changed/new files
            try:
                content = await asyncio.to_thread(file_path.read_text, encoding="utf-8")
                kg_result = await kg_extract.incremental_kg_update(
                    file_path=relative_path,
                    content=content,
                )
                stats["kg_stats"]["entities"] += kg_result["extracted"]["entities"]
                stats["kg_stats"]["relations"] += kg_result["extracted"]["relations"]
                stats["kg_stats"]["files_synced"] += 1
            except Exception as e:
                print(f"[obsidian-pg] KG sync failed for {relative_path} (non-fatal): {e}")

            if stats["processed_files"] % 50 == 0:
                print(f"[obsidian-pg] Progress: {stats['processed_files']} files, {stats['total_chunks']} chunks", flush=True)
        except Exception as e:
            error_msg = f"Failed to ingest {relative_path}: {str(e)}"
            stats["errors"].append(error_msg)
            print(f"[obsidian-pg] {error_msg}")

    # KG cleanup: remove data for files no longer in vault
    try:
        current_paths = {rel for _, rel, _ in file_entries}
        deleted_cleanup = await kg_extract.cleanup_deleted_files(current_paths)
        stats["kg_stats"]["deleted_cleanup"] = deleted_cleanup.get("files_cleaned", 0)
    except Exception as e:
        print(f"[obsidian-pg] KG deleted files cleanup failed (non-fatal): {e}")

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

    kg = stats["kg_stats"]
    print(f"[obsidian-pg] Ingest complete: {stats['processed_files']} files, "
          f"{stats['total_chunks']} chunks, {len(stats['errors'])} errors, "
          f"KG: {kg['files_synced']} synced, {kg['entities']}E/{kg['relations']}R extracted, "
          f"{kg['deleted_cleanup']} deleted files cleaned", flush=True)

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
