"""Obsidian ingestion — hash-based incremental sync, chunking, ChromaDB storage."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb

from ..config import settings
from ..rag.chunker import chunk_markdown, chunk_text
from ..rag.embeddings import embed_texts


def get_obsidian_collection(collection_name: str = "obsidian_notes"):
    """Get or create the Obsidian notes collection."""
    client = chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
    )
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def _generate_obsidian_chunk_id(file_path: str, chunk_index: int, file_hash: str) -> str:
    """Generate deterministic chunk ID from file path, chunk index, and file hash."""
    raw = f"{file_path}::{chunk_index}::{file_hash[:8]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _calculate_file_hash(file_path: Path) -> str:
    """Calculate MD5 hash of file content."""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return ""


def _load_hash_store(hash_file: Path) -> dict[str, str]:
    """Load stored file hashes."""
    if hash_file.exists():
        try:
            with open(hash_file) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_hash_store(hash_file: Path, hashes: dict[str, str]) -> None:
    """Save file hashes to JSON."""
    hash_file.parent.mkdir(parents=True, exist_ok=True)
    with open(hash_file, 'w') as f:
        json.dump(hashes, f, indent=2)


def _extract_folder_type(file_path: Path, vault_path: Path) -> str:
    """Extract folder type from file path (inbox, projects, areas, etc.)."""
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
    """Chunk file content based on size and type."""
    # Small files (<1000 chars) - keep whole
    if len(content) < 1000:
        return [content]
    
    # Large files - use markdown chunking for .md, text chunking for others
    if file_path.endswith('.md'):
        chunks = chunk_markdown(content)
    else:
        chunks = chunk_text(content, chunk_size=1000, chunk_overlap=200)
    
    return chunks if chunks else [content]


def get_last_sync_info() -> dict[str, Any]:
    """Get information about the last vault sync."""
    sync_log_file = Path("/app/data/obsidian_last_sync.json")
    if sync_log_file.exists():
        with open(sync_log_file) as f:
            return json.load(f)
    return {"timestamp": None, "processed_files": 0, "new_files": [], "changed_files": []}


def scan_vault_files(vault_path: str) -> list[Path]:
    """Scan vault for ingestion files, excluding certain patterns."""
    vault = Path(vault_path)
    if not vault.exists():
        raise ValueError(f"Vault path does not exist: {vault_path}")
    
    files = []
    include_folders = {
        "!inbox", "1_projects", "2_areas", "3_resources", 
        "4_archive", "Tags", "TaskNotes"
    }
    
    exclude_patterns = {".obsidian", ".trash"}
    exclude_extensions = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".mp4", ".mov"}
    
    for folder in vault.iterdir():
        if not folder.is_dir() or folder.name in exclude_patterns:
            continue
            
        if folder.name in include_folders:
            for file_path in folder.rglob("*"):
                if (file_path.is_file() 
                    and not any(exclude in str(file_path) for exclude in exclude_patterns)
                    and file_path.suffix.lower() not in exclude_extensions):
                    files.append(file_path)
    
    return files


def ingest_obsidian_file(
    file_path: Path, 
    vault_path: Path, 
    file_hash: str,
    collection_name: str = "obsidian_notes"
) -> int:
    """Ingest a single Obsidian file into ChromaDB."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        return 0
    
    if not content:
        return 0
    
    # Chunk content
    chunks = _chunk_file_content(content, str(file_path))
    if not chunks:
        return 0
    
    # Generate embeddings
    embeddings = embed_texts(chunks)
    
    # Prepare metadata
    relative_path = str(file_path.relative_to(vault_path))
    folder_type = _extract_folder_type(file_path, vault_path)
    modified_time = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
    
    # Get ChromaDB collection
    collection = get_obsidian_collection(collection_name)
    
    # Store chunks
    chunk_ids = []
    chunk_metadatas = []
    
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_id = _generate_obsidian_chunk_id(relative_path, i, file_hash)
        chunk_ids.append(chunk_id)
        
        metadata = {
            "file_path": relative_path,
            "file_name": file_path.name,
            "folder": folder_type,
            "modified": modified_time,
            "hash": file_hash,
            "chunk_index": i,
            "source_type": "obsidian"
        }
        chunk_metadatas.append(metadata)
    
    # Add to ChromaDB
    collection.add(
        ids=chunk_ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=chunk_metadatas
    )
    
    return len(chunks)


def delete_file_chunks(file_path: str, collection_name: str = "obsidian_notes") -> int:
    """Delete all chunks for a specific file from ChromaDB."""
    collection = get_obsidian_collection(collection_name)
    
    # Query for chunks from this file
    results = collection.get(
        where={"file_path": file_path}
    )
    
    if results["ids"]:
        collection.delete(ids=results["ids"])
        return len(results["ids"])
    
    return 0


def ingest_vault(
    vault_path: str, 
    force: bool = False,
    collection_name: str = "obsidian_notes"
) -> dict[str, Any]:
    """Ingest entire Obsidian vault with incremental sync."""
    vault = Path(vault_path)
    hash_file = Path("/app/data/obsidian_hashes.json")
    sync_log_file = Path("/app/data/obsidian_last_sync.json")
    
    # Load existing hashes
    stored_hashes = _load_hash_store(hash_file)
    
    # Scan files
    files = scan_vault_files(vault_path)
    print(f"Found {len(files)} files in vault")
    
    stats = {
        "total_files": len(files),
        "processed_files": 0,
        "skipped_files": 0,
        "total_chunks": 0,
        "errors": [],
        "changed_files": [],  # Track which files were updated
        "new_files": [],      # Track which files are new
    }
    
    new_hashes = {}
    
    for file_path in files:
        relative_path = str(file_path.relative_to(vault))
        current_hash = _calculate_file_hash(file_path)
        
        if not current_hash:
            stats["errors"].append(f"Failed to hash: {relative_path}")
            continue
        
        new_hashes[relative_path] = current_hash
        
        # Skip if hash unchanged (unless force)
        if not force and stored_hashes.get(relative_path) == current_hash:
            stats["skipped_files"] += 1
            continue
        
        # Track if this is an update or new file
        is_update = relative_path in stored_hashes
        
        # Delete old chunks if file existed before
        if is_update:
            deleted = delete_file_chunks(relative_path, collection_name)
            print(f"Deleted {deleted} old chunks for {relative_path}")
            stats["changed_files"].append(relative_path)
        else:
            stats["new_files"].append(relative_path)
        
        # Ingest file
        try:
            chunk_count = ingest_obsidian_file(file_path, vault, current_hash, collection_name)
            stats["processed_files"] += 1
            stats["total_chunks"] += chunk_count
            print(f"Ingested {relative_path}: {chunk_count} chunks")
        except Exception as e:
            error_msg = f"Failed to ingest {relative_path}: {str(e)}"
            stats["errors"].append(error_msg)
            print(error_msg)
    
    # Save updated hashes
    _save_hash_store(hash_file, new_hashes)
    
    # Save sync log for reporting
    sync_log = {
        "timestamp": datetime.now().isoformat(),
        "processed_files": stats["processed_files"],
        "new_files": stats["new_files"],
        "changed_files": stats["changed_files"],
        "total_chunks": stats["total_chunks"],
        "errors": stats["errors"]
    }
    sync_log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(sync_log_file, 'w') as f:
        json.dump(sync_log, f, indent=2)
    
    return stats