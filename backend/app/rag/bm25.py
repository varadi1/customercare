"""BM25 keyword search index for hybrid retrieval."""

from __future__ import annotations

import re
import threading
from typing import Optional
from rank_bm25 import BM25Okapi


class BM25Index:
    """In-memory BM25 index over ChromaDB chunks.

    Loads all documents from ChromaDB on first search, then caches.
    Call invalidate() after ingestion to force reload.
    """

    _instance: Optional["BM25Index"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._bm25: Optional[BM25Okapi] = None
        self._docs: list[dict] = []  # [{text, id, metadata}, ...]
        self._tokenized: list[list[str]] = []
        self._dirty = True

    @classmethod
    def get(cls) -> "BM25Index":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def invalidate(self):
        """Mark index as stale — will reload on next search."""
        self._dirty = True

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer for Hungarian text."""
        text = text.lower()
        # Keep accented chars (Hungarian), remove other punctuation
        tokens = re.findall(r"[a-záéíóöőúüű0-9]+", text)
        # Filter very short tokens
        return [t for t in tokens if len(t) > 1]

    def _load_from_chromadb(self):
        """Load all chunks from ChromaDB into the BM25 index."""
        from .search import get_collection

        collection = get_collection()
        count = collection.count()
        if count == 0:
            self._bm25 = None
            self._docs = []
            self._tokenized = []
            self._dirty = False
            return

        # Fetch all documents (ChromaDB supports get with no filter)
        # Batch if needed (ChromaDB default limit is large enough for ~500 chunks)
        result = collection.get(
            include=["documents", "metadatas"],
            limit=count,
        )

        self._docs = []
        self._tokenized = []

        if result["documents"]:
            for i, doc_text in enumerate(result["documents"]):
                meta = result["metadatas"][i] if result["metadatas"] else {}
                doc_id = result["ids"][i] if result["ids"] else str(i)

                # Skip expired documents
                if meta.get("valid_to", "") != "":
                    continue

                self._docs.append({
                    "id": doc_id,
                    "text": doc_text,
                    "metadata": meta,
                })
                self._tokenized.append(self._tokenize(doc_text))

        if self._tokenized:
            self._bm25 = BM25Okapi(self._tokenized)
        else:
            self._bm25 = None

        self._dirty = False
        print(f"[hanna] BM25 index loaded: {len(self._docs)} documents")

    def search(self, query: str, top_k: int = 40) -> list[dict]:
        """BM25 keyword search.

        Returns list of dicts with: text, source, category, chunk_type, score, metadata
        """
        if self._dirty:
            self._load_from_chromadb()

        if not self._bm25 or not self._docs:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Get top-K indices
        scored_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []
        for idx in scored_indices:
            if scores[idx] <= 0:
                continue
            doc = self._docs[idx]
            meta = doc["metadata"]
            results.append({
                "id": doc["id"],
                "text": doc["text"],
                "source": meta.get("source", ""),
                "category": meta.get("category", ""),
                "chunk_type": meta.get("chunk_type", ""),
                "score": round(float(scores[idx]), 4),
                "metadata": meta,
                "bm25_score": round(float(scores[idx]), 4),
            })

        return results
