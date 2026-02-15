"""Ingest PDF files from /app/data/pdfs/ into the RAG knowledge base."""

import sys
import os
from pathlib import Path

# Map filenames to chunk types
PDF_CHUNK_TYPES = {
    "Felhivas": "palyazat_felhivas",
    "GYIK": "gyik",
    "kitoltesi": "segedlet",
    "Tajekoztatas": "segedlet",
}


def get_chunk_type(filename: str) -> str:
    for key, ctype in PDF_CHUNK_TYPES.items():
        if key.lower() in filename.lower():
            return ctype
    return "palyazat_melleklet"


def ingest_all_pdfs(pdf_dir: str = "/app/data/pdfs"):
    """Ingest all PDFs from the given directory."""
    from app.rag.ingest import ingest_pdf
    from app.rag.bm25 import BM25Index

    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        print(f"[ingest_pdfs] Directory {pdf_dir} not found")
        return

    pdfs = list(pdf_path.glob("*.pdf"))
    if not pdfs:
        print("[ingest_pdfs] No PDF files found")
        return

    total_chunks = 0
    for pdf in pdfs:
        chunk_type = get_chunk_type(pdf.name)
        print(f"[ingest_pdfs] Ingesting {pdf.name} as {chunk_type}...")
        try:
            count = ingest_pdf(
                pdf_path=str(pdf),
                source=pdf.name,
                category="oetp",
                chunk_type=chunk_type,
            )
            total_chunks += count
            print(f"[ingest_pdfs] {pdf.name}: {count} chunks")
        except Exception as e:
            print(f"[ingest_pdfs] ERROR {pdf.name}: {e}")

    BM25Index.get().invalidate()
    print(f"[ingest_pdfs] Done. Total: {total_chunks} chunks from {len(pdfs)} PDFs")


if __name__ == "__main__":
    ingest_all_pdfs(sys.argv[1] if len(sys.argv) > 1 else "/app/data/pdfs")
