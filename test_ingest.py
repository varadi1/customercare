#!/usr/bin/env python3
"""Test script for Obsidian ingestion."""

import sys
import os
import time
from pathlib import Path

# Add the backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

# Set environment variables
os.environ["CHROMA_HOST"] = "localhost"
os.environ["CHROMA_PORT"] = "8100"
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")

# Import the ingest module
from app.obsidian.ingest import ingest_vault

def main():
    vault_path = "/Users/varadiimre/Library/Mobile Documents/iCloud~md~obsidian/Documents/Para/"
    
    print(f"Starting Obsidian vault ingestion...")
    print(f"Vault path: {vault_path}")
    
    start_time = time.time()
    
    try:
        result = ingest_vault(
            vault_path=vault_path,
            force=True,
            collection_name="obsidian_notes"
        )
        
        elapsed = time.time() - start_time
        
        print("\n=== INGESTION COMPLETE ===")
        print(f"Total files: {result['total_files']}")
        print(f"Processed files: {result['processed_files']}")
        print(f"Skipped files: {result['skipped_files']}")
        print(f"Total chunks: {result['total_chunks']}")
        print(f"Time elapsed: {elapsed:.2f} seconds")
        
        if result['errors']:
            print(f"\nErrors ({len(result['errors'])}):")
            for error in result['errors']:
                print(f"  - {error}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()