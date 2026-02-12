#!/usr/bin/env python3
"""Test ChromaDB connection."""

import chromadb
import time

def test_connection():
    print("Testing ChromaDB connection...")
    
    try:
        client = chromadb.HttpClient(host="localhost", port=8100)
        
        # Test heartbeat
        print("Attempting heartbeat...")
        heartbeat = client.heartbeat()
        print(f"Heartbeat successful: {heartbeat}")
        
        # List collections
        collections = client.list_collections()
        print(f"Collections: {len(collections)} found")
        
        # Try to create a test collection
        test_collection = client.get_or_create_collection(
            name="test_collection",
            metadata={"hnsw:space": "cosine"},
        )
        print(f"Test collection created: {test_collection.name}")
        
        # Clean up
        client.delete_collection("test_collection")
        print("Test collection deleted - connection working!")
        
        return True
        
    except Exception as e:
        print(f"ChromaDB connection failed: {e}")
        return False

if __name__ == "__main__":
    if test_connection():
        print("\n✅ ChromaDB is ready for ingestion!")
    else:
        print("\n❌ ChromaDB connection failed")
        print("Make sure ChromaDB is running on localhost:8100")