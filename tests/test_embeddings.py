#!/usr/bin/env python3
"""Quick test to verify OpenAI embeddings are working."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ingestion.document_store import DocumentStore

def test_embeddings():
    """Test OpenAI embeddings functionality."""
    try:
        print("Testing OpenAI embeddings...")
        
        store = DocumentStore()
        
        # Test embedding generation
        test_texts = ["This is a test document", "Another test sentence"]
        embeddings = store._generate_embeddings(test_texts)
        
        print(f"✓ Generated embeddings for {len(test_texts)} texts")
        print(f"  - Embedding dimension: {len(embeddings[0])}")
        print(f"  - First few values: {embeddings[0][:5]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Embedding test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_embeddings()
    if success:
        print("\n🎉 OpenAI embeddings are working correctly!")
    else:
        print("\n❌ OpenAI embeddings test failed.")
        sys.exit(1)