#!/usr/bin/env python3
"""Test script to verify recursive document ingestion."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ingestion.ingest_documents import DocumentIngestionPipeline
from src.ingestion.document_store import DocumentStore


async def test_recursive_ingestion():
    """Test the recursive ingestion functionality."""
    print("🧪 Testing Recursive Document Ingestion")
    print("=" * 50)
    
    try:
        # Initialize pipeline
        pipeline = DocumentIngestionPipeline()
        
        # Clear existing documents for clean test
        print("Clearing existing documents...")
        pipeline.store.clear_all_documents()
        
        # Test basic recursive ingestion
        data_dir = Path("./data")
        print(f"Testing recursive ingestion from {data_dir}")
        
        await pipeline.ingest_directory(data_dir, force_reindex=True)
        
        # Verify results
        store = DocumentStore()
        docs = store.get_unique_documents()
        
        print(f"\n✅ Successfully processed {len(docs)} documents")
        
        # Show directory distribution
        dir_counts = {}
        for doc in docs:
            file_path = Path(doc['file_path'])
            parent_dir = file_path.parent.name
            dir_counts[parent_dir] = dir_counts.get(parent_dir, 0) + 1
        
        print("\nDocuments by directory:")
        for directory, count in sorted(dir_counts.items()):
            print(f"  {directory}: {count} files")
        
        # Test search functionality
        print("\n🔍 Testing search across directories...")
        results = store.search_documents("clinical trial", n_results=3)
        
        print(f"Found {len(results)} results for 'clinical trial':")
        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            file_path = Path(metadata['file_path'])
            print(f"  {i}. {metadata['title']} (from {file_path.parent.name}/)")
        
        # Test with exclusions
        print("\n🚫 Testing directory exclusions...")
        pipeline.store.clear_all_documents()
        
        await pipeline.ingest_directory(
            data_dir, 
            force_reindex=True, 
            exclude_dirs=("temp_files",)
        )
        
        docs_excluded = store.get_unique_documents()
        print(f"With exclusions: {len(docs_excluded)} documents (vs {len(docs)} without)")
        
        # Verify temp_files were excluded
        temp_files_found = any("temp_files" in doc['file_path'] for doc in docs_excluded)
        if not temp_files_found:
            print("✅ Directory exclusion working correctly")
        else:
            print("❌ Directory exclusion failed")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


async def main():
    """Run the test."""
    success = await test_recursive_ingestion()
    
    if success:
        print("\n🎉 All recursive ingestion tests passed!")
    else:
        print("\n💥 Recursive ingestion tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())