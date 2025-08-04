#!/usr/bin/env python3
"""
Script to ingest all documents in data/documents folder using semantic ingestion.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingestion_semantic import SemanticIngestionPipeline


def main():
    """Ingest all documents in data/documents folder."""
    print("🚀 Starting batch ingestion of all documents...")

    # Initialize pipeline
    pipeline = SemanticIngestionPipeline()

    # Ingest all documents in data/documents folder
    results = pipeline.ingest_directory(
        directory_path="data/documents/",
        recursive=True,  # Search subdirectories
        include_vision=True,  # Process images/charts/tables
        apply_contextual_retrieval=True,  # Apply contextual retrieval
        store_embeddings=True,  # Store in vector database
    )

    # Print results
    print("\n" + "=" * 60)
    print("  INGESTION COMPLETED")
    print("=" * 60)
    print(
        f"✅ Successfully processed: {results['successful_documents']}/{results['total_documents']} documents"
    )
    print(f"❌ Failed: {results['failed_documents']} documents")
    print(f"📄 Total chunks created: {results['total_chunks_created']}")
    print(f"🔮 Total embeddings stored: {results['total_embeddings_stored']}")
    print(f"⏱️  Total processing time: {results['batch_processing_time']:.2f}s")

    # Show failed documents if any
    if results["failed_documents"] > 0:
        print("\n❌ Failed documents:")
        for failure in results["failed_documents_details"]:
            print(f"  - {failure['file']}: {failure['error']}")

    print("\n💡 You can now search documents using:")
    print("  pipeline.search('your query here')")


if __name__ == "__main__":
    main()
