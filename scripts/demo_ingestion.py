"""Demo script showcasing the ingestion pipeline capabilities."""

import os
import sys
import time
import logging
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingestion import IngestionPipeline, IngestionConfig


def setup_demo_logging():
    """Set up logging for the demo."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def print_separator(title: str):
    """Print a fancy separator."""
    print("\\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def demo_basic_ingestion():
    """Demonstrate basic document ingestion."""
    print_separator("BASIC DOCUMENT INGESTION DEMO")

    try:
        # Initialize pipeline
        print("🚀 Initializing ingestion pipeline...")
        config = IngestionConfig()
        pipeline = IngestionPipeline(config)

        # Check if data directory exists
        data_dir = "data/documents"
        if not os.path.exists(data_dir):
            print(f"❌ Data directory not found: {data_dir}")
            print("Please create the directory and add some documents for the demo.")
            return

        # List files in data directory
        files = list(Path(data_dir).rglob("*"))
        doc_files = [
            f
            for f in files
            if f.suffix.lower() in [".pdf", ".docx", ".doc", ".xlsx", ".xls"]
        ]

        print(f"📂 Found {len(doc_files)} documents in {data_dir}:")
        for i, file in enumerate(doc_files[:5], 1):  # Show first 5
            print(f"  {i}. {file.name}")
        if len(doc_files) > 5:
            print(f"  ... and {len(doc_files) - 5} more")

        if not doc_files:
            print("❌ No supported documents found in data directory")
            print("Supported formats: PDF, Word, Excel, PowerPoint")
            return

        # Process documents
        print(f"\\n🔄 Processing {len(doc_files)} documents...")
        start_time = time.time()

        results = pipeline.ingest_directory(data_dir)

        end_time = time.time()
        processing_time = end_time - start_time

        # Show results
        print(f"\\n✅ Processing completed in {processing_time:.1f} seconds")
        print(f"  Successfully processed: {results['processed']} documents")
        print(f"  Failed: {results['failed']} documents")

        if results["documents"]:
            print("\\n📚 Processed Documents:")
            for doc in results["documents"]:
                print(f"  - {doc['name']}")
                print(f"    Chunks: {doc['chunks']}, Pages: {doc['pages']}")

        if results["errors"]:
            print("\\n❌ Failed Documents:")
            for error in results["errors"]:
                print(f"  - {os.path.basename(error['path'])}: {error['error']}")

        return pipeline

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        print(f"Traceback: {traceback.format_exc()}")
        return None


def demo_search_capabilities(pipeline):
    """Demonstrate search capabilities."""
    print_separator("SEARCH CAPABILITIES DEMO")

    if not pipeline:
        print("❌ Pipeline not available for search demo")
        return

    # List documents
    documents = pipeline.list_documents()
    print(f"📋 Available documents: {len(documents)}")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc}")

    if not documents:
        print("❌ No documents available for search")
        return

    # Demo search queries
    demo_queries = [
        "financial data",
        "market analysis",
        "revenue projections",
        "competitive landscape",
        "business model",
        "technology",
        "strategy",
    ]

    print("\\n🔍 Testing search with demo queries...")

    for query in demo_queries[:3]:  # Test first 3 queries
        print(f"\\n🔎 Searching for: '{query}'")

        results = pipeline.search_documents(query, limit=3)

        if results:
            print(f"  Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                doc_name = result["metadata"].get("document_name", "unknown")
                page_num = result["metadata"].get("page_number", "unknown")
                score = result["score"]
                content_preview = result["content"][:100].replace("\\n", " ")

                print(f"    {i}. Score: {score:.3f}")
                print(f"       Document: {doc_name} (Page {page_num})")
                print(f"       Preview: {content_preview}...")
        else:
            print(f"  No results found for '{query}'")


def demo_document_details(pipeline):
    """Demonstrate document information retrieval."""
    print_separator("DOCUMENT DETAILS DEMO")

    if not pipeline:
        print("❌ Pipeline not available for document details demo")
        return

    documents = pipeline.list_documents()

    if not documents:
        print("❌ No documents available for details demo")
        return

    # Show details for first document
    doc_name = documents[0]
    print(f"📄 Document Details for: {doc_name}")

    info = pipeline.get_document_info(doc_name)

    if "error" not in info:
        print(f"  Chunks: {info.get('chunk_count', 'unknown')}")
        print(f"  Pages: {info.get('page_count', 'unknown')}")
        print(f"  Total Tokens: {info.get('total_tokens', 'unknown')}")

        pages = info.get("pages", [])
        if pages:
            print(f"  Page Numbers: {', '.join(map(str, pages[:10]))}")
            if len(pages) > 10:
                print(f"    ... and {len(pages) - 10} more pages")
    else:
        print(f"  Error: {info['error']}")

    # Search within this specific document
    print(f"\\n🔍 Searching within '{doc_name}':")

    results = pipeline.search_documents("business", document_name=doc_name, limit=2)

    if results:
        for i, result in enumerate(results, 1):
            page_num = result["metadata"].get("page_number", "unknown")
            score = result["score"]
            content_preview = result["content"][:150].replace("\\n", " ")

            print(f"  {i}. Page {page_num} (Score: {score:.3f})")
            print(f"     {content_preview}...")
    else:
        print("  No results found within this document")


def demo_statistics(pipeline):
    """Demonstrate statistics and monitoring."""
    print_separator("STATISTICS & MONITORING DEMO")

    if not pipeline:
        print("❌ Pipeline not available for statistics demo")
        return

    stats = pipeline.get_stats()

    print("📊 Pipeline Statistics:")
    print(
        f"  Collection: {stats['collection_stats'].get('collection_name', 'unknown')}"
    )
    print(f"  Total Points: {stats['collection_stats'].get('points_count', 'unknown')}")
    print(
        f"  Indexed Vectors: {stats['collection_stats'].get('indexed_vectors_count', 'unknown')}"
    )
    print(f"  Status: {stats['collection_stats'].get('status', 'unknown')}")
    print(f"  Document Count: {stats['document_count']}")

    print("\\n⚙️  Configuration:")
    config = stats["config"]
    print(f"  Chunk Size: {config.get('chunk_size', 'unknown')}")
    print(f"  Chunk Overlap: {config.get('chunk_overlap', 'unknown')}")
    print(f"  Embedding Model: {config.get('openai_embedding_model', 'unknown')}")
    print(f"  OCR Model: {config.get('mistral_parsing_model', 'unknown')}")


def main():
    """Run the complete demo."""
    print("🎬 Document Ingestion Pipeline Demo")
    print("===================================")
    print("This demo showcases the complete ingestion pipeline:")
    print("1. Document processing with OCR")
    print("2. AI-generated descriptions for images and tables")
    print("3. Intelligent content chunking")
    print("4. Vector storage in Qdrant")
    print("5. Semantic search capabilities")

    setup_demo_logging()

    # Run demo sections
    pipeline = demo_basic_ingestion()

    if pipeline:
        demo_search_capabilities(pipeline)
        demo_document_details(pipeline)
        demo_statistics(pipeline)

        print_separator("DEMO COMPLETED")
        print("🎉 All demo sections completed successfully!")
        print("\\nNext steps:")
        print("- Add your own documents to data/documents/")
        print("- Try different search queries")
        print("- Explore the API for integration")
        print("- Check the logs/ directory for detailed processing logs")
    else:
        print_separator("DEMO INCOMPLETE")
        print("❌ Demo could not be completed due to setup issues")
        print("\\nPlease check:")
        print("- Environment variables are set correctly")
        print("- Qdrant is running")
        print("- Documents are available in data/documents/")


if __name__ == "__main__":
    main()
