"""Script to run the document ingestion pipeline."""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingestion import IngestionPipeline, IngestionConfig


def setup_logging():
    """Set up basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main():
    """Main function to run the ingestion pipeline."""
    parser = argparse.ArgumentParser(description="Document Ingestion Pipeline")

    parser.add_argument(
        "--input",
        "-i",
        default="data/documents",
        help="Input directory containing documents (default: data/documents)",
    )

    parser.add_argument(
        "--document", "-d", help="Process a single document instead of directory"
    )

    parser.add_argument(
        "--patterns",
        "-p",
        nargs="+",
        default=["*.pdf", "*.docx", "*.doc", "*.xlsx", "*.xls", "*.pptx", "*.ppt"],
        help="File patterns to match (default: PDF and Office formats)",
    )

    parser.add_argument("--search", "-s", help="Search query to test after ingestion")

    parser.add_argument(
        "--list-docs",
        action="store_true",
        help="List all documents in the vector store",
    )

    parser.add_argument(
        "--stats", action="store_true", help="Show pipeline and storage statistics"
    )

    parser.add_argument(
        "--delete-doc", help="Delete a specific document from the vector store"
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Initialize pipeline
        logger.info("🚀 Initializing ingestion pipeline...")
        config = IngestionConfig()
        pipeline = IngestionPipeline(config)

        # Handle different operations
        if args.list_docs:
            logger.info("📋 Listing all documents...")
            documents = pipeline.list_documents()
            if documents:
                print("\\n📚 Stored Documents:")
                for i, doc in enumerate(documents, 1):
                    print(f"  {i}. {doc}")
                    info = pipeline.get_document_info(doc)
                    print(
                        f"     Chunks: {info.get('chunk_count', 'unknown')}, "
                        f"Pages: {info.get('page_count', 'unknown')}, "
                        f"Tokens: {info.get('total_tokens', 'unknown')}"
                    )
            else:
                print("📭 No documents found in storage")

        elif args.delete_doc:
            logger.info(f"🗑️  Deleting document: {args.delete_doc}")
            success = pipeline.delete_document(args.delete_doc)
            if success:
                print(f"✅ Document '{args.delete_doc}' deleted successfully")
            else:
                print(f"❌ Failed to delete document '{args.delete_doc}'")

        elif args.stats:
            logger.info("📊 Getting pipeline statistics...")
            stats = pipeline.get_stats()
            print("\\n📊 Pipeline Statistics:")
            print(
                f"  Collection: {stats['collection_stats'].get('collection_name', 'unknown')}"
            )
            print(
                f"  Total Points: {stats['collection_stats'].get('points_count', 'unknown')}"
            )
            print(f"  Documents: {stats['document_count']}")
            print(f"  Status: {stats['collection_stats'].get('status', 'unknown')}")

        elif args.document:
            # Process single document
            logger.info(f"📄 Processing single document: {args.document}")
            if not os.path.exists(args.document):
                print(f"❌ Document not found: {args.document}")
                return

            result = pipeline.ingest_document(args.document)
            if result["success"]:
                print(f"✅ Document processed successfully:")
                print(f"  Name: {result['document_name']}")
                print(f"  Pages: {result['pages']}")
                print(f"  Chunks: {result['chunks']}")
                print(f"  Images: {result['images']}")
                print(f"  Tables: {result['tables']}")
            else:
                print(
                    f"❌ Document processing failed: {result.get('error', 'Unknown error')}"
                )

        else:
            # Process directory
            logger.info(f"📂 Processing directory: {args.input}")
            if not os.path.exists(args.input):
                print(f"❌ Directory not found: {args.input}")
                return

            result = pipeline.ingest_directory(args.input, args.patterns)

            if result["success"]:
                print(f"\\n✅ Directory processing completed:")
                print(f"  Processed: {result['processed']} documents")
                print(f"  Failed: {result['failed']} documents")

                if result["documents"]:
                    print("\\n📚 Processed Documents:")
                    for doc in result["documents"]:
                        print(
                            f"  - {doc['name']}: {doc['chunks']} chunks, {doc['pages']} pages"
                        )

                if result["errors"]:
                    print("\\n❌ Failed Documents:")
                    for error in result["errors"]:
                        print(
                            f"  - {os.path.basename(error['path'])}: {error['error']}"
                        )
            else:
                print(
                    f"❌ Directory processing failed: {result.get('error', 'Unknown error')}"
                )

        # Handle search query if provided
        if args.search:
            logger.info(f"🔍 Searching for: '{args.search}'")
            results = pipeline.search_documents(args.search, limit=5)

            if results:
                print(f"\\n🔍 Search Results for '{args.search}':")
                for i, result in enumerate(results, 1):
                    print(f"\\n{i}. Score: {result['score']:.3f}")
                    print(
                        f"   Document: {result['metadata'].get('document_name', 'unknown')}"
                    )
                    print(
                        f"   Page: {result['metadata'].get('page_number', 'unknown')}"
                    )
                    print(f"   Content: {result['content'][:200]}...")
            else:
                print(f"❌ No results found for '{args.search}'")

    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
