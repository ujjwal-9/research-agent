#!/usr/bin/env python3
"""
Script to ingest all documents in data/documents folder using semantic ingestion.
Supports both parallel and sequential processing modes.
"""

import argparse
import time

from src.ingestion_semantic import SemanticIngestionPipeline


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Semantic document ingestion with parallel processing support"
    )

    parser.add_argument(
        "--directory",
        default="data/documents/",
        help="Directory containing documents to process (default: data/documents/)",
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel processing (overrides environment variable)",
    )

    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Force sequential processing (overrides environment variable)",
    )

    parser.add_argument(
        "--max-parallel-docs",
        type=int,
        help="Maximum number of documents to process in parallel",
    )

    parser.add_argument(
        "--embedding-batch-size", type=int, help="Batch size for embedding generation"
    )

    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="Disable vision processing for images/charts/tables",
    )

    parser.add_argument(
        "--no-contextual", action="store_true", help="Disable contextual retrieval"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't store embeddings, just process documents",
    )

    return parser.parse_args()


def main():
    """Ingest all documents in specified folder with parallel processing support."""
    args = parse_arguments()

    print("🚀 Starting batch ingestion with semantic chunking (gradient method)")
    print("=" * 70)

    # Initialize pipeline
    print("🔧 Initializing semantic ingestion pipeline...")
    pipeline = SemanticIngestionPipeline()

    # Override configuration if arguments provided
    if args.max_parallel_docs:
        pipeline.config._max_parallel_documents = args.max_parallel_docs

    if args.embedding_batch_size:
        pipeline.config._embedding_batch_size = args.embedding_batch_size

    if args.parallel:
        pipeline.config._enable_parallel_processing = True
    elif args.sequential:
        pipeline.config._enable_parallel_processing = False

    # Display configuration
    config = pipeline.config
    print("📊 Processing Configuration:")
    print(f"   - Directory: {args.directory}")
    print(f"   - Parallel processing: {config.enable_parallel_processing}")
    if config.enable_parallel_processing:
        print(f"   - Max parallel documents: {config.max_parallel_documents}")
        print(f"   - Max parallel chunks: {config.max_parallel_chunks}")
        print(f"   - Embedding batch size: {config.embedding_batch_size}")
    print(f"   - Vision processing: {not args.no_vision}")
    print(f"   - Contextual retrieval: {not args.no_contextual}")
    print(f"   - Store embeddings: {not args.dry_run}")
    print()

    # Start timing
    start_time = time.time()

    # Ingest all documents in specified folder
    print(f"📂 Processing all documents in {args.directory} (recursive)...")
    results = pipeline.ingest_directory(
        directory_path=args.directory,
        recursive=True,  # Search subdirectories
        include_vision=not args.no_vision,  # Process images/charts/tables
        apply_contextual_retrieval=not args.no_contextual,  # Apply contextual retrieval
        store_embeddings=not args.dry_run,  # Store in vector database
    )

    # Calculate total time
    total_time = time.time() - start_time

    # Print results
    print("\n" + "=" * 70)
    print("  SEMANTIC INGESTION COMPLETED")
    print("=" * 70)

    if results["success"]:
        print(
            f"✅ Successfully processed: {results['successful_documents']}/{results['total_documents']} documents"
        )
        print(f"❌ Failed: {results['failed_documents']} documents")
        print(f"📄 Total chunks created: {results['total_chunks_created']}")
        print(f"🔮 Total embeddings stored: {results['total_embeddings_stored']}")
        print(f"⏱️  Total processing time: {total_time:.2f}s")

        # Calculate performance metrics
        if results["total_documents"] > 0:
            avg_time_per_doc = total_time / results["total_documents"]
            docs_per_minute = (results["total_documents"] / total_time) * 60
            print("📈 Performance metrics:")
            print(f"   - Average time per document: {avg_time_per_doc:.2f}s")
            print(f"   - Documents per minute: {docs_per_minute:.1f}")

            if results["total_chunks_created"] > 0:
                chunks_per_second = results["total_chunks_created"] / total_time
                print(f"   - Chunks processed per second: {chunks_per_second:.1f}")

        # Show processing details
        print("\n🔧 Processing Configuration Used:")
        settings = results["settings"]
        print(
            f"   - Processing mode: {'Parallel' if config.enable_parallel_processing else 'Sequential'}"
        )
        if config.enable_parallel_processing:
            print(f"   - Max parallel documents: {config.max_parallel_documents}")
            print(f"   - Embedding batch size: {config.embedding_batch_size}")
        print("   - Semantic chunking: gradient method")
        print(f"   - Vision processing: {settings['include_vision']}")
        print(f"   - Contextual retrieval: {settings['apply_contextual_retrieval']}")
        print(f"   - Embeddings stored: {settings['store_embeddings']}")
    else:
        print("❌ Batch processing failed")

    # Show failed documents if any
    if results.get("failed_documents", 0) > 0:
        print("\n❌ Failed documents:")
        for failure in results.get("failed_documents_details", []):
            print(f"  - {failure['file']}: {failure['error']}")

    # Show sample successful processing
    if results.get("individual_results"):
        successful_results = [
            r for r in results["individual_results"] if r.get("success")
        ]
        if successful_results:
            print("\n📊 Sample processing results:")
            for result in successful_results[:3]:  # Show first 3
                stats = result.get("statistics", {})
                print(f"  📄 {result['file_name']}:")
                print(f"     - Chunks: {stats.get('total_chunks_created', 0)}")
                print(f"     - Embeddings: {stats.get('embeddings_stored', 0)}")
                print(f"     - Time: {result.get('processing_time', 0):.1f}s")

    print("\n💡 Search your documents using:")
    print(
        "   python -c \"from src.ingestion_semantic import SemanticIngestionPipeline; pipeline = SemanticIngestionPipeline(); print(pipeline.search('your query here'))\""
    )

    print("\n🎯 Semantic Features Used:")
    print("   ✅ Gradient-based semantic chunking for better boundaries")
    print("   ✅ Contextual retrieval for enhanced chunk semantics")
    print("   ✅ OpenAI text-embedding-3-large (3072 dimensions)")
    print("   ✅ Vision processing framework (images/charts/tables)")
    if config.enable_parallel_processing:
        print("   ✅ Parallel processing for faster document ingestion")

    print("\n⚡ Performance Tips:")
    if not config.enable_parallel_processing:
        print(
            "   • Enable parallel processing: python scripts/run_semantic_ingestion.py --parallel"
        )
    print(
        f"   • Adjust parallel documents: --max-parallel-docs {config.max_parallel_documents * 2}"
    )
    print(
        f"   • Increase embedding batch size: --embedding-batch-size {config.embedding_batch_size * 2}"
    )


if __name__ == "__main__":
    main()
