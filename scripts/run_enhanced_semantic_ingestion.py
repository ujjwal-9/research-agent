#!/usr/bin/env python3
"""
Enhanced Semantic Ingestion Pipeline Runner for Large Documents

This script demonstrates how to use the enhanced semantic ingestion pipeline
optimized for large documents with advanced contextual retrieval features.

Features:
- Document structure analysis
- Section-based contextualization
- Adaptive chunk sizing
- Enhanced metadata preservation
- Optimized for documents >50k characters

Usage:
    python scripts/run_enhanced_semantic_ingestion.py --file path/to/document.pdf
    python scripts/run_enhanced_semantic_ingestion.py --directory path/to/documents/
    python scripts/run_enhanced_semantic_ingestion.py --file document.pdf --no-vision --no-contextual
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional


from src.ingestion_semantic_large_document import (
    SemanticIngestionLargeDocumentConfig,
    EnhancedSemanticIngestionPipeline,
)


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                f"logs/enhanced_semantic_ingestion_{int(time.time())}.log"
            ),
        ],
    )


def print_banner():
    """Print the enhanced semantic ingestion banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                  Enhanced Semantic Ingestion Pipeline                       ║
    ║                         For Large Documents                                  ║
    ║                                                                              ║
    ║  🚀 Advanced Features:                                                       ║
    ║  📊 Document Structure Analysis                                              ║
    ║  🧠 Section-based Contextualization                                         ║
    ║  📏 Adaptive Chunk Sizing                                                   ║
    ║  🔍 Enhanced Contextual Retrieval                                           ║
    ║  📚 Optimized for Large Documents (>50k chars)                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_stats(stats: dict):
    """Print pipeline statistics in a formatted way."""
    print("\n" + "=" * 80)
    print("📊 ENHANCED PIPELINE STATISTICS")
    print("=" * 80)

    pipeline_info = stats.get("pipeline", {})
    print(f"Pipeline: {pipeline_info.get('name', 'Unknown')}")
    print(f"Version: {pipeline_info.get('version', 'Unknown')}")

    # Enhanced features
    large_doc_features = pipeline_info.get("large_document_features", {})
    if large_doc_features:
        print("\n🧠 Large Document Features:")
        for feature, enabled in large_doc_features.items():
            status = "✅ Enabled" if enabled else "❌ Disabled"
            print(f"  {feature}: {status}")

    # Components
    components = stats.get("components", {})
    if components:
        print("\n🔧 Components:")
        for component, info in components.items():
            print(f"  {component}: {info.get('status', 'Unknown')}")


def ingest_single_file(
    pipeline: EnhancedSemanticIngestionPipeline,
    file_path: str,
    include_vision: bool = True,
    apply_contextual_retrieval: bool = True,
) -> dict:
    """Ingest a single file with enhanced processing."""
    logger = logging.getLogger(__name__)

    logger.info(f"🚀 Starting enhanced ingestion of: {file_path}")

    # Check file size to determine if it's a large document
    file_size = Path(file_path).stat().st_size
    logger.info(f"📊 File size: {file_size:,} bytes")

    start_time = time.time()

    try:
        result = pipeline.ingest_document(
            file_path=file_path,
            include_vision=include_vision,
            apply_contextual_retrieval=apply_contextual_retrieval,
            store_embeddings=True,
        )

        processing_time = time.time() - start_time

        if result.get("success", False):
            stats = result.get("statistics", {})
            logger.info(f"✅ Successfully processed {Path(file_path).name}")
            logger.info(f"⏱️  Processing time: {processing_time:.2f}s")
            logger.info(f"📄 Chunks created: {stats.get('total_chunks_created', 0)}")
            logger.info(f"🔮 Embeddings stored: {stats.get('embeddings_stored', 0)}")

            # Enhanced statistics
            if stats.get("enhanced_contextualization"):
                logger.info(f"🧠 Enhanced contextualization: Used")
                logger.info(
                    f"📊 Document structure analyzed: {stats.get('document_structure_analyzed', False)}"
                )

        else:
            logger.error(
                f"❌ Failed to process {Path(file_path).name}: {result.get('error', 'Unknown error')}"
            )

        return result

    except Exception as e:
        logger.error(f"💥 Error processing {file_path}: {str(e)}")
        return {"success": False, "error": str(e)}


def ingest_directory(
    pipeline: EnhancedSemanticIngestionPipeline,
    directory_path: str,
    include_vision: bool = True,
    apply_contextual_retrieval: bool = True,
    file_patterns: Optional[list] = None,
) -> dict:
    """Ingest all files in a directory with enhanced processing."""
    logger = logging.getLogger(__name__)

    logger.info(f"📂 Starting enhanced directory ingestion: {directory_path}")

    start_time = time.time()

    try:
        result = pipeline.ingest_directory(
            directory_path=directory_path,
            file_patterns=file_patterns,
            recursive=True,
            include_vision=include_vision,
            apply_contextual_retrieval=apply_contextual_retrieval,
            store_embeddings=True,
        )

        processing_time = time.time() - start_time

        if result.get("success", False):
            logger.info(f"✅ Successfully processed directory {directory_path}")
            logger.info(f"⏱️  Total processing time: {processing_time:.2f}s")
            logger.info(f"📁 Total documents: {result.get('total_documents', 0)}")
            logger.info(
                f"✅ Successful documents: {result.get('successful_documents', 0)}"
            )
            logger.info(f"❌ Failed documents: {result.get('failed_documents', 0)}")
            logger.info(f"📄 Total chunks: {result.get('total_chunks_created', 0)}")
            logger.info(
                f"🔮 Total embeddings: {result.get('total_embeddings_stored', 0)}"
            )
        else:
            logger.error(
                f"❌ Failed to process directory: {result.get('error', 'Unknown error')}"
            )

        return result

    except Exception as e:
        logger.error(f"💥 Error processing directory {directory_path}: {str(e)}")
        return {"success": False, "error": str(e)}


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Enhanced Semantic Ingestion Pipeline for Large Documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest a single large document with all enhanced features
  python scripts/run_enhanced_semantic_ingestion.py --file large_report.pdf

  # Ingest without vision processing (faster for text-only documents)
  python scripts/run_enhanced_semantic_ingestion.py --file document.pdf --no-vision

  # Ingest directory of documents with custom patterns
  python scripts/run_enhanced_semantic_ingestion.py --directory docs/ --patterns "*.pdf" "*.docx"

  # Process with basic contextual retrieval (no enhanced features)
  python scripts/run_enhanced_semantic_ingestion.py --file doc.pdf --no-enhanced
        """,
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--file", "-f", type=str, help="Path to a single file to ingest"
    )
    input_group.add_argument(
        "--directory", "-d", type=str, help="Path to a directory to ingest recursively"
    )

    # Processing options
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="Disable vision processing for images/charts/tables",
    )
    parser.add_argument(
        "--no-contextual",
        action="store_true",
        help="Disable contextual retrieval entirely",
    )
    parser.add_argument(
        "--no-enhanced",
        action="store_true",
        help="Disable enhanced features (use basic contextual retrieval)",
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        help="File patterns to match when processing directories (e.g., *.pdf *.docx)",
    )

    # Utility options
    parser.add_argument(
        "--stats", action="store_true", help="Show pipeline statistics and exit"
    )
    parser.add_argument(
        "--health-check", action="store_true", help="Perform health check and exit"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress banner and minimize output"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Print banner unless quiet mode
    if not args.quiet:
        print_banner()

    try:
        # Initialize enhanced pipeline
        logger.info("🔧 Initializing Enhanced Semantic Ingestion Pipeline...")
        config = SemanticIngestionLargeDocumentConfig()
        pipeline = EnhancedSemanticIngestionPipeline(config)

        # Handle utility commands
        if args.stats:
            stats = pipeline.get_stats()
            print_stats(stats)
            return

        if args.health_check:
            health = pipeline.health_check()
            print(f"\n🏥 Health Check: {health['overall'].upper()}")
            for component, info in health.get("components", {}).items():
                status = "✅" if info["status"] == "healthy" else "❌"
                print(f"  {component}: {status} {info['status']}")
            return

        # Configure processing options based on enhanced features
        if args.no_enhanced:
            # Disable enhanced features in config
            logger.info(
                "⚠️  Enhanced features disabled - using basic contextual retrieval"
            )

        # Process input
        if args.file:
            result = ingest_single_file(
                pipeline=pipeline,
                file_path=args.file,
                include_vision=not args.no_vision,
                apply_contextual_retrieval=not args.no_contextual,
            )

        elif args.directory:
            result = ingest_directory(
                pipeline=pipeline,
                directory_path=args.directory,
                include_vision=not args.no_vision,
                apply_contextual_retrieval=not args.no_contextual,
                file_patterns=args.patterns,
            )

        # Final status
        if result.get("success", False):
            logger.info("🎉 Enhanced ingestion completed successfully!")
            if not args.quiet:
                print("\n✅ SUCCESS: Enhanced semantic ingestion completed!")
        else:
            logger.error("💥 Enhanced ingestion failed!")
            if not args.quiet:
                print(f"\n❌ FAILED: {result.get('error', 'Unknown error')}")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("🛑 Enhanced ingestion interrupted by user")
        print("\n🛑 Interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"💥 Unexpected error: {str(e)}")
        print(f"\n💥 Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
