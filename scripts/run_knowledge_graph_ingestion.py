"""Production script for Knowledge Graph Ingestion.

This script provides a production-ready interface for running knowledge graph ingestion
with command-line arguments and proper error handling.
"""

import argparse
import os
import sys
import logging
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ingestion_knowledge_graph import KnowledgeGraphConfig, KnowledgeGraphPipeline


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "knowledge_graph_ingestion.log"),
            logging.StreamHandler(),
        ],
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Knowledge Graph Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all documents in data/documents
  python run_knowledge_graph_ingestion.py --directory data/documents
  
  # Process single document
  python run_knowledge_graph_ingestion.py --file path/to/document.pdf
  
  # Process with specific file types only
  python run_knowledge_graph_ingestion.py --directory data/documents --extensions .pdf .docx
  
  # Clear database before processing
  python run_knowledge_graph_ingestion.py --directory data/documents --clear-db
  
  # Get graph statistics only
  python run_knowledge_graph_ingestion.py --stats-only
        """,
    )

    parser.add_argument(
        "--directory", "-d", type=str, help="Directory containing documents to process"
    )

    parser.add_argument("--file", "-f", type=str, help="Single file to process")

    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[
            ".pdf",
            ".docx",
            ".doc",
            ".xlsx",
            ".xls",
            ".pptx",
            ".ppt",
            ".txt",
            ".csv",
        ],
        help="File extensions to include (default: all supported types)",
    )

    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        default=True,
        help="Search subdirectories recursively (default: True)",
    )

    parser.add_argument(
        "--clear-db",
        action="store_true",
        help="Clear the knowledge graph database before processing",
    )

    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only display graph statistics, don't process documents",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        help="Number of documents to process in parallel (overrides config)",
    )

    parser.add_argument(
        "--max-concurrent",
        type=int,
        help="Maximum concurrent API requests (overrides config)",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


def display_statistics(stats: dict):
    """Display knowledge graph statistics."""
    print("\n📊 Knowledge Graph Statistics:")
    print("=" * 40)
    print(f"Entities: {stats.get('entities', 0)}")
    print(f"Relationships: {stats.get('relationships', 0)}")
    print(f"Documents: {stats.get('documents', 0)}")

    if stats.get("entity_types"):
        print("\nEntity Types:")
        for entity_type, count in stats["entity_types"].items():
            print(f"  {entity_type}: {count}")

    if stats.get("relationship_types"):
        print("\nTop Relationship Types:")
        for rel_type, count in list(stats["relationship_types"].items())[:10]:
            print(f"  {rel_type}: {count}")

    if stats.get("most_connected_entities"):
        print("\nMost Connected Entities:")
        for entity in stats["most_connected_entities"][:5]:
            print(
                f"  {entity['name']} ({entity['type']}): {entity['connections']} connections"
            )

    print()


def main():
    """Main function."""
    args = parse_arguments()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    try:
        # Initialize configuration
        config = KnowledgeGraphConfig()

        # Override config with command line arguments if provided
        if args.batch_size:
            config.batch_size = args.batch_size
        if args.max_concurrent:
            config.max_concurrent_requests = args.max_concurrent

        logger.info("🚀 Starting Knowledge Graph Ingestion")
        logger.info(f"Neo4j URI: {config.neo4j_uri}")
        logger.info(f"Model: {config.knowledge_graph_model}")

        # Initialize pipeline
        with KnowledgeGraphPipeline(config) as pipeline:

            # Clear database if requested
            if args.clear_db:
                logger.warning("⚠️  Clearing knowledge graph database")
                if pipeline.clear_knowledge_graph():
                    logger.info("✅ Database cleared successfully")
                else:
                    logger.error("❌ Failed to clear database")
                    return 1

            # Display statistics if requested
            if args.stats_only:
                stats = pipeline.get_graph_statistics()
                display_statistics(stats)
                return 0

            # Process documents
            if args.file:
                # Process single file
                if not os.path.exists(args.file):
                    logger.error(f"File not found: {args.file}")
                    return 1

                logger.info(f"📄 Processing single file: {args.file}")
                result = pipeline.ingest_document(args.file)

                if result["success"]:
                    logger.info("✅ File processing completed successfully")
                    print(f"Entities extracted: {result['entities_extracted']}")
                    print(
                        f"Relationships extracted: {result['relationships_extracted']}"
                    )
                    print(f"Chunks processed: {result['chunks_processed']}")
                else:
                    logger.error(f"❌ File processing failed: {result.get('error')}")
                    return 1

            elif args.directory:
                # Process directory
                if not os.path.exists(args.directory):
                    logger.error(f"Directory not found: {args.directory}")
                    return 1

                logger.info(f"📂 Processing directory: {args.directory}")
                logger.info(f"File extensions: {args.extensions}")
                logger.info(f"Recursive: {args.recursive}")

                result = pipeline.ingest_directory(
                    directory_path=args.directory,
                    file_patterns=args.extensions,
                    recursive=args.recursive,
                )

                if result["success"]:
                    logger.info("✅ Directory processing completed successfully")
                    print("\n📈 Processing Results:")
                    print(f"Documents found: {result['documents_found']}")
                    print(f"Documents processed: {result['documents_processed']}")
                    print(f"Documents failed: {result['documents_failed']}")
                    print(f"Total entities extracted: {result['entities_extracted']}")
                    print(
                        f"Total relationships extracted: {result['relationships_extracted']}"
                    )
                    print(
                        f"Processing time: {result['processing_time_seconds']:.2f} seconds"
                    )

                    # Display final statistics
                    if result.get("final_graph_stats"):
                        display_statistics(result["final_graph_stats"])
                else:
                    logger.error(
                        f"❌ Directory processing failed: {result.get('error')}"
                    )
                    return 1
            else:
                logger.error("❌ Please specify either --file or --directory")
                return 1

        logger.info("🎉 Knowledge graph ingestion completed successfully")
        return 0

    except KeyboardInterrupt:
        logger.info("🛑 Ingestion interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
