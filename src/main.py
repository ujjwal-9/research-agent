"""Main entry point for the research system."""

import asyncio
import click
import json
import warnings
from pathlib import Path
from loguru import logger

# Suppress ResourceWarnings for unclosed transports (temporary fix)
warnings.filterwarnings("ignore", category=ResourceWarning)

from src.config import settings
from src.agents.research_orchestrator import ResearchOrchestrator
from src.api.server import create_app


@click.group()
def cli():
    """Research System CLI."""
    pass


@cli.command()
@click.option("--query", "-q", required=True, help="Research query")
@click.option("--interactive", "-i", is_flag=True, help="Interactive mode")
@click.option("--output", "-o", help="Output file for the report")
def research(query: str, interactive: bool, output: str):
    """Run a research query."""

    async def run_research():
        logger.info(f"Starting research for query: {query}")

        async with ResearchOrchestrator() as orchestrator:
            if interactive:
                # Interactive mode with user confirmation
                result = await orchestrator.run_interactive_research(query)
            else:
                # Direct execution
                result = await orchestrator.run_research(query)

            logger.info("Research completed")

            # Display results
            print("\n" + "=" * 80)
            print("RESEARCH REPORT")
            print("=" * 80)
            print(result.final_report)

            # Save to file if requested
            if output:
                output_path = Path(output)
                output_path.write_text(result.final_report)
                print(f"\nReport saved to: {output_path}")

    try:
        asyncio.run(run_research())
    except Exception as e:
        logger.error(f"Research failed: {e}")
        print(f"Error: {e}")


@cli.command()
@click.option("--host", default=settings.api_host, help="API host")
@click.option("--port", default=settings.api_port, help="API port")
def serve(host: str, port: int):
    """Start the API server."""
    import uvicorn

    app = create_app()
    uvicorn.run(app, host=host, port=port)


@cli.command()
def status():
    """Check system status."""
    logger.info("Checking system status...")

    # Check document index
    from src.ingestion.document_store import DocumentStore

    store = DocumentStore()
    doc_count = store.get_document_count()
    unique_docs = store.get_unique_documents()

    print(f"Documents indexed: {doc_count} chunks from {len(unique_docs)} documents")
    print(f"Vector database: Qdrant at {settings.qdrant_host}:{settings.qdrant_port}")
    print(f"OpenAI model: {settings.openai_model}")

    # Show document types
    if unique_docs:
        file_types = {}
        for doc in unique_docs:
            file_type = doc.get("file_type", "unknown")
            file_types[file_type] = file_types.get(file_type, 0) + 1

        print("Document types:")
        for file_type, count in file_types.items():
            print(f"  {file_type}: {count}")

    print("System ready ✓")


@cli.command()
@click.option(
    "--data-dir",
    "-d",
    type=click.Path(exists=True, path_type=Path),
    default="./data",
    help="Root directory containing documents to ingest recursively",
)
@click.option(
    "--force-reindex", "-f", is_flag=True, help="Force reindexing of all documents"
)
@click.option(
    "--clear-existing",
    "-c",
    is_flag=True,
    help="Clear existing documents before ingestion",
)
@click.option(
    "--max-depth", type=int, default=None, help="Maximum directory depth to recurse"
)
@click.option(
    "--exclude-dirs",
    multiple=True,
    help="Directory names to exclude (can be used multiple times)",
)
def ingest(
    data_dir: Path,
    force_reindex: bool,
    clear_existing: bool,
    max_depth: int,
    exclude_dirs: tuple,
):
    """Ingest documents from the specified directory and all subdirectories recursively."""

    async def run_ingestion():
        from src.ingestion.ingest_documents import DocumentIngestionPipeline

        pipeline = DocumentIngestionPipeline()

        if clear_existing:
            logger.info("Clearing existing documents...")
            pipeline.store.clear_all_documents()

        # Log ingestion parameters
        logger.info(f"Starting recursive ingestion:")
        logger.info(f"  Root directory: {data_dir}")
        logger.info(f"  Max depth: {max_depth if max_depth else 'unlimited'}")
        logger.info(
            f"  Excluded directories: {list(exclude_dirs) if exclude_dirs else 'none'}"
        )

        await pipeline.ingest_directory(
            data_dir, force_reindex, max_depth, exclude_dirs
        )

    try:
        asyncio.run(run_ingestion())
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        print(f"Error: {e}")


@cli.command()
@click.option("--query", "-q", required=True, help="Search query")
@click.option("--max-results", "-n", default=10, help="Maximum number of results")
@click.option("--file-types", help="Comma-separated list of file types to filter by")
def search(query: str, max_results: int, file_types: str):
    """Search indexed documents."""
    from src.ingestion.document_store import DocumentStore

    store = DocumentStore()

    file_type_list = None
    if file_types:
        file_type_list = [ft.strip() for ft in file_types.split(",")]

    results = store.search_documents(
        query=query, n_results=max_results, file_types=file_type_list
    )

    print(f"Found {len(results)} results for query: {query}")
    print("=" * 60)

    for i, result in enumerate(results, 1):
        metadata = result.get("metadata", {})
        print(f"\n{i}. {metadata.get('title', 'Untitled')}")
        print(f"   File: {metadata.get('file_path', 'Unknown')}")
        print(f"   Type: {metadata.get('file_type', 'Unknown')}")
        if result.get("distance"):
            print(f"   Relevance: {1.0 - result['distance']:.2f}")
        print(f"   Content: {result.get('content', '')[:200]}...")


@cli.command()
@click.option("--query", "-q", required=True, help="Search query")
@click.option("--max-results", "-n", default=10, help="Maximum number of results")
@click.option(
    "--include-graph/--no-graph", default=True, help="Include knowledge graph results"
)
@click.option("--include-rag/--no-rag", default=True, help="Include RAG results")
@click.option("--entity-boost", default=1.5, help="Boost factor for entity matches")
def hybrid_search(
    query: str,
    max_results: int,
    include_graph: bool,
    include_rag: bool,
    entity_boost: float,
):
    """Perform hybrid search using RAG + Knowledge Graph."""

    async def run_hybrid_search():
        from src.knowledge_graph.hybrid_retriever import HybridRetriever

        retriever = HybridRetriever()

        results = await retriever.search(
            query=query,
            n_results=max_results,
            include_graph=include_graph,
            include_rag=include_rag,
            entity_boost=entity_boost,
        )

        print(f"Found {len(results)} hybrid results for query: {query}")
        print("=" * 60)

        for i, result in enumerate(results, 1):
            print(
                f"\n{i}. [{result.source_type.upper()}] Score: {result.relevance_score:.3f}"
            )
            print(f"   File: {result.metadata.get('file_path', 'Unknown')}")
            if result.entities:
                entities = [
                    f"{e.get('text', '')} ({e.get('label', '')})"
                    for e in result.entities[:3]
                ]
                print(f"   Entities: {', '.join(entities)}")
            if result.relationships:
                rels = [
                    f"{r.get('source', '')} → {r.get('target', '')}"
                    for r in result.relationships[:2]
                ]
                print(f"   Relations: {'; '.join(rels)}")
            print(f"   Content: {result.content[:200]}...")

        retriever.close()

    try:
        asyncio.run(run_hybrid_search())
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        print(f"Error: {e}")


@cli.command()
@click.option("--entity", "-e", required=True, help="Entity text to expand")
@click.option("--max-depth", default=2, help="Maximum relationship depth")
def expand_entity(entity: str, max_depth: int):
    """Get expanded information about an entity from the knowledge graph."""

    async def run_entity_expansion():
        from src.knowledge_graph.hybrid_retriever import HybridRetriever

        retriever = HybridRetriever()

        expansion = await retriever.get_entity_expansion(entity, max_depth=max_depth)

        if not expansion:
            print(f"No information found for entity: {entity}")
            return

        print(f"Entity Expansion for: {entity}")
        print("=" * 60)

        if expansion.get("related_entities"):
            print(f"\nRelated Entities ({len(expansion['related_entities'])}):")
            for rel_entity in expansion["related_entities"][:10]:
                print(
                    f"  - {rel_entity.get('text', '')} ({rel_entity.get('label', '')}) "
                    f"[confidence: {rel_entity.get('confidence', 0):.2f}]"
                )

        if expansion.get("relationships"):
            print(f"\nRelationships ({len(expansion['relationships'])}):")
            for rel in expansion["relationships"][:10]:
                print(
                    f"  - {rel.get('source_entity', '')} → {rel.get('relationship_type', '')} → {rel.get('target_entity', '')}"
                )

        if expansion.get("contexts"):
            print(f"\nDocument Contexts ({len(expansion['contexts'])}):")
            for ctx in expansion["contexts"][:5]:
                print(
                    f"  - {ctx.get('document_id', '')} (chunk {ctx.get('chunk_id', '')})"
                )

        retriever.close()

    try:
        asyncio.run(run_entity_expansion())
    except Exception as e:
        logger.error(f"Entity expansion failed: {e}")
        print(f"Error: {e}")


@cli.command()
def kg_stats():
    """Show knowledge graph statistics."""
    try:
        from src.knowledge_graph.graph_store import KnowledgeGraphStore

        graph_store = KnowledgeGraphStore()
        stats = graph_store.get_graph_statistics()
        graph_store.close()

        print("Knowledge Graph Statistics")
        print("=" * 40)
        print(f"Total Entities: {stats.get('total_entities', 0)}")
        print(f"Total Relationships: {stats.get('total_relationships', 0)}")
        print(f"Total Documents: {stats.get('total_documents', 0)}")

        if stats.get("entity_types"):
            print(f"\nEntity Type Distribution:")
            for entity_type in stats["entity_types"][:10]:
                print(f"  {entity_type['type']}: {entity_type['count']}")

    except Exception as e:
        logger.error(f"Failed to get knowledge graph statistics: {e}")
        print(f"Error: {e}")


@cli.command()
def rebuild_kg():
    """Rebuild the knowledge graph from existing documents."""

    async def run_rebuild():
        from src.ingestion.ingest_documents import DocumentIngestionPipeline

        pipeline = DocumentIngestionPipeline(enable_knowledge_graph=True)

        print("Rebuilding knowledge graph from existing documents...")
        result = await pipeline.rebuild_knowledge_graph()

        if result.get("success"):
            print("✅ Knowledge graph rebuild completed successfully!")
            stats = result.get("knowledge_graph_stats", {})
            print(
                f"Final graph: {stats.get('total_entities', 0)} entities, {stats.get('total_relationships', 0)} relationships"
            )
        else:
            print(
                f"❌ Knowledge graph rebuild failed: {result.get('error', 'Unknown error')}"
            )

        pipeline.close()

    try:
        asyncio.run(run_rebuild())
    except Exception as e:
        logger.error(f"Knowledge graph rebuild failed: {e}")
        print(f"Error: {e}")


@cli.command()
def test():
    """Run system tests."""

    async def run_tests():
        from scripts.quick_test import main as test_main

        await test_main()

    asyncio.run(run_tests())


@cli.command()
@click.option("--queries-file", help="JSON file containing test queries")
@click.option("--output", "-o", help="Output file for evaluation results")
def evaluate(queries_file: str, output: str):
    """Evaluate system performance."""

    async def run_evaluation():
        from src.evaluation.evaluate_system import SystemEvaluator, DEFAULT_TEST_QUERIES

        evaluator = SystemEvaluator()

        # Load queries
        if queries_file:
            with open(queries_file, "r") as f:
                test_queries = json.load(f)
        else:
            test_queries = DEFAULT_TEST_QUERIES

        print(f"Running evaluation with {len(test_queries)} queries...")
        result = await evaluator.evaluate_query_set(test_queries)

        # Display results
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)

        print(f"Queries evaluated: {len(result.test_queries)}")
        print("\nAverage Metrics:")
        for metric, value in result.average_metrics.items():
            print(f"  {metric}: {value}")

        print("\nRecommendations:")
        for rec in result.recommendations:
            print(f"  • {rec}")

        # Save results
        output_file = output or "evaluation_results.json"
        with open(output_file, "w") as f:
            json.dump(result.__dict__, f, indent=2, default=str)

        print(f"\nDetailed results saved to: {output_file}")

    asyncio.run(run_evaluation())


@cli.command()
@click.option("--text", "-t", required=True, help="Text to extract entities from")
@click.option("--domain", "-d", help="Domain context for better extraction")
@click.option(
    "--use-hybrid/--llm-only", default=True, help="Use hybrid (spaCy + LLM) or LLM only"
)
@click.option(
    "--output-format",
    type=click.Choice(["json", "text"]),
    default="text",
    help="Output format",
)
def extract_entities(text: str, domain: str, use_hybrid: bool, output_format: str):
    """Extract entities and relationships from text using LLM."""

    async def run_extraction():
        from src.knowledge_graph.llm_entity_extractor import extract_with_llm

        print(
            f"Extracting entities from text using {'hybrid' if use_hybrid else 'LLM-only'} approach..."
        )

        entities, relationships = await extract_with_llm(
            text=text, domain_context=domain, use_hybrid=use_hybrid
        )

        if output_format == "json":
            # JSON output
            result = {
                "entities": [
                    {
                        "text": e.text,
                        "label": e.label,
                        "confidence": e.confidence,
                        "normalized": e.normalized_text,
                    }
                    for e in entities
                ],
                "relationships": [
                    {
                        "source": r.source_entity.text,
                        "target": r.target_entity.text,
                        "type": r.relation_type,
                        "confidence": r.confidence,
                        "context": r.context,
                    }
                    for r in relationships
                ],
            }
            print(json.dumps(result, indent=2))
        else:
            # Text output
            print(
                f"\n🔍 Found {len(entities)} entities and {len(relationships)} relationships"
            )
            print("=" * 60)

            if entities:
                print("\n📋 ENTITIES:")
                for i, entity in enumerate(entities, 1):
                    print(
                        f"{i:2d}. {entity.text} ({entity.label}) [confidence: {entity.confidence:.2f}]"
                    )

            if relationships:
                print("\n🔗 RELATIONSHIPS:")
                for i, rel in enumerate(relationships, 1):
                    print(
                        f"{i:2d}. {rel.source_entity.text} → {rel.relation_type} → {rel.target_entity.text}"
                    )
                    print(f"    Context: {rel.context[:100]}...")
                    print(f"    Confidence: {rel.confidence:.2f}")

    try:
        asyncio.run(run_extraction())
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        print(f"Error: {e}")


@cli.command()
def setup():
    """Set up the development environment."""
    from scripts.setup import setup_environment, check_dependencies

    success = setup_environment() and check_dependencies()

    if success:
        print("\n🎉 Setup complete! Next steps:")
        print("1. Configure your .env file with API keys")
        print("2. Run document ingestion: python -m src.main ingest")
        print("3. Test the system: python -m src.main test")
        print("4. Start researching: python -m src.main research -q 'your query'")
    else:
        print("\n❌ Setup failed. Please fix the issues above.")


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    cli()
