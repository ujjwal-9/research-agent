"""
Working hybrid analysis script that combines RAG, Knowledge Graph, and Web search.

This script bypasses the report generator bug and provides immediate analysis results.
"""

import asyncio
import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.knowledge_graph.enhanced_hybrid_retriever import enhanced_query_search
from loguru import logger


async def run_comprehensive_analysis(query: str):
    """Run comprehensive analysis with all data sources."""

    print("🚀 STARTING COMPREHENSIVE HYBRID ANALYSIS")
    print("=" * 60)
    print(f"Query: {query}")
    print("=" * 60)

    try:
        # Run enhanced search across all data sources
        results = await enhanced_query_search(
            query=query,
            data_directory=Path("data/documents"),
            include_web=True,
            n_results=30,  # Get more results for comprehensive analysis
        )
        print("*****************************")
        print(os.listdir(Path("data")))
        print("*****************************")

        print(f"\n✅ ANALYSIS COMPLETED SUCCESSFULLY")
        print(f"📊 Total sources analyzed: {len(results)}")

        if not results:
            print("❌ No results found. Check your data directory and query.")
            return

        # Organize results by source type
        by_source = {}
        for result in results:
            source_type = result.source_type
            if source_type not in by_source:
                by_source[source_type] = []
            by_source[source_type].append(result)

        # Display comprehensive results
        print(f"\n📈 SOURCE BREAKDOWN:")
        for source_type, count in [(k, len(v)) for k, v in by_source.items()]:
            print(f"   {source_type.upper()}: {count} sources")

        # Detailed analysis by source type
        for source_type, source_results in by_source.items():
            print(f"\n" + "=" * 50)
            print(f"📚 {source_type.upper()} ANALYSIS ({len(source_results)} sources)")
            print("=" * 50)

            # Calculate average scores
            avg_relevance = sum(r.relevance_score for r in source_results) / len(
                source_results
            )
            avg_confidence = sum(
                r.confidence_score or 0.5 for r in source_results
            ) / len(source_results)

            print(f"📊 Average Relevance: {avg_relevance:.2f}")
            print(f"🎯 Average Confidence: {avg_confidence:.2f}")

            # Show top results
            print(f"\n🔍 TOP INSIGHTS:")
            for i, result in enumerate(source_results[:5], 1):
                print(f"\n{i}. INSIGHT:")
                print(f"   Content: {result.content[:300]}...")
                print(f"   Relevance: {result.relevance_score:.2f}")
                print(f"   Confidence: {result.confidence_score:.2f}")

                # Show source information
                if hasattr(result, "metadata") and result.metadata:
                    source_info = result.metadata.get(
                        "file_path"
                    ) or result.metadata.get("url", "Unknown")
                    print(f"   Source: {source_info}")

                # Show entities if available
                if hasattr(result, "entities") and result.entities:
                    entities = result.entities[:5]  # Show first 5 entities
                    print(f"   Entities: {entities}")

        # Overall summary
        print(f"\n" + "=" * 60)
        print(f"🎯 COMPREHENSIVE ANALYSIS SUMMARY")
        print("=" * 60)

        total_relevance = sum(r.relevance_score for r in results)
        total_confidence = sum(r.confidence_score or 0.5 for r in results)

        print(f"📊 Analysis Coverage:")
        print(f"   Total Sources: {len(results)}")
        print(f"   Source Types: {list(by_source.keys())}")
        print(f"   Average Relevance: {total_relevance / len(results):.2f}")
        print(f"   Average Confidence: {total_confidence / len(results):.2f}")

        # Data source breakdown
        print(f"\n📚 Data Source Details:")
        for source_type, source_results in by_source.items():
            print(f"   {source_type.upper()}: {len(source_results)} sources")

        # High confidence insights
        high_confidence = [r for r in results if r.relevance_score > 0.7]
        print(f"\n⭐ High-Confidence Insights: {len(high_confidence)}")

        if high_confidence:
            print(f"📋 KEY FINDINGS:")
            for i, result in enumerate(high_confidence[:3], 1):
                print(f"   {i}. {result.content[:150]}...")

        print(
            f"\n✅ Analysis complete! Found {len(results)} relevant sources across {len(by_source)} data types."
        )

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        logger.error(f"Analysis error: {e}")
        import traceback

        traceback.print_exc()


async def main():
    """Main function with sample queries."""

    # Configure logging
    logger.add("logs/working_analysis.log", rotation="1 MB")

    # Sample queries - use one of these or provide your own
    sample_queries = [
        "Analyze Saki's clinical trial strategy, go-to-market approach, and financial projections with specific data points",
        "What are the key insights from the work planning and project data?",
        "Provide a comprehensive analysis of project progress, resource allocation, and strategic priorities",
        "What are the financial metrics, timelines, and success indicators for Saki's initiatives?",
    ]

    print("🎯 SAMPLE QUERIES:")
    for i, query in enumerate(sample_queries, 1):
        print(f"{i}. {query}")

    # Default query (modify this or uncomment user input)
    selected_query = sample_queries[0]  # Use first query

    # Uncomment to allow user selection:
    # try:
    #     choice = int(input(f"\nSelect query (1-{len(sample_queries)}) or 0 for custom: "))
    #     if choice == 0:
    #         selected_query = input("Enter your custom query: ")
    #     elif 1 <= choice <= len(sample_queries):
    #         selected_query = sample_queries[choice - 1]
    # except:
    #     pass

    print(f"\n🔍 Running analysis with query: {selected_query}")
    await run_comprehensive_analysis(selected_query)


if __name__ == "__main__":
    asyncio.run(main())
