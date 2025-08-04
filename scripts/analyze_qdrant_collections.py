#!/usr/bin/env python3
"""
Qdrant Collection Analysis Script

This script analyzes Qdrant collections to provide insights about:
- Total unique filenames across all collections
- Number of points per collection
- Number of points per unique filename
- Collection statistics and metadata

Usage:
    python scripts/analyze_qdrant_collections.py
    python scripts/analyze_qdrant_collections.py --collection-name specific_collection
    python scripts/analyze_qdrant_collections.py --detailed
"""

import argparse
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional

from qdrant_client import QdrantClient


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"logs/qdrant_analysis_{int(time.time())}.log"),
        ],
    )


def get_qdrant_client() -> QdrantClient:
    """Initialize Qdrant client from environment variables."""
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    logger = logging.getLogger(__name__)
    logger.info(f"🔌 Connecting to Qdrant at: {qdrant_url}")

    return QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
    )


def get_default_collection_names() -> List[str]:
    """Get default collection names from environment or defaults."""
    collections = []

    # Algorithmic collection
    algorithmic_collection = os.getenv("QDRANT_COLLECTION_NAME", "redesign")
    collections.append(algorithmic_collection)

    # Semantic collection
    semantic_collection = os.getenv(
        "QDRANT_SEMANTIC_COLLECTION_NAME", "semantic_redesign"
    )
    collections.append(semantic_collection)

    # Semantic large document collection
    semantic_large_collection = os.getenv(
        "QDRANT_SEMANTIC_LARGE_DOCUMENT_COLLECTION_NAME",
        "semantic_large_document_redesign",
    )
    collections.append(semantic_large_collection)

    return collections


def extract_filename_from_metadata(payload: Dict[str, Any]) -> Optional[str]:
    """Extract filename from point metadata, handling different formats."""
    # Try different metadata field names used across collections

    # Check for direct filename
    if "file_name" in payload:
        return payload["file_name"]

    # Check for document_name (algorithmic collection)
    if "document_name" in payload:
        return payload["document_name"]

    # Extract from source path
    if "source" in payload:
        source = payload["source"]
        if source:
            # Extract filename from path
            return Path(source).name

    # Extract from document_path (algorithmic collection)
    if "document_path" in payload:
        doc_path = payload["document_path"]
        if doc_path:
            return Path(doc_path).name

    return None


def analyze_collection(
    client: QdrantClient, collection_name: str, detailed: bool = False
) -> Dict[str, Any]:
    """Analyze a single Qdrant collection."""
    logger = logging.getLogger(__name__)

    try:
        # Get collection info
        collection_info = client.get_collection(collection_name)
        logger.info(f"📊 Analyzing collection: {collection_name}")

        # Initialize tracking
        filename_to_points = defaultdict(int)
        unique_filenames = set()
        total_points = 0
        sheet_names = set()

        # Scroll through all points in the collection
        offset = None
        batch_size = 1000

        while True:
            scroll_result = client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,  # We don't need the vectors, just metadata
            )

            if not scroll_result[0]:  # No more points
                break

            points = scroll_result[0]
            offset = scroll_result[1]  # Next offset

            for point in points:
                total_points += 1
                payload = point.payload

                # Extract filename
                filename = extract_filename_from_metadata(payload)
                if filename:
                    unique_filenames.add(filename)
                    filename_to_points[filename] += 1

                # Track Excel sheet names if available
                if "sheet_name" in payload and payload["sheet_name"]:
                    sheet_names.add(f"{filename}::{payload['sheet_name']}")

                if detailed and total_points % 5000 == 0:
                    logger.info(f"   Processed {total_points} points...")

            # Break if we've processed all points
            if len(points) < batch_size:
                break

        # Prepare results
        results = {
            "collection_name": collection_name,
            "total_points": total_points,
            "unique_filenames_count": len(unique_filenames),
            "unique_filenames": sorted(list(unique_filenames)),
            "points_per_filename": dict(filename_to_points),
            "collection_info": {
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "points_count": collection_info.points_count,
                "status": (
                    collection_info.status.value
                    if hasattr(collection_info.status, "value")
                    else str(collection_info.status)
                ),
            },
            "excel_sheets_count": len(sheet_names),
            "excel_sheets": sorted(list(sheet_names)) if sheet_names else [],
        }

        logger.info(
            f"✅ Collection {collection_name}: {total_points} points, {len(unique_filenames)} unique files"
        )

        return results

    except Exception as e:
        logger.error(f"❌ Error analyzing collection {collection_name}: {e}")
        return {
            "collection_name": collection_name,
            "error": str(e),
            "total_points": 0,
            "unique_filenames_count": 0,
            "unique_filenames": [],
            "points_per_filename": {},
        }


def print_analysis_results(results: List[Dict[str, Any]], detailed: bool = False):
    """Print formatted analysis results."""
    print("\n" + "=" * 80)
    print("🔍 QDRANT COLLECTION ANALYSIS RESULTS")
    print("=" * 80)

    # Overall statistics
    total_points_all = sum(r.get("total_points", 0) for r in results)
    all_unique_filenames = set()

    for result in results:
        if "unique_filenames" in result:
            all_unique_filenames.update(result["unique_filenames"])

    print("\n📈 OVERALL STATISTICS:")
    print(f"   Total Points Across All Collections: {total_points_all:,}")
    print(f"   Total Unique Filenames: {len(all_unique_filenames)}")
    print(f"   Collections Analyzed: {len(results)}")

    # Per-collection results
    for result in results:
        print(f"\n📁 COLLECTION: {result['collection_name']}")
        print("-" * 60)

        if "error" in result:
            print(f"   ❌ Error: {result['error']}")
            continue

        print(f"   Total Points: {result.get('total_points', 0):,}")
        print(f"   Unique Filenames: {result.get('unique_filenames_count', 0)}")

        # Collection info
        if "collection_info" in result:
            info = result["collection_info"]
            print(f"   Collection Status: {info.get('status', 'unknown')}")
            print(f"   Indexed Vectors: {info.get('indexed_vectors_count', 0):,}")

        # Excel sheets info
        if result.get("excel_sheets_count", 0) > 0:
            print(f"   Excel Sheets: {result['excel_sheets_count']}")

        # Top files by point count
        if result.get("points_per_filename"):
            print("\n   📊 Top 10 Files by Point Count:")
            sorted_files = sorted(
                result["points_per_filename"].items(), key=lambda x: x[1], reverse=True
            )[:10]

            for filename, count in sorted_files:
                print(f"      {filename}: {count:,} points")

        # Detailed file list
        if detailed and result.get("unique_filenames"):
            print("\n   📋 All Files in Collection:")
            for filename in sorted(result["unique_filenames"]):
                point_count = result["points_per_filename"].get(filename, 0)
                print(f"      {filename}: {point_count:,} points")

        # Excel sheets
        if detailed and result.get("excel_sheets"):
            print("\n   📄 Excel Sheets:")
            for sheet in result["excel_sheets"]:
                print(f"      {sheet}")

    # Global unique filenames
    if detailed and all_unique_filenames:
        print("\n🌍 ALL UNIQUE FILENAMES ACROSS COLLECTIONS:")
        print("-" * 60)
        for filename in sorted(all_unique_filenames):
            # Count total points for this filename across all collections
            total_points_for_file = sum(
                r.get("points_per_filename", {}).get(filename, 0) for r in results
            )
            print(f"   {filename}: {total_points_for_file:,} total points")


def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(description="Analyze Qdrant collections")
    parser.add_argument(
        "--collection-name",
        help="Specific collection name to analyze (default: analyze all default collections)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed file listings and Excel sheet information",
    )
    parser.add_argument(
        "--list-collections",
        action="store_true",
        help="List all available collections and exit",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Initialize Qdrant client
        client = get_qdrant_client()

        # List collections if requested
        if args.list_collections:
            collections = client.get_collections()
            print("\n📋 Available Collections:")
            for collection in collections.collections:
                print(f"   - {collection.name}")
            return

        # Determine which collections to analyze
        if args.collection_name:
            collections_to_analyze = [args.collection_name]
        else:
            # Get all available collections
            available_collections = client.get_collections()
            available_names = [col.name for col in available_collections.collections]

            # Filter default collections that actually exist
            default_collections = get_default_collection_names()
            collections_to_analyze = [
                col for col in default_collections if col in available_names
            ]

            if not collections_to_analyze:
                logger.warning(
                    "⚠️  No default collections found. Available collections:"
                )
                for col in available_names:
                    logger.info(f"   - {col}")
                print(
                    "\nUse --collection-name to specify a collection or --list-collections to see all available collections"
                )
                return

        logger.info(
            f"🎯 Analyzing {len(collections_to_analyze)} collections: {collections_to_analyze}"
        )

        # Analyze each collection
        results = []
        for collection_name in collections_to_analyze:
            result = analyze_collection(client, collection_name, args.detailed)
            results.append(result)

        # Print results
        print_analysis_results(results, args.detailed)

        logger.info("✅ Analysis completed successfully")

    except Exception as e:
        logger.error(f"❌ Failed to analyze collections: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
