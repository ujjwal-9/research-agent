#!/usr/bin/env python3
"""
Benchmark script to compare sequential vs parallel processing performance.
"""

import argparse
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

from src.ingestion_semantic import SemanticIngestionPipeline


def create_test_documents(num_docs: int, doc_size: str = "small") -> str:
    """Create test documents for benchmarking."""
    temp_dir = tempfile.mkdtemp(prefix="benchmark_docs_")

    # Define content based on size
    content_sizes = {
        "small": "This is a small test document. " * 50,  # ~1.5KB
        "medium": "This is a medium test document. " * 200,  # ~6KB
        "large": "This is a large test document. " * 1000,  # ~30KB
    }

    content = content_sizes.get(doc_size, content_sizes["small"])

    print(f"📝 Creating {num_docs} {doc_size} test documents in {temp_dir}")

    for i in range(num_docs):
        doc_path = Path(temp_dir) / f"test_doc_{i:03d}.txt"
        with open(doc_path, "w") as f:
            f.write(f"Test Document {i}\n\n{content}")

    return temp_dir


def run_benchmark(
    test_dir: str,
    parallel: bool,
    max_parallel_docs: int = 4,
    embedding_batch_size: int = 20,
) -> Dict[str, Any]:
    """Run a benchmark test."""
    print(f"\n🏃 Running {'parallel' if parallel else 'sequential'} benchmark...")

    # Initialize pipeline
    pipeline = SemanticIngestionPipeline()

    # Configure parallel settings
    if parallel:
        pipeline.config._enable_parallel_processing = True
        pipeline.config._max_parallel_documents = max_parallel_docs
        pipeline.config._embedding_batch_size = embedding_batch_size
    else:
        pipeline.config._enable_parallel_processing = False

    # Start timing
    start_time = time.time()

    # Process documents
    results = pipeline.ingest_directory(
        directory_path=test_dir,
        recursive=False,
        include_vision=False,  # Disable vision for consistent timing
        apply_contextual_retrieval=True,
        store_embeddings=False,  # Disable storage for pure processing benchmark
    )

    # Calculate timing
    total_time = time.time() - start_time

    return {
        "mode": "parallel" if parallel else "sequential",
        "total_time": total_time,
        "total_documents": results["total_documents"],
        "successful_documents": results["successful_documents"],
        "failed_documents": results["failed_documents"],
        "total_chunks": results["total_chunks_created"],
        "docs_per_minute": (
            (results["total_documents"] / total_time) * 60 if total_time > 0 else 0
        ),
        "chunks_per_second": (
            results["total_chunks_created"] / total_time if total_time > 0 else 0
        ),
        "avg_time_per_doc": (
            total_time / results["total_documents"]
            if results["total_documents"] > 0
            else 0
        ),
        "config": {
            "max_parallel_docs": max_parallel_docs if parallel else 1,
            "embedding_batch_size": embedding_batch_size,
        },
    }


def print_results(results: Dict[str, Any]):
    """Print benchmark results."""
    print(f"\n📊 {results['mode'].upper()} RESULTS")
    print("=" * 50)
    print(f"Total time: {results['total_time']:.2f}s")
    print(
        f"Documents processed: {results['successful_documents']}/{results['total_documents']}"
    )
    print(f"Total chunks created: {results['total_chunks']}")
    print(f"Average time per document: {results['avg_time_per_doc']:.2f}s")
    print(f"Documents per minute: {results['docs_per_minute']:.1f}")
    print(f"Chunks per second: {results['chunks_per_second']:.1f}")

    if results["mode"] == "parallel":
        config = results["config"]
        print(f"Parallel documents: {config['max_parallel_docs']}")
        print(f"Embedding batch size: {config['embedding_batch_size']}")


def compare_results(sequential: Dict[str, Any], parallel: Dict[str, Any]):
    """Compare sequential vs parallel results."""
    print(f"\n🔥 PERFORMANCE COMPARISON")
    print("=" * 50)

    if sequential["total_time"] > 0:
        speedup = sequential["total_time"] / parallel["total_time"]
        time_saved = sequential["total_time"] - parallel["total_time"]
        efficiency = (speedup / parallel["config"]["max_parallel_docs"]) * 100

        print(f"Speedup factor: {speedup:.2f}x")
        print(
            f"Time saved: {time_saved:.2f}s ({(time_saved/sequential['total_time']*100):.1f}%)"
        )
        print(f"Parallel efficiency: {efficiency:.1f}%")

        print(f"\nThroughput comparison:")
        print(f"  Sequential: {sequential['docs_per_minute']:.1f} docs/min")
        print(f"  Parallel:   {parallel['docs_per_minute']:.1f} docs/min")
        print(
            f"  Improvement: {(parallel['docs_per_minute'] - sequential['docs_per_minute']):.1f} docs/min"
        )
    else:
        print("Unable to calculate comparison - sequential time was 0")


def main():
    """Run the benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark parallel vs sequential processing"
    )

    parser.add_argument(
        "--num-docs", type=int, default=10, help="Number of test documents to create"
    )

    parser.add_argument(
        "--doc-size",
        choices=["small", "medium", "large"],
        default="medium",
        help="Size of test documents",
    )

    parser.add_argument(
        "--max-parallel-docs",
        type=int,
        default=4,
        help="Maximum parallel documents for parallel test",
    )

    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=20,
        help="Embedding batch size for parallel test",
    )

    parser.add_argument(
        "--test-dir", help="Use existing directory instead of creating test documents"
    )

    parser.add_argument(
        "--skip-sequential",
        action="store_true",
        help="Skip sequential benchmark (only run parallel)",
    )

    parser.add_argument(
        "--skip-parallel",
        action="store_true",
        help="Skip parallel benchmark (only run sequential)",
    )

    args = parser.parse_args()

    if args.skip_sequential and args.skip_parallel:
        print("Error: Cannot skip both sequential and parallel tests")
        return

    print("🚀 SEMANTIC INGESTION BENCHMARK")
    print("=" * 50)

    # Create or use test directory
    if args.test_dir:
        test_dir = args.test_dir
        if not Path(test_dir).exists():
            print(f"Error: Test directory {test_dir} does not exist")
            return
        print(f"📁 Using existing directory: {test_dir}")
    else:
        test_dir = create_test_documents(args.num_docs, args.doc_size)

    try:
        results = {}

        # Run sequential benchmark
        if not args.skip_sequential:
            results["sequential"] = run_benchmark(test_dir, parallel=False)
            print_results(results["sequential"])

        # Run parallel benchmark
        if not args.skip_parallel:
            results["parallel"] = run_benchmark(
                test_dir,
                parallel=True,
                max_parallel_docs=args.max_parallel_docs,
                embedding_batch_size=args.embedding_batch_size,
            )
            print_results(results["parallel"])

        # Compare results
        if "sequential" in results and "parallel" in results:
            compare_results(results["sequential"], results["parallel"])

        print(f"\n💡 Optimization Tips:")
        if "parallel" in results:
            config = results["parallel"]["config"]
            print(
                f"   • Try increasing parallel docs: --max-parallel-docs {config['max_parallel_docs'] * 2}"
            )
            print(
                f"   • Try larger batch size: --embedding-batch-size {config['embedding_batch_size'] * 2}"
            )
        print(f"   • Test with different document sizes: --doc-size large")
        print(f"   • Test with more documents: --num-docs {args.num_docs * 2}")

    finally:
        # Cleanup test directory if we created it
        if not args.test_dir:
            print(f"\n🧹 Cleaning up test directory: {test_dir}")
            shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
