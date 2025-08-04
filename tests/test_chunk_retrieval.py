"""
Test script for evaluating chunk retrieval accuracy.

This test fetches random chunks from the vector database, uses OpenAI GPT-4
to generate queries from those chunks, and then tests if those generated
queries can successfully retrieve the original chunks.
"""

import os
import sys
import logging
import random
import argparse
import statistics
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from openai import OpenAI

# Load environment variables
load_dotenv()


@dataclass
class ChunkRetrievalResult:
    """Result of chunk retrieval test."""

    original_chunk_id: str
    original_chunk_content: str
    generated_query: str
    retrieved_chunks: List[Dict[str, Any]]
    success: bool
    top_score: float
    rank_of_original: Optional[int] = None


@dataclass
class TestMetrics:
    """Overall test metrics."""

    total_tests: int
    successful_retrievals: int
    success_rate: float
    average_top_score: float
    average_rank: float
    top_1_accuracy: float
    top_3_accuracy: float
    top_5_accuracy: float


class ChunkRetrievalTester:
    """Tests chunk retrieval accuracy using generated queries."""

    def __init__(self, collection_name: str = None):
        """Initialize the tester with environment configuration."""
        self.logger = self._setup_logging()

        # Setup OpenAI client
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embedding_model = os.getenv(
            "OPENAI_INGESTION_EMBEDDING_MODEL", "text-embedding-3-large"
        )
        self.llm_model = "gpt-4o"  # Using GPT-4 as requested (closest available model)

        # Setup Qdrant client
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )

        # Collection to test - try multiple possible collection names
        self.collection_name = (
            collection_name
            or os.getenv("QDRANT_SEMANTIC_COLLECTION_NAME")
            or os.getenv("QDRANT_COLLECTION_NAME")
            or "semantic_redesign"
        )

        self.logger.info("🔧 Initialized ChunkRetrievalTester")
        self.logger.info(f"  - Collection: {self.collection_name}")
        self.logger.info(f"  - LLM Model: {self.llm_model}")
        self.logger.info(f"  - Embedding Model: {self.embedding_model}")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("logs/chunk_retrieval_test.log"),
                logging.StreamHandler(),
            ],
        )

        return logging.getLogger(__name__)

    def fetch_random_chunks(
        self, count: int = 20, file_extensions: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Fetch random chunks from the vector database.

        Args:
            count: Number of chunks to fetch
            file_extensions: Optional list of file extension filters (e.g., [".pdf", ".docx", ".xlsx"])
        """
        try:
            filter_info = ""
            if file_extensions:
                if len(file_extensions) == 1:
                    filter_info = f" with file extension '{file_extensions[0]}'"
                else:
                    filter_info = f" with file extensions {file_extensions}"

            self.logger.info(
                f"📦 Fetching {count} random chunks from collection: {self.collection_name}{filter_info}"
            )

            # Prepare filter if file extensions are specified
            scroll_filter = None
            if file_extensions:
                # Create OR conditions for multiple file extensions
                conditions = [
                    FieldCondition(key="file_extension", match=MatchValue(value=ext))
                    for ext in file_extensions
                ]
                scroll_filter = Filter(should=conditions)

            # Use scroll to get chunks from the collection
            scroll_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=count * 3,  # Get more than needed to allow for filtering
                scroll_filter=scroll_filter,
                with_payload=True,
                with_vectors=False,
            )

            if not scroll_result[0]:
                error_msg = f"No chunks found in collection: {self.collection_name}"
                if file_extensions:
                    if len(file_extensions) == 1:
                        error_msg += f" with file extension '{file_extensions[0]}'"
                    else:
                        error_msg += f" with file extensions {file_extensions}"
                raise ValueError(error_msg)

            # Filter chunks to ensure they have content
            valid_chunks = []
            for point in scroll_result[0]:
                content = (
                    point.payload.get("content")
                    or point.payload.get("page_content")
                    or point.payload.get("original_content")
                    or ""
                )

                # Only include chunks with substantial content
                if len(content.strip()) > 100:
                    chunk_data = {
                        "id": point.id,
                        "content": content,
                        "metadata": point.payload,
                    }
                    valid_chunks.append(chunk_data)

            if len(valid_chunks) < count:
                warning_msg = (
                    f"⚠️  Only found {len(valid_chunks)} valid chunks, requested {count}"
                )
                if file_extensions:
                    if len(file_extensions) == 1:
                        warning_msg += f" with file extension '{file_extensions[0]}'"
                    else:
                        warning_msg += f" with file extensions {file_extensions}"
                self.logger.warning(warning_msg)
                return valid_chunks

            # Randomly sample the requested number of chunks
            random_chunks = random.sample(valid_chunks, count)

            self.logger.info(
                f"✅ Successfully fetched {len(random_chunks)} random chunks"
            )
            return random_chunks

        except Exception as e:
            self.logger.error(f"❌ Failed to fetch random chunks: {e}")
            raise

    def generate_query_from_chunks(
        self, chunk1: Dict[str, Any], chunk2: Dict[str, Any]
    ) -> str:
        """Generate a search query using GPT-4 based on two chunks."""
        try:
            content1 = chunk1["content"][:1000]  # Limit content to avoid token limits
            content2 = chunk2["content"][:1000]

            prompt = f"""
You are tasked with creating a search query that would help retrieve specific document chunks. 
I will provide you with two text chunks, and you need to generate a search query that would 
be likely to find at least one of these chunks in a semantic search.

Chunk 1:
{content1}

Chunk 2:
{content2}

Generate a focused search query (1-2 sentences) that captures the key concepts, topics, or 
information from one or both of these chunks. The query should be something a user might 
naturally search for when looking for this type of information.

Return only the search query, no explanation.
"""

            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150,
            )

            query = response.choices[0].message.content.strip()

            # Remove quotes if the model wrapped the query in them
            if query.startswith('"') and query.endswith('"'):
                query = query[1:-1]
            if query.startswith("'") and query.endswith("'"):
                query = query[1:-1]

            self.logger.debug(f"Generated query: {query}")
            return query

        except Exception as e:
            self.logger.error(f"❌ Failed to generate query: {e}")
            raise

    def search_with_query(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for chunks using the generated query."""
        try:
            # Generate embedding for the query
            response = self.openai_client.embeddings.create(
                input=query, model=self.embedding_model
            )
            query_vector = response.data[0].embedding

            # Search in Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

            # Format results
            results = []
            for result in search_results:
                content = (
                    result.payload.get("content")
                    or result.payload.get("page_content")
                    or result.payload.get("original_content")
                    or ""
                )

                results.append(
                    {
                        "id": result.id,
                        "score": result.score,
                        "content": content,
                        "metadata": result.payload,
                    }
                )

            return results

        except Exception as e:
            self.logger.error(f"❌ Search failed: {e}")
            raise

    def test_single_retrieval(
        self, chunk1: Dict[str, Any], chunk2: Dict[str, Any]
    ) -> ChunkRetrievalResult:
        """Test retrieval for a single pair of chunks."""
        try:
            # Generate query from the two chunks
            query = self.generate_query_from_chunks(chunk1, chunk2)

            # Search using the generated query
            search_results = self.search_with_query(query)

            # Check if either original chunk was retrieved
            target_ids = {chunk1["id"], chunk2["id"]}
            retrieved_ids = [result["id"] for result in search_results]

            # Find if any target chunk was retrieved and at what rank
            success = False
            rank_of_original = None
            top_score = search_results[0]["score"] if search_results else 0.0

            for i, result_id in enumerate(retrieved_ids):
                if result_id in target_ids:
                    success = True
                    rank_of_original = i + 1  # 1-indexed rank
                    break

            # Use the first chunk as the reference for logging
            original_chunk_id = chunk1["id"]
            original_content = (
                chunk1["content"][:200] + "..."
                if len(chunk1["content"]) > 200
                else chunk1["content"]
            )

            return ChunkRetrievalResult(
                original_chunk_id=original_chunk_id,
                original_chunk_content=original_content,
                generated_query=query,
                retrieved_chunks=search_results,
                success=success,
                top_score=top_score,
                rank_of_original=rank_of_original,
            )

        except Exception as e:
            self.logger.error(f"❌ Single retrieval test failed: {e}")
            # Return a failed result
            return ChunkRetrievalResult(
                original_chunk_id=str(chunk1.get("id", "unknown")),
                original_chunk_content=chunk1.get("content", "")[:200],
                generated_query="",
                retrieved_chunks=[],
                success=False,
                top_score=0.0,
            )

    def run_retrieval_test(
        self, num_tests: int = 10, file_extensions: Optional[List[str]] = None
    ) -> TestMetrics:
        """Run the complete chunk retrieval test.

        Args:
            num_tests: Number of test cases to run
            file_extensions: Optional list of file extension filters (e.g., [".pdf", ".docx", ".xlsx"])
        """
        test_info = f"🚀 Starting chunk retrieval test with {num_tests} test cases"
        if file_extensions:
            if len(file_extensions) == 1:
                test_info += f" (filtering by file extension: {file_extensions[0]})"
            else:
                test_info += f" (filtering by file extensions: {file_extensions})"
        self.logger.info(test_info)

        try:
            # Fetch random chunks (need 2x num_tests)
            chunks = self.fetch_random_chunks(
                count=num_tests * 2, file_extensions=file_extensions
            )

            if len(chunks) < num_tests * 2:
                raise ValueError(
                    f"Not enough chunks available. Need {num_tests * 2}, got {len(chunks)}"
                )

            # Run tests
            results = []
            for i in range(num_tests):
                chunk1 = chunks[i * 2]
                chunk2 = chunks[i * 2 + 1]

                self.logger.info(
                    f"🔍 Test {i + 1}/{num_tests}: Testing chunks {chunk1['id']} and {chunk2['id']}"
                )

                result = self.test_single_retrieval(chunk1, chunk2)
                results.append(result)

                # Log result
                if result.success:
                    self.logger.info(
                        f"  ✅ SUCCESS: Retrieved at rank {result.rank_of_original} (score: {result.top_score:.3f})"
                    )
                else:
                    self.logger.info(
                        "  ❌ FAILED: Original chunks not found in top 10 results"
                    )

            # Calculate metrics
            successful_tests = [r for r in results if r.success]

            metrics = TestMetrics(
                total_tests=len(results),
                successful_retrievals=len(successful_tests),
                success_rate=len(successful_tests) / len(results) if results else 0,
                average_top_score=statistics.mean([r.top_score for r in results])
                if results
                else 0,
                average_rank=statistics.mean(
                    [r.rank_of_original for r in successful_tests]
                )
                if successful_tests
                else 0,
                top_1_accuracy=len(
                    [r for r in successful_tests if r.rank_of_original == 1]
                )
                / len(results)
                if results
                else 0,
                top_3_accuracy=len(
                    [r for r in successful_tests if r.rank_of_original <= 3]
                )
                / len(results)
                if results
                else 0,
                top_5_accuracy=len(
                    [r for r in successful_tests if r.rank_of_original <= 5]
                )
                / len(results)
                if results
                else 0,
            )

            # Log results
            self._log_test_results(results, metrics)

            return metrics

        except Exception as e:
            self.logger.error(f"❌ Retrieval test failed: {e}")
            raise

    def _log_test_results(
        self, results: List[ChunkRetrievalResult], metrics: TestMetrics
    ):
        """Log detailed test results."""
        self.logger.info("=" * 80)
        self.logger.info("📊 CHUNK RETRIEVAL TEST RESULTS")
        self.logger.info("=" * 80)

        self.logger.info(f"Total Tests: {metrics.total_tests}")
        self.logger.info(f"Successful Retrievals: {metrics.successful_retrievals}")
        self.logger.info(f"Success Rate: {metrics.success_rate:.1%}")
        self.logger.info(f"Average Top Score: {metrics.average_top_score:.3f}")
        self.logger.info(f"Average Rank (successful): {metrics.average_rank:.1f}")
        self.logger.info(f"Top-1 Accuracy: {metrics.top_1_accuracy:.1%}")
        self.logger.info(f"Top-3 Accuracy: {metrics.top_3_accuracy:.1%}")
        self.logger.info(f"Top-5 Accuracy: {metrics.top_5_accuracy:.1%}")

        self.logger.info("\n📋 DETAILED RESULTS:")
        for i, result in enumerate(results):
            status = "✅ SUCCESS" if result.success else "❌ FAILED"
            rank_info = f" (rank {result.rank_of_original})" if result.success else ""
            self.logger.info(
                f"Test {i + 1}: {status}{rank_info} | Score: {result.top_score:.3f}"
            )
            self.logger.info(f"  Query: {result.generated_query}")
            self.logger.info(f"  Original: {result.original_chunk_content[:100]}...")
            self.logger.info("")


def main():
    """Run the chunk retrieval test with command line arguments."""
    parser = argparse.ArgumentParser(description="Run chunk retrieval accuracy test")
    parser.add_argument(
        "--num-tests",
        type=int,
        default=15,
        help="Number of test cases to run (default: 15)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        help="Qdrant collection name to test (default: from environment)",
    )
    parser.add_argument(
        "--file-extensions",
        type=str,
        help="Filter chunks by file extensions (e.g., '.pdf,.docx,.xlsx' or '.pdf')",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    print("🧪 Chunk Retrieval Accuracy Test")
    print("=" * 60)
    print(f"Number of tests: {args.num_tests}")
    if args.collection:
        print(f"Collection: {args.collection}")
    if args.file_extensions:
        extensions_list = [ext.strip() for ext in args.file_extensions.split(",")]
        print(f"File extensions filter: {extensions_list}")
    else:
        extensions_list = None
    print("")

    try:
        # Initialize tester
        tester = ChunkRetrievalTester(collection_name=args.collection)

        # Run test
        metrics = tester.run_retrieval_test(
            num_tests=args.num_tests, file_extensions=extensions_list
        )

        print("\n🎯 FINAL RESULTS:")
        print(f"Success Rate: {metrics.success_rate:.1%}")
        print(f"Top-1 Accuracy: {metrics.top_1_accuracy:.1%}")
        print(f"Top-3 Accuracy: {metrics.top_3_accuracy:.1%}")
        print(f"Top-5 Accuracy: {metrics.top_5_accuracy:.1%}")
        print(f"Average Rank: {metrics.average_rank:.1f}")

        # Performance assessment
        if metrics.success_rate >= 0.8:
            print("\n🎉 EXCELLENT: Retrieval system is performing very well!")
            return 0
        elif metrics.success_rate >= 0.6:
            print("\n👍 GOOD: Retrieval system is working well.")
            return 0
        elif metrics.success_rate >= 0.4:
            print("\n⚠️  FAIR: Retrieval system is working but could be improved.")
            return 0
        else:
            print("\n❌ POOR: Retrieval system needs significant improvement.")
            return 1

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
