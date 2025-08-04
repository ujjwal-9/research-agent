"""OpenAI text-embedding-3-large integration for semantic embeddings."""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from openai import OpenAI
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)


@dataclass
class EmbeddingResult:
    """Container for embedding results."""

    embedding: List[float]
    text: str
    metadata: Dict[str, Any]
    embedding_model: str
    embedding_time: float


class SemanticEmbeddings:
    """
    Semantic embeddings using OpenAI text-embedding-3-large.

    Provides embeddings generation and vector storage with Qdrant integration.
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize OpenAI client and embeddings
        self.client = OpenAI(api_key=config.openai_api_key)
        self.embeddings = OpenAIEmbeddings(
            model=config.embedding_model,
            openai_api_key=config.openai_api_key,
        )

        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
        )

        # Semaphore for concurrent requests
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)

        # Embedding dimension for text-embedding-3-large
        self.embedding_dimension = 3072

        self._ensure_collection_exists()
        self.logger.info("✅ Semantic embeddings initialized")

    def _ensure_collection_exists(self):
        """Ensure the Qdrant collection exists with proper configuration."""
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections().collections
            collection_names = [col.name for col in collections]

            if self.config.qdrant_collection_name not in collection_names:
                self.logger.info(
                    f"📦 Creating Qdrant collection: {self.config.qdrant_collection_name}"
                )

                self.qdrant_client.create_collection(
                    collection_name=self.config.qdrant_collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dimension,
                        distance=Distance.COSINE,
                    ),
                )

                self.logger.info(
                    f"✅ Created collection: {self.config.qdrant_collection_name}"
                )
            else:
                self.logger.info(
                    f"✅ Collection exists: {self.config.qdrant_collection_name}"
                )

        except Exception as e:
            self.logger.error(f"Error ensuring collection exists: {e}")
            raise

    def _is_excel_file(self, file_path: str) -> bool:
        """Check if the file is an Excel file based on path or metadata."""
        if not file_path:
            return False

        file_path_lower = file_path.lower()
        return (
            file_path_lower.endswith(".xlsx")
            or file_path_lower.endswith(".xls")
            or "excel" in file_path_lower
            or "spreadsheet" in file_path_lower
        )

    def _extract_sheet_name(self, metadata: Dict[str, Any]) -> Optional[str]:
        """Extract sheet name from document metadata."""
        # Try multiple possible metadata keys for sheet name
        sheet_name_keys = [
            "page_name",  # Primary key from UnstructuredExcelLoader
            "sheet_name",  # Direct sheet name if available
            "page_title",  # Alternative key
            "worksheet_name",  # Another possible key
        ]

        for key in sheet_name_keys:
            if key in metadata and metadata[key]:
                sheet_name = str(metadata[key]).strip()
                if sheet_name and sheet_name != "unknown":
                    self.logger.debug(
                        f"Found sheet name '{sheet_name}' from metadata key '{key}'"
                    )
                    return sheet_name

        # Try to extract from source path if contains sheet information
        source = metadata.get("source", "")
        if "_sheet_" in source.lower():
            # Extract from patterns like "file_sheet_SheetName.html"
            parts = source.split("_sheet_")
            if len(parts) > 1:
                sheet_part = parts[1].split(".")[0]  # Remove extension
                if sheet_part:
                    self.logger.debug(
                        f"Extracted sheet name '{sheet_part}' from source path"
                    )
                    return sheet_part

        self.logger.debug("No sheet name found in metadata")
        return None

    async def _embed_text_async(
        self, text: str, metadata: Optional[Dict] = None
    ) -> EmbeddingResult:
        """Generate embedding for a single text asynchronously."""
        async with self.semaphore:
            start_time = time.time()

            try:
                # Generate embedding using OpenAI
                response = self.client.embeddings.create(
                    model=self.config.embedding_model,
                    input=text,
                )

                embedding = response.data[0].embedding
                embedding_time = time.time() - start_time

                return EmbeddingResult(
                    embedding=embedding,
                    text=text,
                    metadata=metadata or {},
                    embedding_model=self.config.embedding_model,
                    embedding_time=embedding_time,
                )

            except Exception as e:
                self.logger.error(f"Error generating embedding: {e}")
                # Return zero embedding as fallback
                return EmbeddingResult(
                    embedding=[0.0] * self.embedding_dimension,
                    text=text,
                    metadata=metadata or {},
                    embedding_model=self.config.embedding_model,
                    embedding_time=time.time() - start_time,
                )

    def embed_text(self, text: str, metadata: Optional[Dict] = None) -> EmbeddingResult:
        """Generate embedding for a single text (synchronous wrapper)."""
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run, self._embed_text_async(text, metadata)
                )
                return future.result()
        except RuntimeError:
            return asyncio.run(self._embed_text_async(text, metadata))

    async def embed_documents_async(
        self, documents: List[Document]
    ) -> List[EmbeddingResult]:
        """Generate embeddings for multiple documents asynchronously."""
        if not documents:
            return []

        self.logger.info(f"🔮 Generating embeddings for {len(documents)} documents")

        # Process in batches to optimize API calls and memory usage
        batch_size = self.config.embedding_batch_size
        all_results = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(documents) + batch_size - 1) // batch_size

            self.logger.info(
                f"🔮 Processing batch {batch_num}/{total_batches} ({len(batch)} documents)"
            )

            # Create tasks for batch
            tasks = []
            for doc in batch:
                task = self._embed_text_async(doc.page_content, doc.metadata)
                tasks.append(task)

            # Execute batch concurrently
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process batch results
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    doc_index = i + j
                    self.logger.error(f"Error embedding document {doc_index}: {result}")
                    # Create fallback embedding result
                    result = EmbeddingResult(
                        embedding=[0.0] * self.embedding_dimension,
                        text=batch[j].page_content,
                        metadata=batch[j].metadata,
                        embedding_model=self.config.embedding_model,
                        embedding_time=0.0,
                    )

                all_results.append(result)

        total_time = sum(r.embedding_time for r in all_results)
        avg_time_per_doc = total_time / len(all_results) if all_results else 0
        self.logger.info(
            f"✅ Generated {len(all_results)} embeddings in {total_time:.2f}s (avg: {avg_time_per_doc:.2f}s/doc)"
        )

        return all_results

    def embed_documents(self, documents: List[Document]) -> List[EmbeddingResult]:
        """Generate embeddings for multiple documents (synchronous wrapper)."""
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run, self.embed_documents_async(documents)
                )
                return future.result()
        except RuntimeError:
            return asyncio.run(self.embed_documents_async(documents))

    def store_embeddings(
        self, embedding_results: List[EmbeddingResult], upsert: bool = True
    ) -> List[str]:
        """
        Store embeddings in Qdrant vector database.

        Args:
            embedding_results: List of embedding results to store
            upsert: Whether to upsert (update if exists) or insert new

        Returns:
            List of point IDs that were stored
        """
        if not embedding_results:
            return []

        self.logger.info(f"💾 Storing {len(embedding_results)} embeddings in Qdrant")

        # Prepare points for Qdrant
        points = []
        point_ids = []

        for i, result in enumerate(embedding_results):
            # Generate point ID
            import hashlib

            point_id = hashlib.md5(
                f"{result.text[:100]}_{result.metadata.get('source', 'unknown')}_{i}".encode()
            ).hexdigest()

            # Prepare metadata for storage
            storage_metadata = {
                **result.metadata,
                "embedding_model": result.embedding_model,
                "embedding_time": result.embedding_time,
                "text_length": len(result.text),
                "stored_at": time.time(),
            }

            # Handle Excel sheet names specifically
            if self._is_excel_file(result.metadata.get("source", "")):
                # Extract sheet name from Excel metadata
                sheet_name = self._extract_sheet_name(result.metadata)
                if sheet_name:
                    storage_metadata["sheet_name"] = sheet_name
                    self.logger.debug(f"Added sheet_name to metadata: {sheet_name}")

            # Create point
            point = PointStruct(
                id=point_id,
                vector=result.embedding,
                payload=storage_metadata,
            )

            points.append(point)
            point_ids.append(point_id)

        try:
            # Store points in Qdrant
            self.qdrant_client.upsert(
                collection_name=self.config.qdrant_collection_name,
                points=points,
            )

            self.logger.info(f"✅ Stored {len(points)} embeddings successfully")
            return point_ids

        except Exception as e:
            self.logger.error(f"Error storing embeddings: {e}")
            raise

    def search_similar(
        self,
        query_text: str,
        limit: int = 10,
        score_threshold: float = 0.0,
        filter_conditions: Optional[Dict[str, Any]] = None,
        sheet_name: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents in the vector database.

        Args:
            query_text: Text to search for
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            filter_conditions: Optional filter conditions
            sheet_name: Optional sheet name to filter Excel documents

        Returns:
            List of tuples (Document, similarity_score)
        """
        self.logger.info(f"🔍 Searching for similar documents: '{query_text[:100]}...'")

        # Generate embedding for query
        query_result = self.embed_text(query_text)

        # Prepare filter if provided
        query_filter = None
        conditions = []

        # Add filter conditions
        if filter_conditions:
            for key, value in filter_conditions.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )

        # Add sheet name filter for Excel files
        if sheet_name:
            conditions.append(
                FieldCondition(key="sheet_name", match=MatchValue(value=sheet_name))
            )
            self.logger.debug(f"Added sheet_name filter: {sheet_name}")

        if conditions:
            query_filter = Filter(must=conditions)

        try:
            # Search in Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.config.qdrant_collection_name,
                query_vector=query_result.embedding,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=True,
            )

            # Convert results to Documents with scores
            results = []
            for result in search_results:
                # Extract text content from payload
                text_content = result.payload.get("page_content", "")
                if not text_content:
                    # Fallback: try to find text in metadata
                    text_content = result.payload.get(
                        "text", result.payload.get("content", "")
                    )

                # Create document
                metadata = {
                    k: v for k, v in result.payload.items() if k != "page_content"
                }
                doc = Document(page_content=text_content, metadata=metadata)

                results.append((doc, result.score))

            self.logger.info(f"✅ Found {len(results)} similar documents")
            return results

        except Exception as e:
            self.logger.error(f"Error searching similar documents: {e}")
            return []

    def search_by_sheet_name(
        self,
        query_text: str,
        sheet_name: str,
        limit: int = 10,
        score_threshold: float = 0.0,
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents within a specific Excel sheet.

        Args:
            query_text: Text to search for
            sheet_name: Name of the Excel sheet to search within
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            List of tuples (Document, similarity_score)
        """
        self.logger.info(
            f"🔍 Searching in sheet '{sheet_name}': '{query_text[:100]}...'"
        )
        return self.search_similar(
            query_text=query_text,
            limit=limit,
            score_threshold=score_threshold,
            sheet_name=sheet_name,
        )

    def delete_by_source(self, source: str) -> int:
        """
        Delete all documents from a specific source.

        Args:
            source: Source identifier to delete

        Returns:
            Number of documents deleted
        """
        try:
            # Create filter for source
            source_filter = Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source))]
            )

            # Delete points
            result = self.qdrant_client.delete(
                collection_name=self.config.qdrant_collection_name,
                points_selector=source_filter,
            )

            self.logger.info(f"🗑️  Deleted documents from source: {source}")
            return result.operation_id or 0

        except Exception as e:
            self.logger.error(f"Error deleting documents by source: {e}")
            return 0

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the vector collection."""
        try:
            info = self.qdrant_client.get_collection(self.config.qdrant_collection_name)
            return {
                "collection_name": self.config.qdrant_collection_name,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "status": info.status,
                "optimizer_status": info.optimizer_status,
                "embedding_dimension": self.embedding_dimension,
                "embedding_model": self.config.embedding_model,
            }
        except Exception as e:
            self.logger.error(f"Error getting collection info: {e}")
            return {"error": str(e)}

    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding configuration and statistics."""
        return {
            "embedding_model": self.config.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "qdrant_url": self.config.qdrant_url,
            "collection_name": self.config.qdrant_collection_name,
            "max_concurrent_requests": self.config.max_concurrent_requests,
            "excel_sheet_tracking": True,  # Indicates sheet names are tracked for Excel files
            "supported_metadata_fields": [
                "source",
                "file_name",
                "file_extension",
                "file_size_bytes",
                "sheet_name",
                "embedding_model",
                "embedding_time",
                "text_length",
                "stored_at",
            ],
            "collection_info": self.get_collection_info(),
        }
