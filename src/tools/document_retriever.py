"""
Document retrieval tool for accessing internal documents from Qdrant.
"""

import logging
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue


@dataclass
class SearchResult:
    """Represents a search result from document retrieval."""

    content: str
    score: float
    metadata: Dict[str, Any]
    source: str
    links: List[Dict[str, str]] = None


class DocumentRetriever:
    """Tool for retrieving documents from Qdrant vector database."""

    def __init__(self, collection_name: str = None):
        """Initialize the document retriever.

        Args:
            collection_name: Qdrant collection name to search in
        """
        self.logger = logging.getLogger(__name__)
        self.qdrant_client = self._initialize_qdrant_client()
        self.collection_name = collection_name or os.getenv(
            "QDRANT_SEMANTIC_COLLECTION_NAME", "semantic_redesign"
        )

    def _initialize_qdrant_client(self) -> QdrantClient:
        """Initialize Qdrant client from environment variables."""
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        self.logger.info(f"🔌 Connecting to Qdrant at: {qdrant_url}")

        return QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=30,
        )

    def search_documents(
        self,
        query: str,
        limit: int = 20,
        score_threshold: float = 0.5,
        filters: Optional[Dict[str, Any]] = None,
        include_context: bool = True,
        context_window: int = 2,
    ) -> List[SearchResult]:
        """Search for documents using semantic similarity.

        Args:
            query: Search query text
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            filters: Optional filters to apply (e.g., {"sheet_name": "data"})
            include_context: Whether to include surrounding chunks for context
            context_window: Number of chunks before/after to include (if available)

        Returns:
            List of SearchResult objects
        """
        try:
            from openai import OpenAI

            # Generate query embedding
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            embedding_model = os.getenv(
                "OPENAI_INGESTION_EMBEDDING_MODEL", "text-embedding-3-large"
            )

            response = client.embeddings.create(input=query, model=embedding_model)
            query_vector = response.data[0].embedding

            # Build filter conditions
            filter_conditions = []
            if filters:
                for key, value in filters.items():
                    if isinstance(value, str):
                        filter_conditions.append(
                            FieldCondition(key=key, match=MatchValue(value=value))
                        )
                    elif isinstance(value, list):
                        for v in value:
                            filter_conditions.append(
                                FieldCondition(key=key, match=MatchValue(value=v))
                            )

            query_filter = Filter(must=filter_conditions) if filter_conditions else None

            # Search in Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=True,
            )

            # Convert to SearchResult objects
            results = []
            for result in search_results:
                # Try multiple possible content fields
                content = (
                    result.payload.get("page_content")
                    or result.payload.get("content")
                    or result.payload.get("original_content")
                    or ""
                )

                # If content is still empty, try to combine original_content and context
                if not content:
                    original = result.payload.get("original_content", "")
                    context = result.payload.get("context", "")
                    if original or context:
                        content = (
                            f"{context}\n\n{original}".strip() if context else original
                        )

                # Enhance content with context if requested and metadata available
                if include_context and content:
                    enhanced_content = self._get_content_with_context(
                        result.payload, content, context_window
                    )
                    if enhanced_content != content:
                        content = enhanced_content

                # Extract links from metadata if available
                links = result.payload.get("links", [])

                # Try multiple possible source fields
                source = (
                    result.payload.get("document_name")
                    or result.payload.get("file_name")
                    or result.payload.get("source")
                    or "unknown"
                )

                search_result = SearchResult(
                    content=content,
                    score=result.score,
                    metadata=result.payload,
                    source=source,
                    links=links,
                )
                results.append(search_result)

            self.logger.info(f"✅ Found {len(results)} documents for query: {query}")
            return results

        except Exception as e:
            self.logger.error(f"❌ Error searching documents: {e}")
            return []

    def get_documents_by_source(
        self, source_name: str, limit: int = 50
    ) -> List[SearchResult]:
        """Get all documents from a specific source.

        Args:
            source_name: Name of the document source
            limit: Maximum number of results

        Returns:
            List of SearchResult objects
        """
        try:
            # Scroll through all points with the given source
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="document_name", match=MatchValue(value=source_name)
                    )
                ]
            )

            points = list(
                self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=filter_condition,
                    limit=limit,
                    with_payload=True,
                )[0]
            )

            results = []
            for point in points:
                content = point.payload.get(
                    "page_content", point.payload.get("content", "")
                )
                links = point.payload.get("links", [])

                search_result = SearchResult(
                    content=content,
                    score=1.0,  # No similarity score for exact source match
                    metadata=point.payload,
                    source=source_name,
                    links=links,
                )
                results.append(search_result)

            self.logger.info(
                f"✅ Retrieved {len(results)} documents from source: {source_name}"
            )
            return results

        except Exception as e:
            self.logger.error(
                f"❌ Error retrieving documents from source {source_name}: {e}"
            )
            return []

    def get_all_sources(self) -> List[str]:
        """Get list of all available document sources.

        Returns:
            List of unique document source names
        """
        try:
            # Get a sample of points to extract unique sources
            points = list(
                self.qdrant_client.scroll(
                    collection_name=self.collection_name, limit=1000, with_payload=True
                )[0]
            )

            sources = set()
            for point in points:
                if "document_name" in point.payload:
                    sources.add(point.payload["document_name"])

            source_list = sorted(list(sources))
            self.logger.info(f"✅ Found {len(source_list)} unique document sources")
            return source_list

        except Exception as e:
            self.logger.error(f"❌ Error retrieving document sources: {e}")
            return []

    def search_with_metadata_filter(
        self,
        query: str,
        metadata_filters: Dict[str, Any],
        limit: int = 20,
        score_threshold: float = 0.5,
    ) -> List[SearchResult]:
        """Search documents with specific metadata filters.

        Args:
            query: Search query text
            metadata_filters: Dictionary of metadata key-value pairs to filter by
            limit: Maximum number of results
            score_threshold: Minimum similarity score threshold

        Returns:
            List of SearchResult objects
        """
        return self.search_documents(
            query=query,
            limit=limit,
            score_threshold=score_threshold,
            filters=metadata_filters,
        )

    def _get_content_with_context(
        self, payload: Dict[str, Any], original_content: str, context_window: int
    ) -> str:
        """Enhance content with surrounding chunks for better context.

        Args:
            payload: Metadata payload from the original chunk
            original_content: Original chunk content
            context_window: Number of chunks before/after to include

        Returns:
            Enhanced content with context or original content if context unavailable
        """
        try:
            # Extract chunk identification info
            document_name = payload.get("document_name", payload.get("file_name"))
            chunk_id = payload.get("chunk_id")
            chunk_index = payload.get("chunk_index")

            if not document_name or (chunk_id is None and chunk_index is None):
                return original_content

            # Try to get surrounding chunks
            surrounding_chunks = self._get_surrounding_chunks(
                document_name, chunk_id, chunk_index, context_window
            )

            if not surrounding_chunks:
                return original_content

            # Find the position of the original chunk
            original_chunk_pos = -1
            for i, chunk in enumerate(surrounding_chunks):
                chunk_content = (
                    chunk.get("page_content")
                    or chunk.get("content")
                    or chunk.get("original_content", "")
                )
                if chunk_content.strip() == original_content.strip():
                    original_chunk_pos = i
                    break

            if original_chunk_pos == -1:
                # If we can't find the original chunk, just return it
                return original_content

            # Assemble content with context
            content_parts = []

            # Add preceding context
            for i in range(
                max(0, original_chunk_pos - context_window), original_chunk_pos
            ):
                prev_content = (
                    surrounding_chunks[i].get("page_content")
                    or surrounding_chunks[i].get("content")
                    or surrounding_chunks[i].get("original_content", "")
                )
                if prev_content.strip():
                    content_parts.append(f"[Previous Context]: {prev_content.strip()}")

            # Add original content
            content_parts.append(f"[Main Content]: {original_content.strip()}")

            # Add following context
            for i in range(
                original_chunk_pos + 1,
                min(len(surrounding_chunks), original_chunk_pos + context_window + 1),
            ):
                next_content = (
                    surrounding_chunks[i].get("page_content")
                    or surrounding_chunks[i].get("content")
                    or surrounding_chunks[i].get("original_content", "")
                )
                if next_content.strip():
                    content_parts.append(f"[Following Context]: {next_content.strip()}")

            enhanced_content = "\n\n".join(content_parts)

            # Log context enhancement
            if len(content_parts) > 1:
                self.logger.debug(
                    f"🔍 Enhanced chunk with {len(content_parts) - 1} context pieces"
                )

            return enhanced_content

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to enhance content with context: {e}")
            return original_content

    def _get_surrounding_chunks(
        self, document_name: str, chunk_id: Any, chunk_index: Any, context_window: int
    ) -> List[Dict[str, Any]]:
        """Get surrounding chunks from the same document.

        Args:
            document_name: Name of the document
            chunk_id: ID of the chunk (if available)
            chunk_index: Index of the chunk (if available)
            context_window: Number of chunks to get before/after

        Returns:
            List of chunk payloads in order
        """
        try:
            # Build filter for same document
            filter_conditions = [
                FieldCondition(
                    key="document_name", match=MatchValue(value=document_name)
                )
            ]

            document_filter = Filter(must=filter_conditions)

            # Get all chunks from the document
            points = list(
                self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=document_filter,
                    limit=1000,  # Reasonable limit for most documents
                    with_payload=True,
                )[0]
            )

            if not points:
                return []

            # Sort chunks by chunk_index if available, otherwise try to sort by chunk_id
            sorted_chunks = []
            for point in points:
                payload = point.payload
                sort_key = payload.get("chunk_index")
                if sort_key is None:
                    # Try to extract numeric part from chunk_id if it exists
                    chunk_id_val = payload.get("chunk_id")
                    if chunk_id_val and isinstance(chunk_id_val, str):
                        try:
                            # Extract numbers from chunk_id (e.g., "chunk_5" -> 5)
                            import re

                            numbers = re.findall(r"\d+", str(chunk_id_val))
                            sort_key = int(numbers[-1]) if numbers else 0
                        except:
                            sort_key = 0
                    else:
                        sort_key = 0

                sorted_chunks.append((sort_key, payload))

            # Sort by the extracted key
            sorted_chunks.sort(key=lambda x: x[0])

            # Return just the payloads
            return [chunk[1] for chunk in sorted_chunks]

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to get surrounding chunks: {e}")
            return []
