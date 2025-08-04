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
    ) -> List[SearchResult]:
        """Search for documents using semantic similarity.

        Args:
            query: Search query text
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            filters: Optional filters to apply (e.g., {"sheet_name": "data"})

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
