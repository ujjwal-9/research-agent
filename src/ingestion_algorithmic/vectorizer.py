"""Vector storage and retrieval using Qdrant and OpenAI embeddings."""

import logging
import uuid
from typing import List, Dict, Any, Optional
import openai
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from .config import IngestionConfig


class VectorStore:
    """Handles vectorization and storage in Qdrant."""

    def __init__(self, config: IngestionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize OpenAI client
        openai.api_key = config.openai_api_key
        self.openai_client = openai.OpenAI(api_key=config.openai_api_key)

        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
        )

        self.collection_name = config.qdrant_collection_name
        self.embedding_model = config.openai_embedding_model

        self.logger.info(f"🗄️  VectorStore initialized:")
        self.logger.info(f"  - Qdrant URL: {config.qdrant_url}")
        self.logger.info(f"  - Collection: {self.collection_name}")
        self.logger.info(f"  - Embedding Model: {self.embedding_model}")

        # Ensure collection exists
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist."""
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name not in collection_names:
                self.logger.info(
                    f"📦 Creating Qdrant collection: {self.collection_name}"
                )

                # Create collection with OpenAI embedding dimensions
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=3072, distance=Distance.COSINE
                    ),  # text-embedding-3-large
                )

                self.logger.info(
                    f"✅ Collection '{self.collection_name}' created successfully"
                )
            else:
                self.logger.info(
                    f"✅ Collection '{self.collection_name}' already exists"
                )

        except Exception as e:
            self.logger.error(f"❌ Failed to create/verify collection: {e}")
            raise

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI."""
        try:
            response = self.openai_client.embeddings.create(
                input=text, model=self.embedding_model
            )
            return response.data[0].embedding

        except Exception as e:
            self.logger.error(f"❌ Failed to generate embedding: {e}")
            raise

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in batch."""
        try:
            self.logger.info(f"🔄 Generating embeddings for {len(texts)} texts")

            # OpenAI has limits on batch size, so we chunk the requests
            batch_size = 2048  # Conservative batch size
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]

                response = self.openai_client.embeddings.create(
                    input=batch_texts, model=self.embedding_model
                )

                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)

                self.logger.info(
                    f"📊 Processed {min(i + batch_size, len(texts))}/{len(texts)} embeddings"
                )

            return all_embeddings

        except Exception as e:
            self.logger.error(f"❌ Failed to generate batch embeddings: {e}")
            raise

    def store_chunks(
        self, chunks: List[Dict[str, Any]], document_name: str, document_path: str
    ) -> bool:
        """Store document chunks in Qdrant with metadata."""
        try:
            self.logger.info(
                f"💾 Storing {len(chunks)} chunks for document: {document_name}"
            )

            # Extract texts for embedding
            texts = [chunk["content"] for chunk in chunks]

            # Generate embeddings in batch
            embeddings = self.generate_embeddings_batch(texts)

            # Prepare points for Qdrant
            points = []

            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Create unique point ID
                point_id = str(uuid.uuid4())

                # Prepare metadata
                metadata = {
                    "document_name": document_name,
                    "document_path": document_path,
                    "content": chunk["content"],
                    **chunk["metadata"],  # Include all chunk metadata
                }

                # Create point
                point = PointStruct(id=point_id, vector=embedding, payload=metadata)

                points.append(point)

            # Upload to Qdrant in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch_points = points[i : i + batch_size]

                self.qdrant_client.upsert(
                    collection_name=self.collection_name, points=batch_points
                )

                self.logger.info(
                    f"📤 Uploaded batch {i//batch_size + 1}/{(len(points) + batch_size - 1)//batch_size}"
                )

            self.logger.info(
                f"✅ Successfully stored {len(chunks)} chunks for {document_name}"
            )
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to store chunks: {e}")
            return False

    def search(
        self, query: str, limit: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks in Qdrant."""
        try:
            # Generate query embedding
            query_embedding = self.generate_embedding(query)

            # Prepare filters if provided
            qdrant_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )

                if conditions:
                    qdrant_filter = Filter(must=conditions)

            # Search in Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=qdrant_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

            # Format results
            results = []
            for result in search_results:
                results.append(
                    {
                        "id": result.id,
                        "score": result.score,
                        "content": result.payload.get("content", ""),
                        "metadata": {
                            k: v for k, v in result.payload.items() if k != "content"
                        },
                    }
                )

            return results

        except Exception as e:
            self.logger.error(f"❌ Search failed: {e}")
            return []

    def search_by_document(
        self, query: str, document_name: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search within a specific document."""
        filters = {"document_name": document_name}
        return self.search(query, limit, filters)

    def search_by_page(
        self, query: str, document_name: str, page_number: int, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search within a specific page of a document."""
        filters = {"document_name": document_name, "page_number": page_number}
        return self.search(query, limit, filters)

    def get_document_info(self, document_name: str) -> Dict[str, Any]:
        """Get information about a stored document."""
        try:
            # Search for any chunk from this document to get metadata
            filters = {"document_name": document_name}
            qdrant_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_name", match=MatchValue(value=document_name)
                    )
                ]
            )

            # Get count and sample data
            scroll_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=qdrant_filter,
                limit=100,
                with_payload=True,
                with_vectors=False,
            )

            if not scroll_result[0]:
                return {"error": "Document not found"}

            # Analyze chunks
            chunks = scroll_result[0]
            pages = set()
            total_tokens = 0
            chunk_count = len(chunks)

            for chunk in chunks:
                payload = chunk.payload
                if "page_number" in payload:
                    pages.add(payload["page_number"])
                if "token_count" in payload:
                    total_tokens += payload["token_count"]

            return {
                "document_name": document_name,
                "chunk_count": chunk_count,
                "page_count": len(pages),
                "total_tokens": total_tokens,
                "pages": sorted(list(pages)) if pages else [],
            }

        except Exception as e:
            self.logger.error(f"❌ Failed to get document info: {e}")
            return {"error": str(e)}

    def list_documents(self) -> List[str]:
        """List all documents in the collection."""
        try:
            # Get all points and extract unique document names
            scroll_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Large limit to get all documents
                with_payload=True,
                with_vectors=False,
            )

            document_names = set()
            for point in scroll_result[0]:
                if "document_name" in point.payload:
                    document_names.add(point.payload["document_name"])

            return sorted(list(document_names))

        except Exception as e:
            self.logger.error(f"❌ Failed to list documents: {e}")
            return []

    def delete_document(self, document_name: str) -> bool:
        """Delete all chunks for a specific document."""
        try:
            self.logger.info(f"🗑️  Deleting document: {document_name}")

            # Create filter for this document
            qdrant_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_name", match=MatchValue(value=document_name)
                    )
                ]
            )

            # Delete points matching the filter
            self.qdrant_client.delete(
                collection_name=self.collection_name, points_selector=qdrant_filter
            )

            self.logger.info(f"✅ Successfully deleted document: {document_name}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to delete document: {e}")
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)

            return {
                "collection_name": self.collection_name,
                "points_count": collection_info.points_count,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "status": (
                    collection_info.status.value
                    if collection_info.status
                    else "unknown"
                ),
            }

        except Exception as e:
            self.logger.error(f"❌ Failed to get collection stats: {e}")
            return {"error": str(e)}
