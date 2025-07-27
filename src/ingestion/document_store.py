"""Document storage and retrieval using vector database."""

import os
from typing import List, Dict, Any, Optional
from dataclasses import asdict
import json
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
)
from openai import OpenAI
from loguru import logger

from src.config import settings
from src.ingestion.document_processor import ProcessedDocument
from src.knowledge_graph.entity_extractor import EntityExtractor
from src.knowledge_graph.llm_entity_extractor import HybridEntityExtractor
from src.knowledge_graph.graph_store import KnowledgeGraphStore


class DocumentStore:
    """Manages document storage and retrieval using Qdrant."""

    def __init__(self, enable_knowledge_graph: bool = True):
        # Initialize Qdrant client to connect to Docker server
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )

        self.collection_name = settings.qdrant_collection_name

        # Ensure collection exists
        self._ensure_collection_exists()

        # Initialize OpenAI client for embeddings
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
        self.embedding_model = settings.embedding_model

        # Initialize knowledge graph components
        self.enable_kg = enable_knowledge_graph
        if self.enable_kg:
            try:
                # Use hybrid extractor that combines spaCy and LLM
                if settings.use_llm_extraction:
                    self.entity_extractor = HybridEntityExtractor()
                    logger.info(
                        "Knowledge graph integration enabled with LLM-enhanced extraction"
                    )
                else:
                    self.entity_extractor = EntityExtractor()
                    logger.info(
                        "Knowledge graph integration enabled with spaCy extraction"
                    )

                self.graph_store = KnowledgeGraphStore()
            except Exception as e:
                logger.warning(f"Failed to initialize knowledge graph: {e}")
                self.enable_kg = False

        logger.info(
            f"Initialized document store with {self.get_document_count()} documents"
        )

    def _ensure_collection_exists(self):
        """Ensure the collection exists, create if it doesn't."""
        try:
            collections = self.client.get_collections()
            collection_names = [
                collection.name for collection in collections.collections
            ]

            if self.collection_name not in collection_names:
                # Create collection with vector configuration
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=1536,  # OpenAI embedding size for text-embedding-3-small
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Using existing Qdrant collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise

    def close(self):
        """Close the OpenAI client connection."""
        if hasattr(self.openai_client, "close"):
            self.openai_client.close()

        if self.enable_kg and hasattr(self, "graph_store"):
            self.graph_store.close()

    async def store_document(self, document: ProcessedDocument) -> bool:
        """Store a processed document in the vector database."""
        try:
            # Prepare document chunks for storage
            chunk_ids = []
            chunk_texts = []
            chunk_metadatas = []

            for i, chunk in enumerate(document.chunks):
                chunk_id = f"{document.file_path}_{i}"
                chunk_ids.append(chunk_id)
                chunk_texts.append(chunk)

                chunk_metadata = {
                    "file_path": document.file_path,
                    "file_type": document.file_type,
                    "title": document.title,
                    "chunk_index": i,
                    "total_chunks": len(document.chunks),
                    **document.metadata,
                }
                chunk_metadatas.append(chunk_metadata)

            if chunk_texts:
                # Generate embeddings using OpenAI
                embeddings = self._generate_embeddings(chunk_texts)

                # Create points for Qdrant
                points = []
                for chunk_id, text, metadata, embedding in zip(
                    chunk_ids, chunk_texts, chunk_metadatas, embeddings
                ):
                    point = PointStruct(
                        id=str(uuid.uuid4()),  # Generate unique UUID for Qdrant
                        vector=embedding,
                        payload={"chunk_id": chunk_id, "content": text, **metadata},
                    )
                    points.append(point)

                # Store in Qdrant
                self.client.upsert(collection_name=self.collection_name, points=points)

                # Extract entities and relationships for knowledge graph
                if self.enable_kg:
                    await self._process_knowledge_graph(
                        document, chunk_texts, chunk_ids
                    )

                logger.info(
                    f"Stored document {document.file_path} with {len(chunk_texts)} chunks"
                )
                return True
            else:
                logger.warning(f"No content to store for {document.file_path}")
                return False

        except Exception as e:
            logger.error(f"Error storing document {document.file_path}: {e}")
            return False

    async def _process_knowledge_graph(
        self, document: ProcessedDocument, chunk_texts: List[str], chunk_ids: List[str]
    ):
        """Process document chunks for knowledge graph extraction."""
        try:
            # Determine domain context from document metadata
            domain_context = self._get_domain_context(document)

            for i, (chunk_text, chunk_id) in enumerate(zip(chunk_texts, chunk_ids)):
                # Extract entities and relationships from chunk
                if isinstance(self.entity_extractor, HybridEntityExtractor):
                    # Use hybrid extraction with LLM
                    entities, relationships = (
                        await self.entity_extractor.extract_entities_and_relationships(
                            chunk_text,
                            chunk_id,
                            use_llm=settings.use_llm_extraction,
                            domain_context=domain_context,
                        )
                    )
                else:
                    # Use traditional spaCy extraction
                    entities, relationships = (
                        await self.entity_extractor.extract_entities_and_relationships(
                            chunk_text, chunk_id
                        )
                    )

                # Store in knowledge graph
                if entities or relationships:
                    await self.graph_store.store_entities_and_relationships(
                        entities, relationships, document.file_path, str(i)
                    )

            logger.debug(
                f"Processed knowledge graph for {document.file_path} with {len(chunk_texts)} chunks"
            )

        except Exception as e:
            logger.warning(
                f"Knowledge graph processing failed for {document.file_path}: {e}"
            )

    def _get_domain_context(self, document: ProcessedDocument) -> str:
        """Extract domain context from document for better LLM extraction."""
        context_parts = []

        # Add file type context
        if document.file_type:
            context_parts.append(f"Document type: {document.file_type}")

        # Add title context
        if document.title and document.title != document.file_path:
            context_parts.append(f"Title: {document.title}")

        # Add metadata context
        if document.metadata:
            if "author" in document.metadata and document.metadata["author"]:
                context_parts.append(f"Author: {document.metadata['author']}")

            if "subject" in document.metadata and document.metadata["subject"]:
                context_parts.append(f"Subject: {document.metadata['subject']}")

        return "; ".join(context_parts) if context_parts else None

    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        import time

        try:
            # OpenAI has a limit on batch size, so we process in chunks
            batch_size = 100
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]

                # Retry logic for API calls
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = self.openai_client.embeddings.create(
                            model=self.embedding_model, input=batch
                        )

                        batch_embeddings = [
                            embedding.embedding for embedding in response.data
                        ]
                        all_embeddings.extend(batch_embeddings)
                        break

                    except Exception as api_error:
                        if attempt == max_retries - 1:
                            raise api_error

                        logger.warning(
                            f"Embedding API attempt {attempt + 1} failed: {api_error}"
                        )
                        time.sleep(2**attempt)  # Exponential backoff

            return all_embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def search_documents(
        self, query: str, n_results: int = 10, file_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search for relevant document chunks."""
        try:
            # Generate query embedding
            query_embedding = self._generate_embeddings([query])[0]

            # Build filter for file types
            search_filter = None
            if file_types:
                search_filter = Filter(
                    must=[
                        FieldCondition(key="file_type", match=MatchAny(any=file_types))
                    ]
                )

            # Perform similarity search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=n_results,
                query_filter=search_filter,
                with_payload=True,
                with_vectors=False,
            )

            # Format results
            formatted_results = []
            for result in search_results:
                formatted_result = {
                    "content": result.payload.get("content", ""),
                    "metadata": {
                        k: v for k, v in result.payload.items() if k != "content"
                    },
                    "distance": 1.0 - result.score,  # Convert similarity to distance
                }
                formatted_results.append(formatted_result)

            logger.info(
                f"Found {len(formatted_results)} results for query: {query[:50]}..."
            )
            return formatted_results

        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

    def get_document_by_path(self, file_path: str) -> List[Dict[str, Any]]:
        """Retrieve all chunks for a specific document."""
        try:
            search_filter = Filter(
                must=[
                    FieldCondition(key="file_path", match=MatchValue(value=file_path))
                ]
            )

            # Scroll through all matching points
            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=search_filter,
                with_payload=True,
                with_vectors=False,
                limit=10000,  # Large limit to get all chunks
            )

            formatted_results = []
            for result in results:
                formatted_result = {
                    "content": result.payload.get("content", ""),
                    "metadata": {
                        k: v for k, v in result.payload.items() if k != "content"
                    },
                }
                formatted_results.append(formatted_result)

            # Sort by chunk index
            formatted_results.sort(key=lambda x: x["metadata"].get("chunk_index", 0))
            return formatted_results

        except Exception as e:
            logger.error(f"Error retrieving document {file_path}: {e}")
            return []

    def get_document_count(self) -> int:
        """Get total number of document chunks stored."""
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0

    def get_unique_documents(self) -> List[Dict[str, Any]]:
        """Get list of unique documents with metadata."""
        try:
            # Scroll through all points to collect unique documents
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                with_payload=True,
                with_vectors=False,
                limit=10000,  # Large limit to get all documents
            )

            # Group by file path
            unique_docs = {}
            for point in points:
                payload = point.payload
                file_path = payload.get("file_path")
                if file_path and file_path not in unique_docs:
                    unique_docs[file_path] = {
                        "file_path": file_path,
                        "title": payload.get("title", ""),
                        "file_type": payload.get("file_type", ""),
                        "total_chunks": payload.get("total_chunks", 0),
                    }

            return list(unique_docs.values())

        except Exception as e:
            logger.error(f"Error getting unique documents: {e}")
            return []

    def delete_document(self, file_path: str) -> bool:
        """Delete all chunks for a specific document."""
        try:
            search_filter = Filter(
                must=[
                    FieldCondition(key="file_path", match=MatchValue(value=file_path))
                ]
            )

            # Get all points for this document
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=search_filter,
                with_payload=False,
                with_vectors=False,
                limit=10000,
            )

            if points:
                point_ids = [point.id for point in points]
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(points=point_ids),
                )
                logger.info(f"Deleted document {file_path} ({len(point_ids)} chunks)")
                return True
            else:
                logger.warning(f"Document {file_path} not found")
                return False

        except Exception as e:
            logger.error(f"Error deleting document {file_path}: {e}")
            return False

    def clear_all_documents(self) -> bool:
        """Clear all documents from the store."""
        try:
            # Delete and recreate collection
            self.client.delete_collection(self.collection_name)
            self._ensure_collection_exists()
            logger.info("Cleared all documents from store")
            return True
        except Exception as e:
            logger.error(f"Error clearing documents: {e}")
            return False

    def get_documents_by_file_type(self, file_type: str) -> List[Dict[str, Any]]:
        """Get all documents of a specific file type."""
        try:
            search_filter = Filter(
                must=[
                    FieldCondition(key="file_type", match=MatchValue(value=file_type))
                ]
            )

            # Scroll through all matching points
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=search_filter,
                with_payload=True,
                with_vectors=False,
                limit=10000,  # Large limit to get all chunks
            )

            # Group by file path to get unique documents
            unique_docs = {}
            for point in points:
                payload = point.payload
                file_path = payload.get("file_path")
                if file_path and file_path not in unique_docs:
                    unique_docs[file_path] = {
                        "file_path": file_path,
                        "title": payload.get("title", ""),
                        "file_type": payload.get("file_type", ""),
                        "total_chunks": payload.get("total_chunks", 0),
                        "point_id": point.id,  # Keep track for potential deletion
                    }

            result = list(unique_docs.values())
            logger.info(
                f"Found {len(result)} unique documents with file type: {file_type}"
            )
            return result

        except Exception as e:
            logger.error(f"Error getting documents by file type {file_type}: {e}")
            return []

    def delete_documents_by_file_type(self, file_type: str) -> Dict[str, Any]:
        """Delete all documents of a specific file type from the vector database."""
        try:
            deletion_stats = {"documents_deleted": 0, "chunks_deleted": 0, "errors": []}

            search_filter = Filter(
                must=[
                    FieldCondition(key="file_type", match=MatchValue(value=file_type))
                ]
            )

            # Get all points for documents of this file type
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=search_filter,
                with_payload=True,
                with_vectors=False,
                limit=10000,
            )

            if points:
                # Group by file path to count unique documents
                unique_file_paths = set()
                point_ids = []

                for point in points:
                    point_ids.append(point.id)
                    file_path = point.payload.get("file_path")
                    if file_path:
                        unique_file_paths.add(file_path)

                # Delete all points
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(points=point_ids),
                )

                deletion_stats["documents_deleted"] = len(unique_file_paths)
                deletion_stats["chunks_deleted"] = len(point_ids)

                logger.info(
                    f"Deleted {len(unique_file_paths)} documents ({len(point_ids)} chunks) with file type: {file_type}"
                )

            else:
                logger.info(f"No documents found with file type: {file_type}")

            return deletion_stats

        except Exception as e:
            error_msg = f"Error deleting documents by file type {file_type}: {e}"
            logger.error(error_msg)
            return {"documents_deleted": 0, "chunks_deleted": 0, "errors": [error_msg]}

    def delete_documents_by_paths(self, file_paths: List[str]) -> Dict[str, Any]:
        """Delete multiple documents by their file paths from the vector database."""
        try:
            deletion_stats = {"documents_deleted": 0, "chunks_deleted": 0, "errors": []}

            for file_path in file_paths:
                try:
                    search_filter = Filter(
                        must=[
                            FieldCondition(
                                key="file_path", match=MatchValue(value=file_path)
                            )
                        ]
                    )

                    # Get all points for this document
                    points, _ = self.client.scroll(
                        collection_name=self.collection_name,
                        scroll_filter=search_filter,
                        with_payload=False,
                        with_vectors=False,
                        limit=10000,
                    )

                    if points:
                        point_ids = [point.id for point in points]
                        self.client.delete(
                            collection_name=self.collection_name,
                            points_selector=models.PointIdsList(points=point_ids),
                        )

                        deletion_stats["documents_deleted"] += 1
                        deletion_stats["chunks_deleted"] += len(point_ids)

                        logger.debug(
                            f"Deleted document {file_path} ({len(point_ids)} chunks)"
                        )
                    else:
                        logger.warning(
                            f"Document {file_path} not found in vector database"
                        )

                except Exception as e:
                    error_msg = f"Error deleting {file_path} from vector database: {e}"
                    logger.error(error_msg)
                    deletion_stats["errors"].append(error_msg)

            logger.info(f"Vector database deletion completed: {deletion_stats}")
            return deletion_stats

        except Exception as e:
            error_msg = f"Error in bulk document deletion: {e}"
            logger.error(error_msg)
            return {"documents_deleted": 0, "chunks_deleted": 0, "errors": [error_msg]}
