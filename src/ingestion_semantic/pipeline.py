"""Complete semantic ingestion pipeline orchestrating all components."""

import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from .config import SemanticIngestionConfig
from .document_processor import SemanticDocumentProcessor


class SemanticIngestionPipeline:
    """
    Complete semantic ingestion pipeline.

    Orchestrates the entire semantic document processing workflow:
    1. Document loading with LangChain
    2. Contextual retrieval for better chunk semantics
    3. Vision processing for images/charts/tables
    4. Semantic embeddings with OpenAI text-embedding-3-large
    5. Vector storage in Qdrant
    """

    def __init__(self, config: Optional[SemanticIngestionConfig] = None):
        """
        Initialize the semantic ingestion pipeline.

        Args:
            config: Optional configuration object. If None, creates default config.
        """
        self.config = config or SemanticIngestionConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize the main document processor
        self.processor = SemanticDocumentProcessor(self.config)

        self.logger.info("🚀 Semantic Ingestion Pipeline initialized")
        self.logger.info(f"📊 Configuration: {self.config.qdrant_collection_name}")
        self.logger.info(
            f"🤖 Contextual Model: {self.config.contextual_retrieval_model}"
        )
        self.logger.info(f"🔮 Embedding Model: {self.config.embedding_model}")
        self.logger.info(f"👁️  Vision Model: {self.config.vision_model}")

    def ingest_document(
        self,
        file_path: str,
        include_vision: bool = True,
        apply_contextual_retrieval: bool = True,
        store_embeddings: bool = True,
    ) -> Dict[str, Any]:
        """
        Ingest a single document through the semantic pipeline.

        Args:
            file_path: Path to the document file
            include_vision: Whether to process images/charts/tables with vision models
            apply_contextual_retrieval: Whether to apply contextual retrieval for better chunks
            store_embeddings: Whether to store embeddings in vector database

        Returns:
            Dictionary with ingestion results and statistics
        """
        self.logger.info(f"📄 Ingesting document: {Path(file_path).name}")

        return self.processor.process_document(
            file_path=file_path,
            include_vision=include_vision,
            apply_contextual_retrieval=apply_contextual_retrieval,
            store_embeddings=store_embeddings,
        )

    def ingest_documents(
        self,
        file_paths: List[str],
        include_vision: bool = True,
        apply_contextual_retrieval: bool = True,
        store_embeddings: bool = True,
        continue_on_error: bool = True,
    ) -> Dict[str, Any]:
        """
        Ingest multiple documents through the semantic pipeline.

        Args:
            file_paths: List of document file paths
            include_vision: Whether to process images/charts/tables with vision models
            apply_contextual_retrieval: Whether to apply contextual retrieval for better chunks
            store_embeddings: Whether to store embeddings in vector database
            continue_on_error: Whether to continue processing if one document fails

        Returns:
            Dictionary with batch ingestion results and statistics
        """
        start_time = time.time()

        self.logger.info(f"📚 Starting batch ingestion of {len(file_paths)} documents")

        results = self.processor.process_multiple_documents(
            file_paths=file_paths,
            include_vision=include_vision,
            apply_contextual_retrieval=apply_contextual_retrieval,
            store_embeddings=store_embeddings,
        )

        # Calculate summary statistics
        successful = [r for r in results if r.get("success", False)]
        failed = [r for r in results if not r.get("success", False)]

        total_processing_time = time.time() - start_time
        total_chunks = sum(
            r.get("statistics", {}).get("total_chunks_created", 0) for r in successful
        )
        total_embeddings = sum(
            r.get("statistics", {}).get("embeddings_stored", 0) for r in successful
        )

        summary = {
            "success": True,
            "batch_processing_time": total_processing_time,
            "total_documents": len(file_paths),
            "successful_documents": len(successful),
            "failed_documents": len(failed),
            "total_chunks_created": total_chunks,
            "total_embeddings_stored": total_embeddings,
            "settings": {
                "include_vision": include_vision,
                "apply_contextual_retrieval": apply_contextual_retrieval,
                "store_embeddings": store_embeddings,
            },
            "individual_results": results,
            "failed_documents_details": [
                {
                    "file": r.get("file_name", "unknown"),
                    "error": r.get("error", "unknown"),
                }
                for r in failed
            ],
        }

        self.logger.info(
            f"✅ Batch ingestion completed in {total_processing_time:.2f}s"
        )
        self.logger.info(
            f"📊 Success rate: {len(successful)}/{len(file_paths)} documents"
        )
        self.logger.info(f"📄 Total chunks: {total_chunks}")
        self.logger.info(f"🔮 Total embeddings: {total_embeddings}")

        return summary

    def ingest_directory(
        self,
        directory_path: str,
        file_patterns: Optional[List[str]] = None,
        recursive: bool = True,
        include_vision: bool = True,
        apply_contextual_retrieval: bool = True,
        store_embeddings: bool = True,
    ) -> Dict[str, Any]:
        """
        Ingest all supported documents in a directory.

        Args:
            directory_path: Path to directory containing documents
            file_patterns: List of file patterns to match (e.g., ['*.pdf', '*.docx'])
            recursive: Whether to search subdirectories recursively
            include_vision: Whether to process images/charts/tables with vision models
            apply_contextual_retrieval: Whether to apply contextual retrieval for better chunks
            store_embeddings: Whether to store embeddings in vector database

        Returns:
            Dictionary with directory ingestion results and statistics
        """
        self.logger.info(f"📂 Ingesting directory: {directory_path}")

        results = self.processor.process_directory(
            directory_path=directory_path,
            file_patterns=file_patterns,
            recursive=recursive,
            include_vision=include_vision,
            apply_contextual_retrieval=apply_contextual_retrieval,
            store_embeddings=store_embeddings,
        )

        # Convert to batch format for consistency
        file_paths = [r.get("file_path", "") for r in results]

        return self.ingest_documents(
            file_paths=file_paths,
            include_vision=include_vision,
            apply_contextual_retrieval=apply_contextual_retrieval,
            store_embeddings=store_embeddings,
        )

    def search(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float = 0.0,
        filter_by_source: Optional[str] = None,
        filter_by_type: Optional[str] = None,
        filter_by_sheet: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for semantically similar content in the ingested documents.

        Args:
            query: Search query text
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold (0.0 to 1.0)
            filter_by_source: Optional filter by source file path
            filter_by_type: Optional filter by content type ('text', 'vision', etc.)
            filter_by_sheet: Optional filter by Excel sheet name

        Returns:
            List of search results with content and metadata
        """
        self.logger.info(f"🔍 Searching: '{query[:100]}...' (limit: {limit})")

        # Prepare filter conditions
        filter_conditions = {}
        if filter_by_source:
            filter_conditions["source"] = filter_by_source
        if filter_by_type:
            filter_conditions["content_type"] = filter_by_type

        # Use sheet-specific search if sheet filter is provided
        if filter_by_sheet:
            search_results = self.processor.embeddings.search_by_sheet_name(
                query_text=query,
                sheet_name=filter_by_sheet,
                limit=limit,
                score_threshold=score_threshold,
            )
        else:
            # Perform regular search
            search_results = self.processor.search_documents(
                query=query,
                limit=limit,
                score_threshold=score_threshold,
                filter_conditions=filter_conditions if filter_conditions else None,
            )

        # Format results for API consumption
        formatted_results = []
        for i, (document, score) in enumerate(search_results):
            result = {
                "rank": i + 1,
                "score": round(score, 4),
                "content": document.page_content,
                "metadata": document.metadata,
                "source": document.metadata.get("source", "unknown"),
                "content_type": document.metadata.get("content_type", "text"),
                "chunk_index": document.metadata.get("chunk_index", 0),
            }
            formatted_results.append(result)

        self.logger.info(f"✅ Found {len(formatted_results)} results")
        return formatted_results

    def delete_document(self, file_path: str) -> Dict[str, Any]:
        """
        Delete all content from a specific document.

        Args:
            file_path: Path of the document to delete

        Returns:
            Dictionary with deletion results
        """
        self.logger.info(f"🗑️  Deleting document: {Path(file_path).name}")

        deleted_count = self.processor.delete_document(file_path)

        result = {
            "success": True,
            "file_path": file_path,
            "file_name": Path(file_path).name,
            "deleted_chunks": deleted_count,
        }

        self.logger.info(
            f"✅ Deleted {deleted_count} chunks from {Path(file_path).name}"
        )
        return result

    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive pipeline and storage statistics.

        Returns:
            Dictionary with pipeline configuration and storage statistics
        """
        processor_stats = self.processor.get_processor_stats()

        return {
            "pipeline": {
                "name": "Semantic Ingestion Pipeline",
                "version": "1.0.0",
                "features": [
                    "Contextual Retrieval",
                    "Vision Processing",
                    "Semantic Embeddings",
                    "LangChain Integration",
                ],
            },
            "configuration": processor_stats["config"],
            "supported_file_types": processor_stats["supported_file_types"],
            "components": {
                "contextual_retrieval": processor_stats["contextual_retrieval"],
                "semantic_chunker": processor_stats["semantic_chunker"],
                "vision_processor": processor_stats["vision_processor"],
                "embeddings": processor_stats["embeddings"],
            },
        }

    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of all pipeline components.

        Returns:
            Dictionary with health status of each component
        """
        self.logger.info("🏥 Performing health check...")

        health = {"overall": "healthy", "timestamp": time.time(), "components": {}}

        try:
            # Check Qdrant connection
            collection_info = self.processor.embeddings.get_collection_info()
            health["components"]["qdrant"] = {
                "status": "healthy" if "error" not in collection_info else "unhealthy",
                "collection": collection_info,
            }
        except Exception as e:
            health["components"]["qdrant"] = {"status": "unhealthy", "error": str(e)}
            health["overall"] = "degraded"

        try:
            # Check OpenAI API (try a simple embedding)
            test_result = self.processor.embeddings.embed_text("health check test")
            health["components"]["openai"] = {
                "status": "healthy",
                "embedding_model": self.config.embedding_model,
                "contextual_model": self.config.contextual_retrieval_model,
                "vision_model": self.config.vision_model,
            }
        except Exception as e:
            health["components"]["openai"] = {"status": "unhealthy", "error": str(e)}
            health["overall"] = "degraded"

        # Check configuration
        try:
            config_dict = self.config.to_dict()
            health["components"]["configuration"] = {
                "status": "healthy",
                "cache_enabled": config_dict.get("enable_prompt_caching", False),
            }
        except Exception as e:
            health["components"]["configuration"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health["overall"] = "degraded"

        status_msg = f"Health check completed - {health['overall']}"
        if health["overall"] == "healthy":
            self.logger.info(f"✅ {status_msg}")
        else:
            self.logger.warning(f"⚠️  {status_msg}")

        return health
