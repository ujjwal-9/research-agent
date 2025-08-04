"""Main semantic document processor using LangChain."""

import asyncio
import concurrent.futures
import logging
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
)

from .contextual_retrieval import ContextualRetrieval
from .semantic_chunker import SemanticChunker
from .vision_processor import VisionProcessor
from .embeddings import SemanticEmbeddings


class SemanticDocumentProcessor:
    """
    Main semantic document processor that orchestrates the entire pipeline.

    Uses LangChain for document loading and integrates:
    - Contextual retrieval for better chunk semantics
    - Vision processing for images/charts/tables
    - Semantic embeddings with OpenAI text-embedding-3-large
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize core components
        self.contextual_retrieval = ContextualRetrieval(config)
        self.semantic_chunker = SemanticChunker(config, self.contextual_retrieval)
        self.vision_processor = VisionProcessor(config, self.contextual_retrieval)
        self.embeddings = SemanticEmbeddings(config)

        # Supported file types and their loaders
        self.loaders = {
            ".pdf": PyPDFLoader,
            ".docx": UnstructuredWordDocumentLoader,
            ".doc": UnstructuredWordDocumentLoader,
            ".xlsx": UnstructuredExcelLoader,
            ".xls": UnstructuredExcelLoader,
            ".pptx": UnstructuredPowerPointLoader,
            ".ppt": UnstructuredPowerPointLoader,
            ".txt": TextLoader,
            ".html": UnstructuredHTMLLoader,
            ".htm": UnstructuredHTMLLoader,
            ".md": UnstructuredMarkdownLoader,
            ".markdown": UnstructuredMarkdownLoader,
        }

        self.logger.info("✅ Semantic document processor initialized")

    def _get_file_loader(self, file_path: str) -> Optional[type]:
        """Get appropriate document loader for file type."""
        file_extension = Path(file_path).suffix.lower()
        return self.loaders.get(file_extension)

    def _extract_images_from_document(
        self, file_path: str
    ) -> List[Tuple[bytes, str, str]]:
        """
        Extract images from document for vision processing.

        Returns:
            List of tuples (image_data, content_type, surrounding_text)
        """
        # This is a simplified implementation
        # In a real scenario, you would use specialized libraries to extract images
        # from PDFs, Word docs, etc. along with their surrounding context

        images = []

        # For now, return empty list - this would be implemented based on
        # specific document processing libraries like pymupdf for PDFs,
        # python-docx for Word documents, etc.

        self.logger.info(
            f"🖼️  Image extraction for {file_path} - implementation needed for production"
        )
        return images

    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a document using appropriate LangChain loader.

        Args:
            file_path: Path to the document file

        Returns:
            List of loaded Document objects
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get appropriate loader
        loader_class = self._get_file_loader(file_path)
        if not loader_class:
            raise ValueError(f"Unsupported file type: {Path(file_path).suffix}")

        self.logger.info(f"📖 Loading document: {file_path}")

        try:
            # Load document
            loader = loader_class(file_path)
            documents = loader.load()

            # Add file metadata
            for doc in documents:
                doc.metadata.update(
                    {
                        "source": file_path,
                        "file_name": Path(file_path).name,
                        "file_extension": Path(file_path).suffix.lower(),
                        "file_size_bytes": os.path.getsize(file_path),
                        "loaded_at": time.time(),
                    }
                )

            self.logger.info(
                f"✅ Loaded {len(documents)} document sections from {file_path}"
            )
            return documents

        except Exception as e:
            self.logger.error(f"Error loading document {file_path}: {e}")
            raise

    def process_document(
        self,
        file_path: str,
        include_vision: bool = True,
        apply_contextual_retrieval: bool = True,
        store_embeddings: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a complete document through the semantic pipeline.

        Args:
            file_path: Path to the document file
            include_vision: Whether to process images/charts/tables
            apply_contextual_retrieval: Whether to apply contextual retrieval
            store_embeddings: Whether to store embeddings in vector database

        Returns:
            Dictionary with processing results and statistics
        """
        start_time = time.time()
        file_name = Path(file_path).name

        self.logger.info(f"🚀 Starting semantic processing: {file_name}")

        try:
            # Step 1: Load document
            self.logger.info("📖 Step 1: Loading document...")
            documents = self.load_document(file_path)

            if not documents:
                raise ValueError("No content loaded from document")

            # Combine all document content for contextual retrieval
            full_content = "\n\n".join([doc.page_content for doc in documents])

            # Step 2: Process vision content if enabled
            vision_documents = []
            if include_vision:
                self.logger.info("🖼️  Step 2: Processing vision content...")
                images = self._extract_images_from_document(file_path)

                if images:
                    vision_contents = self.vision_processor.process_vision_content(
                        images=images,
                        document_content=full_content,
                        source_document=file_path,
                        metadata={"vision_processing": True},
                    )

                    vision_documents = (
                        self.vision_processor.vision_content_to_documents(
                            vision_contents
                        )
                    )
                    self.logger.info(
                        f"✅ Processed {len(vision_documents)} vision contents"
                    )
                else:
                    self.logger.info("ℹ️  No images found in document")

            # Step 3: Chunk documents
            self.logger.info("🔪 Step 3: Creating semantic chunks...")
            text_chunks = self.semantic_chunker.chunk_documents(
                documents=documents,
                apply_contextual_retrieval=apply_contextual_retrieval,
            )

            # Combine text and vision chunks
            all_chunks = text_chunks + vision_documents

            self.logger.info(
                f"📄 Created {len(text_chunks)} text chunks and {len(vision_documents)} vision chunks"
            )

            # Step 4: Generate embeddings
            self.logger.info("🔮 Step 4: Generating embeddings...")
            if self.config.parallel_chunk_processing and len(all_chunks) > 1:
                # Use async embeddings for better parallel performance
                embedding_results = self.embeddings.embed_documents(all_chunks)
            else:
                # Use standard synchronous embedding generation
                embedding_results = self.embeddings.embed_documents(all_chunks)

            # Step 5: Store embeddings if requested
            stored_ids = []
            if store_embeddings:
                self.logger.info("💾 Step 5: Storing embeddings...")
                stored_ids = self.embeddings.store_embeddings(embedding_results)

            # Calculate statistics
            processing_time = time.time() - start_time

            results = {
                "success": True,
                "file_path": file_path,
                "file_name": file_name,
                "processing_time": processing_time,
                "statistics": {
                    "total_documents_loaded": len(documents),
                    "total_chunks_created": len(all_chunks),
                    "text_chunks": len(text_chunks),
                    "vision_chunks": len(vision_documents),
                    "embeddings_generated": len(embedding_results),
                    "embeddings_stored": len(stored_ids),
                    "contextual_retrieval_applied": apply_contextual_retrieval,
                    "vision_processing_applied": include_vision,
                },
                "chunks": all_chunks,
                "embeddings": embedding_results,
                "stored_ids": stored_ids,
            }

            self.logger.info(
                f"✅ Completed semantic processing: {file_name} in {processing_time:.2f}s"
            )
            return results

        except Exception as e:
            self.logger.error(f"❌ Error processing document {file_name}: {e}")
            return {
                "success": False,
                "file_path": file_path,
                "file_name": file_name,
                "error": str(e),
                "processing_time": time.time() - start_time,
            }

    async def _process_document_async(
        self,
        file_path: str,
        include_vision: bool = True,
        apply_contextual_retrieval: bool = True,
        store_embeddings: bool = True,
    ) -> Dict[str, Any]:
        """Process a single document asynchronously."""
        loop = asyncio.get_running_loop()

        # Run the synchronous process_document in a thread pool
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                self.process_document,
                file_path,
                include_vision,
                apply_contextual_retrieval,
                store_embeddings,
            )
            result = await loop.run_in_executor(None, lambda: future.result())
            return result

    async def process_multiple_documents_async(
        self,
        file_paths: List[str],
        include_vision: bool = True,
        apply_contextual_retrieval: bool = True,
        store_embeddings: bool = True,
    ) -> List[Dict[str, Any]]:
        """Process multiple documents through the semantic pipeline in parallel."""
        if not self.config.enable_parallel_processing:
            # Fall back to sequential processing
            return self.process_multiple_documents_sequential(
                file_paths, include_vision, apply_contextual_retrieval, store_embeddings
            )

        self.logger.info(
            f"🚀 Processing {len(file_paths)} documents in parallel (max: {self.config.max_parallel_documents})"
        )

        # Create semaphore to limit concurrent document processing
        semaphore = asyncio.Semaphore(self.config.max_parallel_documents)

        async def process_with_semaphore(file_path: str, index: int) -> Dict[str, Any]:
            async with semaphore:
                self.logger.info(
                    f"📄 Processing document {index + 1}/{len(file_paths)}: {Path(file_path).name}"
                )
                return await self._process_document_async(
                    file_path,
                    include_vision,
                    apply_contextual_retrieval,
                    store_embeddings,
                )

        # Create tasks for all documents
        tasks = [
            process_with_semaphore(file_path, i)
            for i, file_path in enumerate(file_paths)
        ]

        # Execute all tasks with progress tracking
        results = []
        completed = 0

        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            completed += 1

            if completed % max(1, len(file_paths) // 10) == 0:  # Log every 10%
                progress = (completed / len(file_paths)) * 100
                self.logger.info(
                    f"📊 Progress: {completed}/{len(file_paths)} documents ({progress:.1f}%)"
                )

        # Sort results to match original order
        file_to_result = {r["file_path"]: r for r in results}
        results = [file_to_result[fp] for fp in file_paths]

        # Calculate overall statistics
        successful = sum(1 for r in results if r.get("success", False))
        failed = len(results) - successful
        total_chunks = sum(
            r.get("statistics", {}).get("total_chunks_created", 0) for r in results
        )

        self.logger.info("✅ Parallel batch processing completed:")
        self.logger.info(f"  📊 Successful: {successful}/{len(file_paths)} documents")
        self.logger.info(f"  📊 Failed: {failed}/{len(file_paths)} documents")
        self.logger.info(f"  📊 Total chunks created: {total_chunks}")

        return results

    def process_multiple_documents_sequential(
        self,
        file_paths: List[str],
        include_vision: bool = True,
        apply_contextual_retrieval: bool = True,
        store_embeddings: bool = True,
    ) -> List[Dict[str, Any]]:
        """Process multiple documents sequentially (original implementation)."""
        self.logger.info(f"🚀 Processing {len(file_paths)} documents sequentially")

        results = []
        for i, file_path in enumerate(file_paths, 1):
            self.logger.info(
                f"📄 Processing document {i}/{len(file_paths)}: {Path(file_path).name}"
            )

            result = self.process_document(
                file_path=file_path,
                include_vision=include_vision,
                apply_contextual_retrieval=apply_contextual_retrieval,
                store_embeddings=store_embeddings,
            )

            results.append(result)

        # Calculate overall statistics
        successful = sum(1 for r in results if r.get("success", False))
        failed = len(results) - successful
        total_chunks = sum(
            r.get("statistics", {}).get("total_chunks_created", 0) for r in results
        )

        self.logger.info("✅ Sequential batch processing completed:")
        self.logger.info(f"  📊 Successful: {successful}/{len(file_paths)} documents")
        self.logger.info(f"  📊 Failed: {failed}/{len(file_paths)} documents")
        self.logger.info(f"  📊 Total chunks created: {total_chunks}")

        return results

    def process_multiple_documents(
        self,
        file_paths: List[str],
        include_vision: bool = True,
        apply_contextual_retrieval: bool = True,
        store_embeddings: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Process multiple documents through the semantic pipeline.

        Args:
            file_paths: List of document file paths
            include_vision: Whether to process images/charts/tables
            apply_contextual_retrieval: Whether to apply contextual retrieval
            store_embeddings: Whether to store embeddings in vector database

        Returns:
            List of processing results for each document
        """
        if self.config.enable_parallel_processing:
            # Use async processing for better performance
            try:
                # Check if we're already in an event loop
                asyncio.get_running_loop()
                # If we are, we can't use asyncio.run(), so fall back to sequential
                self.logger.warning(
                    "Already in event loop, falling back to sequential processing"
                )
                return self.process_multiple_documents_sequential(
                    file_paths,
                    include_vision,
                    apply_contextual_retrieval,
                    store_embeddings,
                )
            except RuntimeError:
                # No event loop running, create one
                return asyncio.run(
                    self.process_multiple_documents_async(
                        file_paths,
                        include_vision,
                        apply_contextual_retrieval,
                        store_embeddings,
                    )
                )
        else:
            # Use sequential processing
            return self.process_multiple_documents_sequential(
                file_paths, include_vision, apply_contextual_retrieval, store_embeddings
            )

    def process_directory(
        self,
        directory_path: str,
        file_patterns: Optional[List[str]] = None,
        recursive: bool = True,
        include_vision: bool = True,
        apply_contextual_retrieval: bool = True,
        store_embeddings: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Process all supported documents in a directory.

        Args:
            directory_path: Path to directory containing documents
            file_patterns: List of file patterns to match (e.g., ['*.pdf', '*.docx'])
            recursive: Whether to search subdirectories
            include_vision: Whether to process images/charts/tables
            apply_contextual_retrieval: Whether to apply contextual retrieval
            store_embeddings: Whether to store embeddings in vector database

        Returns:
            List of processing results for each document
        """
        directory = Path(directory_path)
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Directory not found: {directory_path}")

        # Default file patterns if not provided
        if file_patterns is None:
            file_patterns = [
                "*.pdf",
                "*.docx",
                "*.doc",
                "*.xlsx",
                "*.xls",
                "*.pptx",
                "*.ppt",
                "*.txt",
                "*.html",
                "*.htm",
                "*.md",
                "*.markdown",
            ]

        # Find matching files
        file_paths = []
        for pattern in file_patterns:
            if recursive:
                files = directory.rglob(pattern)
            else:
                files = directory.glob(pattern)

            file_paths.extend([str(f) for f in files if f.is_file()])

        # Remove duplicates and sort
        file_paths = sorted(list(set(file_paths)))

        self.logger.info(f"📂 Found {len(file_paths)} documents in {directory_path}")

        if not file_paths:
            self.logger.warning(f"No matching documents found in {directory_path}")
            return []

        # Process all files
        return self.process_multiple_documents(
            file_paths=file_paths,
            include_vision=include_vision,
            apply_contextual_retrieval=apply_contextual_retrieval,
            store_embeddings=store_embeddings,
        )

    def search_documents(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float = 0.0,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents in the semantic vector database.

        Args:
            query: Search query text
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            filter_conditions: Optional filter conditions

        Returns:
            List of tuples (Document, similarity_score)
        """
        return self.embeddings.search_similar(
            query_text=query,
            limit=limit,
            score_threshold=score_threshold,
            filter_conditions=filter_conditions,
        )

    def delete_document(self, file_path: str) -> int:
        """
        Delete all chunks from a specific document.

        Args:
            file_path: Path of the document to delete

        Returns:
            Number of chunks deleted
        """
        return self.embeddings.delete_by_source(file_path)

    def get_processor_stats(self) -> Dict[str, Any]:
        """Get comprehensive processor statistics."""
        return {
            "config": self.config.to_dict(),
            "supported_file_types": list(self.loaders.keys()),
            "contextual_retrieval": self.contextual_retrieval.get_cache_stats(),
            "semantic_chunker": self.semantic_chunker.get_chunker_stats(),
            "vision_processor": self.vision_processor.get_processor_stats(),
            "embeddings": self.embeddings.get_embedding_stats(),
        }
