"""Knowledge Graph Ingestion Pipeline."""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    TextLoader,
    CSVLoader,
)

from .config import KnowledgeGraphConfig
from .extractor import KnowledgeGraphExtractor
from .neo4j_manager import Neo4jManager


class KnowledgeGraphPipeline:
    """Main pipeline for knowledge graph ingestion from documents."""

    def __init__(self, config: Optional[KnowledgeGraphConfig] = None):
        """Initialize the knowledge graph ingestion pipeline."""
        self.config = config or KnowledgeGraphConfig()

        # Validate configuration
        if not self.config.validate():
            raise ValueError("Invalid configuration")

        # Initialize components
        self.extractor = KnowledgeGraphExtractor(self.config)
        self.neo4j_manager = Neo4jManager(self.config)

        # Initialize text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.max_chunk_size,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Connect to Neo4j
        if not self.neo4j_manager.connect():
            raise ConnectionError("Failed to connect to Neo4j database")

        # Setup database schema
        self.neo4j_manager.setup_schema()

        self.logger.info("🚀 Knowledge Graph Pipeline initialized successfully")

    def ingest_directory(
        self,
        directory_path: str,
        file_patterns: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> Dict[str, Any]:
        """
        Ingest all documents in a directory into the knowledge graph.

        Args:
            directory_path: Path to directory containing documents
            file_patterns: Optional list of file extensions to include (e.g., ['.pdf', '.docx'])
            recursive: Whether to search subdirectories

        Returns:
            Dictionary with ingestion results and statistics
        """
        self.logger.info(f"📂 Starting directory ingestion: {directory_path}")

        # Find all documents
        documents = self._find_documents(directory_path, file_patterns, recursive)

        if not documents:
            self.logger.warning("No documents found in directory")
            return {
                "success": False,
                "error": "No documents found",
                "documents_processed": 0,
            }

        self.logger.info(f"📄 Found {len(documents)} documents to process")

        # Process documents in batches
        total_entities = 0
        total_relationships = 0
        processed_docs = 0
        failed_docs = 0

        start_time = time.time()

        # Process documents with controlled concurrency
        with ThreadPoolExecutor(
            max_workers=self.config.max_concurrent_requests
        ) as executor:
            # Submit batch of documents
            batch_size = self.config.batch_size
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]

                # Submit batch for processing
                future_to_doc = {
                    executor.submit(self.ingest_document, doc_path): doc_path
                    for doc_path in batch
                }

                # Process completed futures
                for future in as_completed(future_to_doc):
                    doc_path = future_to_doc[future]
                    try:
                        result = future.result()
                        if result["success"]:
                            total_entities += result["entities_extracted"]
                            total_relationships += result["relationships_extracted"]
                            processed_docs += 1
                            self.logger.info(
                                f"✅ Completed: {os.path.basename(doc_path)}"
                            )
                        else:
                            failed_docs += 1
                            self.logger.error(
                                f"❌ Failed: {os.path.basename(doc_path)} - {result.get('error', 'Unknown error')}"
                            )
                    except Exception as e:
                        failed_docs += 1
                        self.logger.error(
                            f"❌ Exception processing {os.path.basename(doc_path)}: {e}"
                        )

        end_time = time.time()
        processing_time = end_time - start_time

        # Get final statistics
        final_stats = self.neo4j_manager.get_graph_stats()

        result = {
            "success": True,
            "documents_found": len(documents),
            "documents_processed": processed_docs,
            "documents_failed": failed_docs,
            "entities_extracted": total_entities,
            "relationships_extracted": total_relationships,
            "processing_time_seconds": processing_time,
            "avg_time_per_document": (
                processing_time / len(documents) if documents else 0
            ),
            "final_graph_stats": final_stats,
        }

        self.logger.info(
            f"🎉 Directory ingestion completed: {processed_docs}/{len(documents)} documents processed, "
            f"{total_entities} entities, {total_relationships} relationships extracted"
        )

        return result

    def ingest_document(self, file_path: str) -> Dict[str, Any]:
        """
        Ingest a single document into the knowledge graph.

        Args:
            file_path: Path to the document file

        Returns:
            Dictionary with ingestion results
        """
        document_name = os.path.basename(file_path)
        self.logger.info(f"📄 Processing document: {document_name}")

        try:
            # Load document
            documents = self._load_document(file_path)
            if not documents:
                return {
                    "success": False,
                    "error": "Failed to load document",
                    "document": document_name,
                }

            # Combine all text content
            full_text = "\n\n".join([doc.page_content for doc in documents])

            # Split into chunks
            chunks = self.text_splitter.split_text(full_text)

            if not chunks:
                return {
                    "success": False,
                    "error": "No text chunks created",
                    "document": document_name,
                }

            self.logger.info(f"📄 Created {len(chunks)} chunks from {document_name}")

            # Extract knowledge graph from each chunk
            total_entities = 0
            total_relationships = 0

            for i, chunk in enumerate(chunks):
                chunk_id = f"{document_name}_chunk_{i}"

                # Skip very short chunks
                if len(chunk.strip()) < 100:
                    continue

                # Extract knowledge graph
                kg_data = self.extractor.extract_knowledge_graph(
                    text=chunk,
                    source_document=file_path,
                    chunk_id=chunk_id,
                    context=f"Document: {document_name}",
                )

                # Store in Neo4j
                if kg_data.entities or kg_data.relationships:
                    success = self.neo4j_manager.store_knowledge_graph(kg_data)
                    if success:
                        total_entities += len(kg_data.entities)
                        total_relationships += len(kg_data.relationships)
                    else:
                        self.logger.warning(f"Failed to store chunk {chunk_id}")

            result = {
                "success": True,
                "document": document_name,
                "chunks_processed": len(chunks),
                "entities_extracted": total_entities,
                "relationships_extracted": total_relationships,
            }

            self.logger.info(
                f"✅ Document completed: {document_name} - "
                f"{total_entities} entities, {total_relationships} relationships"
            )

            return result

        except Exception as e:
            error_msg = f"Failed to process document {document_name}: {e}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg, "document": document_name}

    def _find_documents(
        self,
        directory_path: str,
        file_patterns: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> List[str]:
        """Find all documents in directory matching patterns."""
        supported_extensions = [
            ".pdf",
            ".docx",
            ".doc",
            ".xlsx",
            ".xls",
            ".pptx",
            ".ppt",
            ".txt",
            ".csv",
        ]

        if file_patterns:
            # Use provided patterns
            extensions = file_patterns
        else:
            # Use all supported extensions
            extensions = supported_extensions

        documents = []
        directory = Path(directory_path)

        if not directory.exists():
            self.logger.error(f"Directory does not exist: {directory_path}")
            return []

        # Search for files
        pattern = "**/*" if recursive else "*"
        for file_path in directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                documents.append(str(file_path))

        return sorted(documents)

    def _load_document(self, file_path: str):
        """Load document content using appropriate loader based on file type."""
        file_extension = Path(file_path).suffix.lower()

        try:
            if file_extension == ".pdf":
                loader = PyPDFLoader(file_path)
            elif file_extension in [".docx", ".doc"]:
                loader = UnstructuredWordDocumentLoader(file_path)
            elif file_extension in [".xlsx", ".xls"]:
                loader = UnstructuredExcelLoader(file_path)
            elif file_extension in [".pptx", ".ppt"]:
                loader = UnstructuredPowerPointLoader(file_path)
            elif file_extension == ".txt":
                loader = TextLoader(file_path)
            elif file_extension == ".csv":
                loader = CSVLoader(file_path)
            else:
                self.logger.warning(
                    f"Unsupported file type: {file_extension}, trying PDF loader"
                )
                loader = PyPDFLoader(file_path)

            return loader.load()

        except Exception as e:
            self.logger.error(f"Failed to load document {file_path}: {e}")
            return []

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the knowledge graph."""
        return self.neo4j_manager.get_graph_stats()

    def clear_knowledge_graph(self) -> bool:
        """
        Clear all data from the knowledge graph. Use with caution!

        Returns:
            bool: True if successful, False otherwise
        """
        self.logger.warning("⚠️  Clearing all knowledge graph data")
        return self.neo4j_manager.clear_database()

    def close(self):
        """Close connections and cleanup resources."""
        if self.neo4j_manager:
            self.neo4j_manager.disconnect()
        self.logger.info("🔌 Knowledge Graph Pipeline closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
