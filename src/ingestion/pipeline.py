"""Main ingestion pipeline that orchestrates document processing, description generation, chunking, and vectorization."""

import os
import re
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import gc

from .config import IngestionConfig
from .document_processor import DocumentProcessor
from .chunker import ContentChunker
from .vectorizer import VectorStore


class IngestionPipeline:
    """Main pipeline for document ingestion into Qdrant."""

    def __init__(self, config: Optional[IngestionConfig] = None):
        self.config = config or IngestionConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.document_processor = DocumentProcessor(self.config)
        self.chunker = ContentChunker(self.config)
        self.vector_store = VectorStore(self.config)

        self.logger.info("🚀 IngestionPipeline initialized successfully")
        self.logger.info(f"📊 Configuration: {self.config.to_dict()}")

    def ingest_directory(
        self, directory_path: str, file_patterns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Ingest all documents in a directory."""
        self.logger.info(f"📂 Starting directory ingestion: {directory_path}")

        if not os.path.exists(directory_path):
            error_msg = f"Directory does not exist: {directory_path}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg}

        # Find all documents
        documents = self._find_documents(directory_path, file_patterns)

        if not documents:
            warning_msg = f"No documents found in directory: {directory_path}"
            self.logger.warning(warning_msg)
            return {
                "success": True,
                "warning": warning_msg,
                "processed": 0,
                "failed": 0,
            }

        self.logger.info(f"📋 Found {len(documents)} documents to process")

        # Process all documents
        results = {
            "success": True,
            "processed": 0,
            "failed": 0,
            "documents": [],
            "errors": [],
        }

        for doc_path in documents:
            try:
                self.logger.info(
                    f"🔄 Processing document: {os.path.basename(doc_path)}"
                )
                result = self.ingest_document(doc_path)

                if result["success"]:
                    results["processed"] += 1
                    results["documents"].append(
                        {
                            "path": doc_path,
                            "name": os.path.basename(doc_path),
                            "chunks": result.get("chunks", 0),
                            "pages": result.get("pages", 0),
                        }
                    )
                else:
                    results["failed"] += 1
                    results["errors"].append(
                        {
                            "path": doc_path,
                            "error": result.get("error", "Unknown error"),
                        }
                    )

                # Force garbage collection between documents
                gc.collect()

            except Exception as e:
                self.logger.error(f"❌ Failed to process {doc_path}: {e}")
                results["failed"] += 1
                results["errors"].append({"path": doc_path, "error": str(e)})

        self.logger.info(
            f"✅ Directory ingestion completed: {results['processed']} successful, {results['failed']} failed"
        )
        return results

    def ingest_document(self, file_path: str) -> Dict[str, Any]:
        """Ingest a single document."""
        try:
            document_name = os.path.splitext(os.path.basename(file_path))[0]
            self.logger.info(f"📄 Starting document ingestion: {document_name}")

            # Create output directory structure for images
            doc_dir = os.path.join("data/processed_documents", document_name)
            files_dir = os.path.join(doc_dir, "files")
            os.makedirs(files_dir, exist_ok=True)

            # Step 1: Process document with OCR
            self.logger.info("🔍 Step 1: OCR processing")
            pages = self.document_processor.process_document(
                file_path, files_dir, document_name
            )

            if not pages:
                error_msg = "No pages extracted from document"
                self.logger.error(error_msg)
                return {"success": False, "error": error_msg}

            self.logger.info(f"✅ OCR completed: {len(pages)} pages extracted")

            # Step 2: Generate descriptions for images and tables
            self.logger.info("🖼️  Step 2: Generating image and table descriptions")
            image_descriptions, table_descriptions = asyncio.run(
                self._generate_all_descriptions(pages)
            )

            self.logger.info(
                f"✅ Descriptions generated: {len(image_descriptions)} images, {len(table_descriptions)} tables"
            )

            # Step 3: Create chunks with descriptions
            self.logger.info("🔪 Step 3: Content chunking")
            chunks = self.chunker.chunk_document_pages(
                pages, image_descriptions, table_descriptions, document_name
            )

            if not chunks:
                error_msg = "No chunks generated from document"
                self.logger.error(error_msg)
                return {"success": False, "error": error_msg}

            self.logger.info(f"✅ Chunking completed: {len(chunks)} chunks created")

            # Step 4: Generate embeddings and store in Qdrant
            self.logger.info("🗄️  Step 4: Vectorization and storage")
            storage_success = self.vector_store.store_chunks(
                chunks, document_name, file_path
            )

            if not storage_success:
                error_msg = "Failed to store chunks in Qdrant"
                self.logger.error(error_msg)
                return {"success": False, "error": error_msg}

            # Save processed markdown
            self._save_processed_markdown(
                pages, image_descriptions, table_descriptions, document_name
            )

            self.logger.info(
                f"🎉 Document ingestion completed successfully: {document_name}"
            )

            return {
                "success": True,
                "document_name": document_name,
                "pages": len(pages),
                "chunks": len(chunks),
                "images": len(image_descriptions),
                "tables": len(table_descriptions),
            }

        except Exception as e:
            error_msg = f"Document ingestion failed: {e}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg}

    async def _generate_all_descriptions(
        self, pages: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Generate descriptions for all images and tables in the document."""

        # Collect all image and table tasks
        image_tasks = []
        table_tasks = []

        for page in pages:
            page_context = page["text"][:500]  # First 500 chars as context

            # Collect image tasks
            for image_info in page.get("images", []):
                image_tasks.append((image_info["path"], page_context))

            # Collect table tasks
            for i, table_info in enumerate(page.get("tables", [])):
                table_key = f"table_{page['page_index']}_{table_info.get('id', i)}"
                table_tasks.append((table_info["content"], page_context))

        # Process images and tables in parallel
        image_descriptions_list = []
        table_descriptions_list = []

        if image_tasks:
            image_descriptions_list = (
                await self.document_processor.process_images_batch(image_tasks)
            )

        if table_tasks:
            table_descriptions_list = (
                await self.document_processor.process_tables_batch(table_tasks)
            )

        # Create mapping dictionaries
        image_descriptions = {}
        for i, (image_path, _) in enumerate(image_tasks):
            if i < len(image_descriptions_list):
                image_descriptions[image_path] = image_descriptions_list[i]

        table_descriptions = {}
        table_idx = 0
        for page in pages:
            for i, table_info in enumerate(page.get("tables", [])):
                table_key = f"table_{page['page_index']}_{table_info.get('id', i)}"
                if table_idx < len(table_descriptions_list):
                    table_descriptions[table_key] = table_descriptions_list[table_idx]
                table_idx += 1

        return image_descriptions, table_descriptions

    def _find_documents(
        self, directory_path: str, file_patterns: Optional[List[str]] = None
    ) -> List[str]:
        """Find all documents in directory matching patterns."""
        if file_patterns is None:
            file_patterns = [
                "*.pdf",
                "*.docx",
                "*.doc",
                "*.xlsx",
                "*.xls",
                "*.pptx",
                "*.ppt",
            ]

        documents = []
        directory = Path(directory_path)

        for pattern in file_patterns:
            documents.extend([str(p) for p in directory.rglob(pattern)])

        # Remove duplicates and sort
        documents = sorted(list(set(documents)))

        self.logger.info(
            f"📋 Found {len(documents)} documents with patterns: {file_patterns}"
        )
        return documents

    def _save_processed_markdown(
        self,
        pages: List[Dict[str, Any]],
        image_descriptions: Dict[str, str],
        table_descriptions: Dict[str, str],
        document_name: str,
    ):
        """Save processed document as markdown file with proper file structure."""
        try:
            # Get output directory structure (already created earlier)
            doc_dir = os.path.join("data/processed_documents", document_name)
            files_dir = os.path.join(doc_dir, "files")

            # Build markdown content
            markdown_content = f"# {document_name}\\n\\n"

            # Process pages and reference images properly
            for page in pages:
                markdown_content += f"## Page {page['page_index']}\\n\\n"

                # Process page content and update image references
                page_content = self._process_page_content_with_files(
                    page,
                    image_descriptions,
                    table_descriptions,
                    files_dir,
                    document_name,
                )
                markdown_content += page_content + "\\n\\n"

            # Save markdown file
            output_path = os.path.join(doc_dir, f"{document_name}.md")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)

            self.logger.info(f"💾 Saved processed markdown: {output_path}")
            self.logger.info(f"📁 Document files saved in: {files_dir}")

        except Exception as e:
            self.logger.error(f"❌ Failed to save processed markdown: {e}")

    def _process_page_content_with_files(
        self,
        page: Dict[str, Any],
        image_descriptions: Dict[str, str],
        table_descriptions: Dict[str, str],
        files_dir: str,
        document_name: str,
    ) -> str:
        """Process page content and update image references (images are already in correct location)."""
        content = page["text"]

        # Apply image descriptions (images are already saved in correct location with correct names)
        for image_info in page.get("images", []):
            image_path = image_info["path"]
            original_filename = image_info["filename"]

            # Images are already saved with the correct filename
            current_filename = os.path.basename(image_path)

            description = image_descriptions.get(
                image_path, f"Image: {original_filename}"
            )

            # Create enhanced image description with correct path
            enhanced_description = f"<description_{current_filename}>{description}</description_{current_filename}>"

            # Replace image references with correct path
            patterns = [
                rf"!\\[([^\\]]*)\\]\\({re.escape(original_filename)}\\)",
                re.escape(original_filename),
            ]

            for pattern in patterns:
                if re.search(pattern, content):
                    # Update the image reference to point to the files directory
                    image_ref = f"![{current_filename}](files/{current_filename})"
                    content = re.sub(
                        pattern, f"{enhanced_description}\\n\\n{image_ref}", content
                    )
                    break

        # Apply table descriptions and save CSV files
        for i, table_info in enumerate(page.get("tables", [])):
            table_id = table_info.get("id", f"table_{i}")
            table_content = table_info["content"]

            table_key = f"table_{page['page_index']}_{table_id}"
            description = table_descriptions.get(
                table_key, f"Table: {table_content[:100]}..."
            )

            # Create CSV filename for table: {document_name}_table_{table_number}.csv
            table_number = i + 1
            csv_filename = f"{document_name}_table_{table_number}.csv"
            csv_path = os.path.join(files_dir, csv_filename)

            # Save table as CSV file
            try:
                self._save_table_as_csv(table_content, csv_path)
                self.logger.info(f"📋 Saved table as CSV: {csv_filename}")
            except Exception as e:
                self.logger.warning(
                    f"⚠️  Failed to save table as CSV {csv_filename}: {e}"
                )

            # Create enhanced table description with CSV reference
            enhanced_description = f"<description_table_{table_id}>{description}</description_table_{table_id}>"
            csv_ref = f"[Download CSV: {csv_filename}](files/{csv_filename})"

            # Replace table content with description and CSV link
            content = content.replace(
                table_content, f"{enhanced_description}\n\n{csv_ref}"
            )

        return content

    def _save_table_as_csv(self, table_content: str, csv_path: str) -> None:
        """Save table content as CSV file."""
        import csv
        import io
        import re

        # Parse markdown table to CSV
        lines = table_content.strip().split("\n")

        # Filter out separator lines (those with just |, -, :, and spaces)
        data_lines = []
        for line in lines:
            if line.strip() and not re.match(r"^[\s\|:\-]+$", line.strip()):
                data_lines.append(line)

        if not data_lines:
            return

        # Parse table rows
        rows = []
        for line in data_lines:
            # Split by | and clean up
            cells = [cell.strip() for cell in line.split("|")]
            # Remove empty cells at start/end (from leading/trailing |)
            if cells and not cells[0]:
                cells = cells[1:]
            if cells and not cells[-1]:
                cells = cells[:-1]
            if cells:  # Only add non-empty rows
                rows.append(cells)

        # Write to CSV
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            if rows:
                writer = csv.writer(csvfile)
                writer.writerows(rows)

    def search_documents(
        self, query: str, limit: int = 10, document_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search across all documents or within a specific document."""
        self.logger.info(f"🔍 Searching for: '{query}' (limit: {limit})")

        if document_name:
            results = self.vector_store.search_by_document(query, document_name, limit)
            self.logger.info(
                f"📊 Found {len(results)} results in document '{document_name}'"
            )
        else:
            results = self.vector_store.search(query, limit)
            self.logger.info(f"📊 Found {len(results)} results across all documents")

        return results

    def get_document_info(self, document_name: str) -> Dict[str, Any]:
        """Get information about a stored document."""
        return self.vector_store.get_document_info(document_name)

    def list_documents(self) -> List[str]:
        """List all stored documents."""
        return self.vector_store.list_documents()

    def delete_document(self, document_name: str) -> bool:
        """Delete a document and all its chunks."""
        return self.vector_store.delete_document(document_name)

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline and storage statistics."""
        collection_stats = self.vector_store.get_collection_stats()
        documents = self.list_documents()

        return {
            "collection_stats": collection_stats,
            "document_count": len(documents),
            "documents": documents,
            "config": self.config.to_dict(),
        }
