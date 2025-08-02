"""Document processing module for OCR and content extraction."""

import os
import re
import base64
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import magic
from mistralai import Mistral
import anthropic

from .config import IngestionConfig


class DocumentProcessor:
    """Handles document processing with OCR and content description."""

    def __init__(self, config: IngestionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize API clients
        self.mistral_client = Mistral(api_key=config.mistral_api_key)
        self.anthropic_client = anthropic.Anthropic(api_key=config.anthropic_api_key)

        self.logger.info("🔧 DocumentProcessor initialized with models:")
        self.logger.info(f"  - Mistral OCR: {config.mistral_parsing_model}")
        self.logger.info(
            f"  - Anthropic Description: {config.anthropic_description_model}"
        )

    def get_file_type(self, file_path: str) -> str:
        """Detect file type using python-magic."""
        try:
            mime_type = magic.from_file(file_path, mime=True)
            return mime_type
        except Exception as e:
            self.logger.warning(f"Could not detect MIME type for {file_path}: {e}")
            # Fallback to extension-based detection
            return self._get_mime_from_extension(file_path)

    def _get_mime_from_extension(self, file_path: str) -> str:
        """Fallback MIME type detection based on file extension."""
        ext = Path(file_path).suffix.lower()
        mime_map = {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".doc": "application/msword",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".xls": "application/vnd.ms-excel",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".ppt": "application/vnd.ms-powerpoint",
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
        }
        return mime_map.get(ext, "application/octet-stream")

    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a document and extract pages with OCR."""
        self.logger.info(f"📄 Processing document: {os.path.basename(file_path)}")

        file_type = self.get_file_type(file_path)
        self.logger.info(f"📋 Detected file type: {file_type}")

        if file_type == "application/pdf":
            return self._process_pdf(file_path)
        elif "image/" in file_type:
            return self._process_image(file_path)
        elif "officedocument" in file_type or file_type in [
            "application/msword",
            "application/vnd.ms-excel",
            "application/vnd.ms-powerpoint",
        ]:
            return self._process_office_document(file_path)
        else:
            # Try to process as PDF anyway (Mistral can handle various formats)
            self.logger.warning(
                f"Unknown file type {file_type}, attempting PDF processing"
            )
            return self._process_pdf(file_path)

    def _process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Process PDF document with Mistral OCR."""
        try:
            # Create images directory for extracted images
            image_dir = "data/pdf_images"
            os.makedirs(image_dir, exist_ok=True)

            # Read PDF file
            with open(pdf_path, "rb") as pdf_file:
                base64_pdf = base64.b64encode(pdf_file.read()).decode("utf-8")

            self.logger.info("🔍 Calling Mistral OCR...")

            # Call Mistral OCR
            ocr_response = self.mistral_client.ocr.process(
                model=self.config.mistral_parsing_model,
                document={
                    "type": "document_url",
                    "document_url": f"data:application/pdf;base64,{base64_pdf}",
                },
                include_image_base64=True,
            )

            pages = []
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

            for page_data in ocr_response.pages:
                page_index = page_data.index
                markdown_content = page_data.markdown
                saved_images = []

                # Process images from this page
                if hasattr(page_data, "images") and page_data.images:
                    for img_idx, image_data in enumerate(page_data.images):
                        if (
                            hasattr(image_data, "image_base64")
                            and image_data.image_base64
                        ):
                            # Clean base64 data
                            if image_data.image_base64.startswith("data:image"):
                                image_data.image_base64 = image_data.image_base64.split(
                                    ",", 1
                                )[1]

                            image_filename = f"{pdf_name}-{page_index}-{img_idx}.png"
                            image_path = os.path.join(image_dir, image_filename)

                            try:
                                # Save image
                                image_bytes = base64.b64decode(image_data.image_base64)
                                with open(image_path, "wb") as img_file:
                                    img_file.write(image_bytes)

                                saved_images.append(
                                    {
                                        "filename": image_filename,
                                        "path": image_path,
                                        "id": getattr(image_data, "id", None),
                                    }
                                )

                                # Replace image ID with filename in markdown
                                if hasattr(image_data, "id") and image_data.id:
                                    markdown_content = markdown_content.replace(
                                        image_data.id, image_filename
                                    )

                                self.logger.info(f"💾 Saved image: {image_filename}")

                            except Exception as e:
                                self.logger.error(f"Failed to save image: {e}")

                pages.append(
                    {
                        "text": markdown_content,
                        "images": saved_images,
                        "page_index": page_index,
                        "tables": self._extract_tables_from_markdown(markdown_content),
                    }
                )

            self.logger.info(f"✅ Processed {len(pages)} pages with Mistral OCR")
            return pages

        except Exception as e:
            self.logger.error(f"Mistral OCR failed: {e}")
            return []

    def _process_image(self, image_path: str) -> List[Dict[str, Any]]:
        """Process single image file."""
        try:
            # For images, we create a single page with the image
            image_filename = os.path.basename(image_path)

            page = {
                "text": f"![{image_filename}]({image_filename})",
                "images": [
                    {
                        "filename": image_filename,
                        "path": image_path,
                        "id": None,
                    }
                ],
                "page_index": 1,
                "tables": [],
            }

            return [page]

        except Exception as e:
            self.logger.error(f"Image processing failed: {e}")
            return []

    def _process_office_document(self, doc_path: str) -> List[Dict[str, Any]]:
        """Process Office documents (Word, Excel, PowerPoint)."""
        # For Office documents, we'll try to process them with Mistral OCR
        # which can handle various document formats
        return self._process_pdf(doc_path)

    def _extract_tables_from_markdown(
        self, markdown_content: str
    ) -> List[Dict[str, Any]]:
        """Extract table information from markdown content."""
        tables = []
        lines = markdown_content.split("\\n")

        table_lines = []
        in_table = False
        table_id = 0

        for i, line in enumerate(lines):
            if self._is_table_line(line.strip()):
                if not in_table:
                    in_table = True
                    table_id += 1
                    table_lines = [line]
                else:
                    table_lines.append(line)
            else:
                if in_table and table_lines:
                    # Table ended, process it
                    tables.append(
                        {
                            "id": table_id,
                            "content": "\\n".join(table_lines),
                            "start_line": i - len(table_lines),
                            "end_line": i - 1,
                        }
                    )
                    table_lines = []
                in_table = False

        # Handle table at end of content
        if in_table and table_lines:
            tables.append(
                {
                    "id": table_id,
                    "content": "\\n".join(table_lines),
                    "start_line": len(lines) - len(table_lines),
                    "end_line": len(lines) - 1,
                }
            )

        return tables

    def _is_table_line(self, line: str) -> bool:
        """Check if a line is part of a markdown table."""
        if not line:
            return False
        if re.match(r"^[\\s\\|:\\-]+$", line):
            return True
        return "|" in line and line.count("|") >= 2

    async def generate_image_description(
        self, image_path: str, context_text: str = ""
    ) -> str:
        """Generate description for an image using Anthropic Claude."""
        try:
            # Check file size
            file_size = os.path.getsize(image_path)
            max_size = self.config.max_image_size_mb * 1024 * 1024

            if file_size > max_size:
                self.logger.warning(
                    f"Image {image_path} too large ({file_size/1024/1024:.1f}MB), skipping"
                )
                return (
                    f"Image: {os.path.basename(image_path)} (too large for processing)"
                )

            # Read image data
            with open(image_path, "rb") as img_file:
                image_data = img_file.read()

            # Detect media type
            media_type = "image/png"
            if image_data.startswith(b"\\xff\\xd8\\xff"):
                media_type = "image/jpeg"

            b64_image = base64.b64encode(image_data).decode("utf-8")

            # Create prompt for image description
            prompt = self._create_image_description_prompt(context_text)

            # Call Anthropic API
            response = self.anthropic_client.messages.create(
                model=self.config.anthropic_description_model,
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": b64_image,
                                },
                            },
                        ],
                    }
                ],
            )

            return response.content[0].text.strip()

        except Exception as e:
            self.logger.error(f"Image description failed for {image_path}: {e}")
            return f"Image: {os.path.basename(image_path)}"

    async def generate_table_description(
        self, table_content: str, context_text: str = ""
    ) -> str:
        """Generate description for a table using Anthropic Claude."""
        try:
            prompt = self._create_table_description_prompt(table_content, context_text)

            response = self.anthropic_client.messages.create(
                model=self.config.anthropic_description_model,
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
            )

            return response.content[0].text.strip()

        except Exception as e:
            self.logger.error(f"Table description failed: {e}")
            return f"Table content: {table_content[:200]}..."

    def _create_image_description_prompt(self, context_text: str) -> str:
        """Create prompt for image description."""
        return f"""
Please analyze this image and provide a detailed description. Consider the following context from the document:

Context: {context_text[:500] if context_text else "No context provided"}

Provide a comprehensive description that includes:
1. What the image shows (objects, people, scenes, charts, diagrams, etc.)
2. Key visual elements and their relationships
3. Any text visible in the image
4. The purpose or function of the image in the document context
5. Technical details if it's a diagram, chart, or technical illustration

Be specific and detailed, as this description will be used for document search and retrieval.
"""

    def _create_table_description_prompt(
        self, table_content: str, context_text: str
    ) -> str:
        """Create prompt for table description."""
        return f"""
Please analyze this table and provide a comprehensive description. Consider the following context from the document:

Context: {context_text[:500] if context_text else "No context provided"}

Table Content:
{table_content}

Provide a detailed description that includes:
1. The purpose and subject of the table
2. Column headers and their meanings
3. Key data patterns, trends, or insights
4. Number of rows and columns
5. Data types (numerical, categorical, dates, etc.)
6. Notable values, outliers, or important entries
7. How this table relates to the document context

Be comprehensive and specific, as this description will be used for document search and retrieval.
"""

    async def process_images_batch(
        self, image_tasks: List[Tuple[str, str]]
    ) -> List[str]:
        """Process multiple images in batches for efficiency."""
        self.logger.info(f"🖼️  Processing {len(image_tasks)} images in batches")

        batch_size = self.config.batch_size
        all_descriptions = []

        for i in range(0, len(image_tasks), batch_size):
            batch = image_tasks[i : i + batch_size]
            self.logger.info(
                f"Processing image batch {i//batch_size + 1}/{(len(image_tasks) + batch_size - 1)//batch_size}"
            )

            # Process batch concurrently
            tasks = [
                self.generate_image_description(image_path, context)
                for image_path, context in batch
            ]

            batch_descriptions = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle exceptions
            for j, result in enumerate(batch_descriptions):
                if isinstance(result, Exception):
                    image_path = batch[j][0]
                    self.logger.error(f"Failed to process image {image_path}: {result}")
                    all_descriptions.append(f"Image: {os.path.basename(image_path)}")
                else:
                    all_descriptions.append(result)

            # Add delay between batches to respect rate limits
            if i + batch_size < len(image_tasks):
                await asyncio.sleep(2)

        return all_descriptions

    async def process_tables_batch(
        self, table_tasks: List[Tuple[str, str]]
    ) -> List[str]:
        """Process multiple tables in batches for efficiency."""
        self.logger.info(f"📊 Processing {len(table_tasks)} tables in batches")

        batch_size = self.config.batch_size
        all_descriptions = []

        for i in range(0, len(table_tasks), batch_size):
            batch = table_tasks[i : i + batch_size]
            self.logger.info(
                f"Processing table batch {i//batch_size + 1}/{(len(table_tasks) + batch_size - 1)//batch_size}"
            )

            # Process batch concurrently
            tasks = [
                self.generate_table_description(table_content, context)
                for table_content, context in batch
            ]

            batch_descriptions = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle exceptions
            for j, result in enumerate(batch_descriptions):
                if isinstance(result, Exception):
                    self.logger.error(f"Failed to process table {j}: {result}")
                    all_descriptions.append(f"Table: {table_tasks[j][0][:100]}...")
                else:
                    all_descriptions.append(result)

            # Add delay between batches to respect rate limits
            if i + batch_size < len(table_tasks):
                await asyncio.sleep(1)

        return all_descriptions
