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
import pandas as pd

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
            ".csv": "text/csv",
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
        elif file_type in [
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-excel",
        ]:
            return self._process_excel_document(file_path)
        elif "officedocument" in file_type or file_type in [
            "application/msword",
            "application/vnd.ms-powerpoint",
        ]:
            return self._process_office_document(file_path)
        elif file_type == "text/csv":
            return self._process_csv_document(file_path)
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
        """Generate description for an image using Anthropic Claude with retry logic."""
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

            # Call Anthropic API with retry logic
            response = await self._call_anthropic_with_retry(
                "image",
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
        """Generate description for a table using Anthropic Claude with retry logic."""
        try:
            prompt = self._create_table_description_prompt(table_content, context_text)

            response = await self._call_anthropic_with_retry(
                "table",
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
        """Process multiple images with controlled parallelization."""
        self.logger.info(
            f"🖼️  Processing {len(image_tasks)} images with max {self.config.parallel_description_calls} parallel calls"
        )

        # Use semaphore to limit concurrent API calls
        semaphore = asyncio.Semaphore(self.config.parallel_description_calls)

        async def process_single_image(image_path: str, context: str) -> str:
            async with semaphore:
                return await self.generate_image_description(image_path, context)

        # Create all tasks
        tasks = [
            process_single_image(image_path, context)
            for image_path, context in image_tasks
        ]

        # Process all tasks concurrently with controlled parallelism
        self.logger.info(
            f"Starting parallel processing of {len(tasks)} image descriptions"
        )
        descriptions = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions and build results
        all_descriptions = []
        for i, result in enumerate(descriptions):
            if isinstance(result, Exception):
                image_path, _ = image_tasks[i]
                self.logger.error(
                    f"Image description failed for {image_path}: {result}"
                )
                all_descriptions.append(f"Image: {os.path.basename(image_path)}")
            else:
                all_descriptions.append(result)

        self.logger.info(
            f"✅ Completed processing {len(all_descriptions)} image descriptions"
        )
        return all_descriptions

    async def process_tables_batch(
        self, table_tasks: List[Tuple[str, str]]
    ) -> List[str]:
        """Process multiple tables with controlled parallelization."""
        self.logger.info(
            f"📊 Processing {len(table_tasks)} tables with max {self.config.parallel_description_calls} parallel calls"
        )

        # Use semaphore to limit concurrent API calls
        semaphore = asyncio.Semaphore(self.config.parallel_description_calls)

        async def process_single_table(table_content: str, context: str) -> str:
            async with semaphore:
                return await self.generate_table_description(table_content, context)

        # Create all tasks
        tasks = [
            process_single_table(table_content, context)
            for table_content, context in table_tasks
        ]

        # Process all tasks concurrently with controlled parallelism
        self.logger.info(
            f"Starting parallel processing of {len(tasks)} table descriptions"
        )
        descriptions = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions and build results
        all_descriptions = []
        for i, result in enumerate(descriptions):
            if isinstance(result, Exception):
                table_content, _ = table_tasks[i]
                self.logger.error(f"Table description failed: {result}")
                all_descriptions.append(f"Table content: {table_content[:200]}...")
            else:
                all_descriptions.append(result)

        self.logger.info(
            f"✅ Completed processing {len(all_descriptions)} table descriptions"
        )
        return all_descriptions

    async def _call_anthropic_with_retry(self, request_type: str, **kwargs) -> Any:
        """Call Anthropic API with retry logic and proper error handling."""
        last_exception = None

        for attempt in range(self.config.api_retry_attempts):
            try:
                # Add a small delay between retries
                if attempt > 0:
                    delay = self.config.api_retry_delay * (
                        2 ** (attempt - 1)
                    )  # Exponential backoff
                    self.logger.info(
                        f"Retrying {request_type} description (attempt {attempt + 1}/{self.config.api_retry_attempts}) after {delay:.1f}s delay"
                    )
                    await asyncio.sleep(delay)

                # Make the API call
                response = self.anthropic_client.messages.create(**kwargs)

                # If successful, return the response
                return response

            except anthropic.APIError as e:
                last_exception = e
                error_code = getattr(e, "status_code", "unknown")
                error_type = getattr(e, "type", "unknown")
                error_message = str(e)

                self.logger.warning(
                    f"Anthropic API error for {request_type} description (attempt {attempt + 1}/{self.config.api_retry_attempts}): "
                    f"Status {error_code}, Type: {error_type}, Message: {error_message}"
                )

                # Don't retry on certain error types
                if error_code in [
                    400,
                    401,
                    403,
                ]:  # Bad request, unauthorized, forbidden
                    self.logger.error(
                        f"Non-retryable error for {request_type} description: {error_message}"
                    )
                    break

            except Exception as e:
                last_exception = e
                self.logger.warning(
                    f"Unexpected error for {request_type} description (attempt {attempt + 1}/{self.config.api_retry_attempts}): {e}"
                )

        # If we get here, all retries failed
        self.logger.error(
            f"All {self.config.api_retry_attempts} retry attempts failed for {request_type} description. "
            f"Last error: {last_exception}"
        )
        raise last_exception

    def _process_excel_document(self, file_path: str) -> List[Dict[str, Any]]:
        """Process Excel document by reading all sheets and generating descriptions."""
        try:
            self.logger.info(f"📊 Processing Excel file: {os.path.basename(file_path)}")

            # Read all sheets from the Excel file
            excel_data = pd.read_excel(file_path, sheet_name=None)

            pages = []
            page_index = 1

            for sheet_name, df in excel_data.items():
                self.logger.info(f"📋 Processing sheet: {sheet_name}")

                # Convert DataFrame to markdown table format
                if not df.empty:
                    # Clean up the data - handle NaN values
                    df_clean = df.fillna("")

                    # Convert to markdown table
                    markdown_table = df_clean.to_markdown(index=False)

                    # Create sheet content with context
                    sheet_content = f"## Sheet: {sheet_name}\n\n"
                    sheet_content += f"Rows: {len(df)}, Columns: {len(df.columns)}\n\n"
                    sheet_content += f"Columns: {', '.join(df.columns)}\n\n"
                    sheet_content += markdown_table

                    # Create page data structure
                    page_data = {
                        "page_index": page_index,
                        "text": sheet_content,
                        "images": [],
                        "tables": [
                            {
                                "content": markdown_table,
                                "sheet_name": sheet_name,
                                "rows": len(df),
                                "columns": len(df.columns),
                                "column_names": list(df.columns),
                            }
                        ],
                    }

                    pages.append(page_data)
                    page_index += 1
                else:
                    self.logger.warning(f"Sheet '{sheet_name}' is empty, skipping")

            if not pages:
                # Create a minimal page if no sheets processed
                pages.append(
                    {
                        "page_index": 1,
                        "text": f"Excel file: {os.path.basename(file_path)} (no readable data)",
                        "images": [],
                        "tables": [],
                    }
                )

            self.logger.info(f"✅ Processed Excel file with {len(pages)} sheets")
            return pages

        except Exception as e:
            self.logger.error(f"❌ Failed to process Excel file {file_path}: {e}")
            # Fallback to basic text processing
            return [
                {
                    "page_index": 1,
                    "text": f"Excel file: {os.path.basename(file_path)} (processing failed: {e})",
                    "images": [],
                    "tables": [],
                }
            ]

    def _process_csv_document(self, file_path: str) -> List[Dict[str, Any]]:
        """Process CSV document by reading and generating description."""
        try:
            self.logger.info(f"📊 Processing CSV file: {os.path.basename(file_path)}")

            # Try to read CSV with different encodings
            encodings = ["utf-8", "latin-1", "cp1252"]
            df = None

            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    self.logger.info(
                        f"✅ Successfully read CSV with {encoding} encoding"
                    )
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    self.logger.warning(f"Failed to read CSV with {encoding}: {e}")
                    continue

            if df is None:
                raise Exception("Could not read CSV file with any encoding")

            # Clean up the data
            df_clean = df.fillna("")

            # Convert to markdown table
            markdown_table = df_clean.to_markdown(index=False)

            # Create content with context
            content = f"## CSV File: {os.path.basename(file_path)}\n\n"
            content += f"Rows: {len(df)}, Columns: {len(df.columns)}\n\n"
            content += f"Columns: {', '.join(df.columns)}\n\n"
            content += markdown_table

            # Create page data structure
            page_data = {
                "page_index": 1,
                "text": content,
                "images": [],
                "tables": [
                    {
                        "content": markdown_table,
                        "file_name": os.path.basename(file_path),
                        "rows": len(df),
                        "columns": len(df.columns),
                        "column_names": list(df.columns),
                    }
                ],
            }

            self.logger.info(
                f"✅ Processed CSV file with {len(df)} rows and {len(df.columns)} columns"
            )
            return [page_data]

        except Exception as e:
            self.logger.error(f"❌ Failed to process CSV file {file_path}: {e}")
            # Fallback to basic text processing
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                return [
                    {
                        "page_index": 1,
                        "text": f"CSV file: {os.path.basename(file_path)}\n\n{content[:2000]}{'...' if len(content) > 2000 else ''}",
                        "images": [],
                        "tables": [],
                    }
                ]
            except:
                return [
                    {
                        "page_index": 1,
                        "text": f"CSV file: {os.path.basename(file_path)} (processing failed: {e})",
                        "images": [],
                        "tables": [],
                    }
                ]
