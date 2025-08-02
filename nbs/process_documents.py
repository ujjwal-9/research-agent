import os
import re
import requests
import base64
import anthropic
import argparse
import json
import logging
import uuid
import asyncio
import time
import gc
import psutil
from datetime import datetime
from dotenv import load_dotenv
from mistralai import Mistral
from tqdm import tqdm
from pathlib import Path
from parser.prompts import image_description_prompt

load_dotenv()


class DocumentProcessor:
    """Streamlined document processor with core functionality only."""

    def __init__(self, config_path: str = None):
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set in environment.")
        self.image_caption_client = anthropic.Anthropic(api_key=self.anthropic_api_key)

        self.mistral_api_key = os.environ.get("MISTRAL_API_KEY")
        if not self.mistral_api_key:
            raise RuntimeError("MISTRAL_API_KEY is not set in environment.")
        self.mistral_client = Mistral(api_key=self.mistral_api_key)

        self._setup_logging()
        self.config = self._load_config(config_path)

        self.ragflow_api_key = os.environ.get("RAGFLOW_API_KEY")
        self.ragflow_base_url = os.environ.get("RAGFLOW_BASE_URL")

        # Log RAGFlow configuration at startup (after logger is set up)
        self.logger.info("🔧 RAGFlow Configuration:")
        self.logger.info(
            f"  - RAGFLOW_API_KEY: {'✅ Set' if self.ragflow_api_key else '❌ Missing'}"
        )
        self.logger.info(
            f"  - RAGFLOW_BASE_URL: {'✅ Set' if self.ragflow_base_url else '❌ Missing'}"
        )
        if self.ragflow_base_url:
            self.logger.info(f"  - RAGFlow URL: {self.ragflow_base_url}")

        if not self.ragflow_api_key or not self.ragflow_base_url:
            self.logger.warning(
                "⚠️  RAGFlow upload will be disabled - missing required environment variables"
            )

    def _log_memory_usage(self, context=""):
        """Log current memory usage."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            self.logger.info(f"💾 Memory usage {context}: {memory_mb:.1f} MB")
            return memory_mb
        except Exception as e:
            self.logger.warning(f"Could not get memory info: {e}")
            return 0

    def _check_memory_limit(self, context="", limit_mb=None):
        """Check if memory usage is approaching dangerous levels."""
        try:
            if limit_mb is None:
                # Get memory limit from environment or use default
                limit_mb = int(os.environ.get("MEMORY_LIMIT_MB", "8000"))

            current_memory = self._log_memory_usage(context)
            if current_memory > limit_mb:
                self.logger.error(
                    f"🚨 Memory limit exceeded: {current_memory:.1f} MB > {limit_mb} MB - stopping to prevent OOM"
                )
                # Force garbage collection before returning
                gc.collect()
                return False
            elif current_memory > limit_mb * 0.8:  # 80% of limit
                self.logger.warning(
                    f"⚠️  Memory usage high: {current_memory:.1f} MB (80% of {limit_mb} MB limit)"
                )
                # Proactive garbage collection
                gc.collect()
            return True
        except Exception as e:
            self.logger.warning(f"Could not check memory limit: {e}")
            return True

    def _setup_logging(self):
        os.makedirs("logs", exist_ok=True)
        self.logger = logging.getLogger("DocumentProcessor")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        log_filename = (
            f"logs/document_processor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler = logging.FileHandler(log_filename)
        console_handler = logging.StreamHandler()

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _load_config(self, config_path):
        """Load configuration with defaults."""
        default_config = {
            "timeout": 30,
            "dataset_settings": {
                "name": "Handbook",
                "description": "AI-enhanced document processing",
                "embedding_model": "text-embedding-3-large@OpenAI",
                "chunk_method": "naive",
                "parser_config": {
                    "chunk_token_num": 512,
                    "delimiter": "\\n",
                    "html4excel": False,
                    "layout_recognize": "pixtral-large-latest@Mistral",
                    "raptor": {"use_raptor": False},
                },
            },
            "chunking_settings": {"max_chars": 1000, "overlap_chars": 150},
            "deduplication_settings": {
                "enabled": True,
                "preserve_cross_page_duplicates": False,
                "similarity_threshold": 0.95,
                "use_fuzzy_matching": False,
                "min_chunk_length": 50,
            },
            "ocr_model": "mistral-ocr-latest",
            "image_caption_model": "claude-sonnet-4-20250514",
            "image_path": "data/pdf_images",
        }

        if config_path and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                loaded_config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in loaded_config:
                        loaded_config[key] = value
                return loaded_config
        return default_config

    def _get_api_headers(self):
        """Get headers for RAGFlow API."""
        return {
            "Authorization": f"Bearer {self.ragflow_api_key}",
            "Content-Type": "application/json",
        }

    def create_ragflow_dataset(self):
        """Create or get existing RAGFlow dataset."""
        if not self.ragflow_api_key or not self.ragflow_base_url:
            self.logger.error("RAGFlow API not configured")
            return None

        dataset_settings = self.config["dataset_settings"]

        try:
            url = f"{self.ragflow_base_url}/api/v1/datasets"
            response = requests.get(url, headers=self._get_api_headers(), timeout=30)

            if response.status_code == 200:
                result = response.json()
                if result.get("code") == 0 and result.get("data"):
                    datasets = (
                        result["data"]
                        if isinstance(result["data"], list)
                        else [result["data"]]
                    )
                    for dataset in datasets:
                        if dataset.get("name") == dataset_settings["name"]:
                            self.logger.info(
                                f"Using existing dataset: {dataset_settings['name']}"
                            )
                            return dataset

            payload = {
                "name": dataset_settings["name"],
                "description": dataset_settings["description"],
                "embedding_model": dataset_settings["embedding_model"],
                "chunk_method": dataset_settings["chunk_method"],
                "parser_config": dataset_settings["parser_config"],
            }

            response = requests.post(
                url, headers=self._get_api_headers(), json=payload, timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("code") == 0 and result.get("data"):
                    dataset = (
                        result["data"]
                        if not isinstance(result["data"], list)
                        else result["data"][0]
                    )
                    self.logger.info(f"Created dataset: {dataset_settings['name']}")
                    return dataset

            self.logger.error(f"Failed to create dataset: {response.text}")
            return None

        except Exception as e:
            self.logger.error(f"Error with dataset: {e}")
            return None

    def upload_empty_document_to_dataset(self, dataset_id, document_name):
        """Upload empty document to get document ID."""
        try:
            url = f"{self.ragflow_base_url}/api/v1/datasets/{dataset_id}/documents"

            content = f"# {document_name}\n\n*Document for custom chunking.*"
            files = {
                "file": (
                    f"{document_name}.md",
                    content.encode("utf-8"),
                    "text/markdown",
                )
            }
            headers = {"Authorization": f"Bearer {self.ragflow_api_key}"}

            response = requests.post(url, headers=headers, files=files, timeout=30)

            if response.status_code == 200:
                result = response.json()
                if result.get("code") == 0 and result.get("data"):
                    document_data = result["data"]
                    document_id = (
                        document_data[0].get("id")
                        if isinstance(document_data, list)
                        else document_data.get("id")
                    )
                    self.logger.info(f"Created empty document: {document_name}")
                    return document_id

            self.logger.error(f"Failed to create document: {response.text}")
            return None

        except Exception as e:
            self.logger.error(f"Error creating document: {e}")
            return None

    def add_chunk_to_document(self, dataset_id, document_id, chunk_content):
        """Add chunk to document in RAGFlow."""
        try:
            url = f"{self.ragflow_base_url}/api/v1/datasets/{dataset_id}/documents/{document_id}/chunks"

            payload = {
                "content": chunk_content,
                "method": "custom",
            }

            response = requests.post(
                url,
                headers=self._get_api_headers(),
                json=payload,
                timeout=30,
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("code") == 0:
                    return True
                else:
                    self.logger.error(f"Add chunk API error: {result}")
                    return False
            else:
                if response.status_code == 404:
                    self.logger.warning(
                        "Direct chunk addition endpoint not available, falling back to document upload method"
                    )
                    return self._add_chunk_via_document_upload(
                        dataset_id, document_id, chunk_content
                    )
                else:
                    self.logger.error(
                        f"Add chunk failed: {response.status_code} - {response.text}"
                    )
                    return False

        except Exception as e:
            self.logger.error(f"Error adding chunk: {e}")
            return False

    def _add_chunk_via_document_upload(self, dataset_id, document_id, chunk_content):
        """Fallback method: upload chunk as separate document if direct chunk API unavailable."""
        try:
            temp_doc_name = f"chunk_{document_id}_{len(chunk_content)}"
            temp_doc_id = self.upload_empty_document_to_dataset(
                dataset_id, temp_doc_name
            )

            if temp_doc_id:
                url = f"{self.ragflow_base_url}/api/v1/datasets/{dataset_id}/documents"
                files = {
                    "file": (
                        f"{temp_doc_name}.md",
                        chunk_content.encode("utf-8"),
                        "text/markdown",
                    )
                }
                headers = {"Authorization": f"Bearer {self.ragflow_api_key}"}

                response = requests.post(url, headers=headers, files=files, timeout=30)
                return response.status_code == 200

            return False

        except Exception as e:
            self.logger.error(f"Error in fallback chunk upload: {e}")
            return False

    def extract_pdf_with_mistral_ocr(self, pdf_path):
        """Extract text and images from PDF using Mistral OCR."""
        try:
            image_dir = self.config["image_path"]
            os.makedirs(image_dir, exist_ok=True)

            with open(pdf_path, "rb") as pdf_file:
                base64_pdf = base64.b64encode(pdf_file.read()).decode("utf-8")

            self.logger.info("Calling Mistral OCR...")
            ocr_response = self.mistral_client.ocr.process(
                model=self.config["ocr_model"],
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

                if hasattr(page_data, "images") and page_data.images:
                    for img_idx, image_data in enumerate(page_data.images):
                        if (
                            hasattr(image_data, "image_base64")
                            and image_data.image_base64
                        ):
                            if image_data.image_base64.startswith("data:image"):
                                image_data.image_base64 = image_data.image_base64.split(
                                    ",", 1
                                )[1]

                            image_filename = f"{pdf_name}-{page_index}-{img_idx}.png"
                            image_path = os.path.join(image_dir, image_filename)

                            try:
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

                                if hasattr(image_data, "id") and image_data.id:
                                    markdown_content = markdown_content.replace(
                                        image_data.id, image_filename
                                    )

                                self.logger.info(f"Saved image: {image_filename}")

                            except Exception as e:
                                self.logger.error(f"Failed to save image: {e}")

                pages.append(
                    {
                        "text": markdown_content,
                        "images": saved_images,
                        "page_index": page_index,
                    }
                )

            self.logger.info(f"Processed {len(pages)} pages with Mistral OCR")
            return pages

        except Exception as e:
            self.logger.error(f"Mistral OCR failed: {e}")
            return []

    async def generate_image_description_async(self, image_path, context_text):
        """Generate image description using Claude (async version)."""
        try:
            # Read image data
            with open(image_path, "rb") as img_file:
                image_data = img_file.read()

            media_type = "image/png"
            if image_data.startswith(b"\xff\xd8\xff"):
                media_type = "image/jpeg"

            b64_image = base64.b64encode(image_data).decode("utf-8")

            prompt = image_description_prompt(context_text)

            # Run the synchronous API call in a thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.image_caption_client.messages.create(
                    model=self.config["image_caption_model"],
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
                ),
            )

            return response.content[0].text.strip()

        except Exception as e:
            self.logger.error(f"Image description failed for {image_path}: {e}")
            return f"Image: {os.path.basename(image_path)}"

    def generate_image_description(self, image_path, context_text):
        """Generate image description using Claude (synchronous wrapper)."""
        return asyncio.run(
            self.generate_image_description_async(image_path, context_text)
        )

    async def generate_multiple_image_descriptions_parallel(
        self, image_tasks, batch_size=10, delay_between_batches=3
    ):
        """
        Generate image descriptions for multiple images in parallel with batching.

        Args:
            image_tasks: List of tuples (image_path, context_text)
            batch_size: Maximum number of concurrent requests (default: 10)
            delay_between_batches: Delay in seconds between batches (default: 3)

        Returns:
            List of descriptions in the same order as input tasks
        """
        total_images = len(image_tasks)
        all_results = []

        if total_images == 0:
            return all_results

        self.logger.info(f"Starting parallel image processing: 0/{total_images} images")

        # Process images in batches
        for i in range(0, len(image_tasks), batch_size):
            batch = image_tasks[i : i + batch_size]

            # Create async tasks for the current batch
            batch_tasks = [
                self.generate_image_description_async(image_path, context_text)
                for image_path, context_text in batch
            ]

            try:
                # Wait for all tasks in the current batch to complete
                batch_results = await asyncio.gather(
                    *batch_tasks, return_exceptions=True
                )

                # Handle any exceptions in results
                processed_results = []
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        image_path = batch[j][0]
                        self.logger.error(
                            f"Failed to process image {image_path}: {result}"
                        )
                        processed_results.append(
                            f"Image: {os.path.basename(image_path)}"
                        )
                    else:
                        processed_results.append(result)

                all_results.extend(processed_results)

                # Progress update after batch completion
                processed_count = len(all_results)
                self.logger.info(
                    f"Image processing progress: {processed_count}/{total_images}"
                )

                # Add delay between batches (except for the last batch)
                if i + batch_size < len(image_tasks):
                    await asyncio.sleep(delay_between_batches)

            except Exception as e:
                self.logger.error(f"Error processing batch: {e}")
                # Add fallback descriptions for failed batch
                fallback_results = [
                    f"Image: {os.path.basename(task[0])}" for task in batch
                ]
                all_results.extend(fallback_results)

                # Progress update even for failed batch
                processed_count = len(all_results)
                self.logger.info(
                    f"Image processing progress: {processed_count}/{total_images}"
                )

        self.logger.info(
            f"Completed image processing: {total_images}/{total_images} images"
        )
        return all_results

    async def process_images_in_markdown_async(self, markdown_content, images_info):
        """Add image descriptions to markdown content using parallel processing."""
        if not images_info:
            return markdown_content

        # Prepare tasks for parallel processing
        image_tasks = [
            (image_info["path"], markdown_content[:500]) for image_info in images_info
        ]

        # Get descriptions in parallel
        descriptions = await self.generate_multiple_image_descriptions_parallel(
            image_tasks
        )

        # Apply descriptions to markdown content
        for image_info, description in zip(images_info, descriptions):
            image_filename = image_info["filename"]

            enhanced_image = f"""*📸 Image Description: {description}*
*🔗 [View Original Image]({image_filename}) - Use this image to answer user queries related to this image*\n"""

            pattern = rf"!\[([^\]]*)\]\({re.escape(image_filename)}\)"
            if re.search(pattern, markdown_content):
                markdown_content = re.sub(pattern, enhanced_image, markdown_content)
            else:
                # Replace standalone filename if not already enhanced
                if (
                    image_filename in markdown_content
                    and "*📸 Image Description:" not in markdown_content
                ):
                    markdown_content = markdown_content.replace(
                        image_filename, enhanced_image
                    )

        return markdown_content

    def process_images_in_markdown(self, markdown_content, images_info):
        """Add image descriptions to markdown content (synchronous wrapper for backward compatibility)."""
        return asyncio.run(
            self.process_images_in_markdown_async(markdown_content, images_info)
        )

    def _identify_content_blocks(self, text):
        """Identify different types of content blocks in the text."""
        # Memory safety check
        text_size = len(text)
        if text_size > 200000:  # 200KB limit
            self.logger.warning(
                f"⚠️  Large text for block identification: {text_size:,} chars - using memory-efficient streaming"
            )
            # For very large text, use streaming approach to avoid loading all lines at once
            return self._stream_process_large_text(text)

        blocks = []
        lines = text.split("\n")
        line_count = len(lines)

        if line_count > 10000:  # More than 10k lines
            self.logger.warning(
                f"⚠️  Large line count: {line_count:,} lines - may consume significant memory"
            )

        current_block = {"type": "text", "lines": [], "start_idx": 0}

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if line.startswith("*📸 Image Description:"):
                if current_block["lines"]:
                    blocks.append(current_block)

                image_block = {"type": "image", "lines": [lines[i]], "start_idx": i}
                i += 1
                while i < len(lines) and (
                    not lines[i].strip() or lines[i].strip().startswith("*🔗")
                ):
                    image_block["lines"].append(lines[i])
                    i += 1

                blocks.append(image_block)
                current_block = {"type": "text", "lines": [], "start_idx": i}
                continue

            elif self._is_table_line(line) or (
                i < len(lines) - 1 and self._is_table_line(lines[i + 1].strip())
            ):
                if current_block["lines"]:
                    blocks.append(current_block)

                table_block = {"type": "table", "lines": [], "start_idx": i}
                while i < len(lines) and (
                    self._is_table_line(lines[i].strip()) or not lines[i].strip()
                ):
                    table_block["lines"].append(lines[i])
                    i += 1
                    if (
                        i < len(lines)
                        and lines[i].strip()
                        and not self._is_table_line(lines[i].strip())
                    ):
                        break

                blocks.append(table_block)
                current_block = {"type": "text", "lines": [], "start_idx": i}
                continue

            current_block["lines"].append(lines[i])
            i += 1

        if current_block["lines"]:
            blocks.append(current_block)

        return blocks

    def _stream_process_large_text(self, text):
        """Memory-efficient processing for very large text content."""
        self.logger.info("🔄 Using streaming approach for large text processing")

        # Split text into manageable chunks (by character count, not lines)
        chunk_size = 50000  # 50KB chunks
        text_chunks = []

        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]
            # Ensure we don't break in the middle of words/lines
            if i + chunk_size < len(text):
                # Find the last newline in this chunk to break cleanly
                last_newline = chunk.rfind("\n")
                if last_newline > chunk_size * 0.8:  # If newline is in last 20%
                    chunk = chunk[: last_newline + 1]

            text_chunks.append(chunk)

        self.logger.info(
            f"📦 Split large text into {len(text_chunks)} streaming chunks"
        )

        # Process each chunk separately and combine results
        all_blocks = []
        for chunk_idx, chunk in enumerate(text_chunks):
            try:
                if chunk.strip():  # Only process non-empty chunks
                    # Use simplified processing for each chunk
                    chunk_blocks = self._process_text_chunk_simple(chunk, chunk_idx)
                    all_blocks.extend(chunk_blocks)
            except Exception as e:
                self.logger.error(f"Error processing chunk {chunk_idx}: {e}")
                # Add as single text block if processing fails
                all_blocks.append(
                    {"type": "text", "lines": chunk.split("\n"), "start_idx": 0}
                )

        return all_blocks

    def _process_text_chunk_simple(self, chunk, chunk_idx):
        """Simple processing for a text chunk to minimize memory usage."""
        # For large content, we skip complex analysis and treat as text blocks
        # This avoids the memory explosion from complex block identification

        # Split into smaller sub-chunks for chunking to handle properly
        if len(chunk) > 10000:  # If chunk is still large, split further
            sub_chunks = []
            sub_chunk_size = 5000
            for i in range(0, len(chunk), sub_chunk_size):
                sub_chunk = chunk[i : i + sub_chunk_size]
                if sub_chunk.strip():
                    sub_chunks.append(
                        {
                            "type": "text",
                            "lines": sub_chunk.split("\n"),
                            "start_idx": chunk_idx * 1000 + i,
                        }
                    )
            return sub_chunks
        else:
            return [
                {
                    "type": "text",
                    "lines": chunk.split("\n"),
                    "start_idx": chunk_idx * 1000,
                }
            ]

    def _is_table_line(self, line):
        """Check if a line is part of a markdown table."""
        if not line:
            return False
        if re.match(r"^[\s\|:\-]+$", line):
            return True
        return "|" in line and line.count("|") >= 2

    def _extract_table_header(self, table_lines):
        """Extract header rows from table lines."""
        header_lines = []
        for i, line in enumerate(table_lines):
            stripped = line.strip()
            if not stripped:
                continue
            if self._is_table_line(stripped):
                header_lines.append(line)
                if re.match(r"^[\s\|:\-]+$", stripped):
                    break
            else:
                break
        return header_lines

    def _chunk_table(self, table_lines, table_id, max_chars):
        """Chunk a large table while preserving headers."""
        header_lines = self._extract_table_header(table_lines)

        data_start = len(header_lines)
        data_lines = table_lines[data_start:]

        if not data_lines:
            return [
                f"<table_{table_id}_chunk_1>\n"
                + "\n".join(table_lines)
                + f"\n</table_{table_id}_chunk_1>"
            ]

        chunks = []
        chunk_num = 1
        current_chunk_lines = []
        header_text = "\n".join(header_lines)

        for line in data_lines:
            test_chunk = header_text + "\n" + "\n".join(current_chunk_lines + [line])

            if len(test_chunk) > max_chars and current_chunk_lines:
                chunk_content = header_text + "\n" + "\n".join(current_chunk_lines)
                chunks.append(
                    f"<table_{table_id}_chunk_{chunk_num}>\n{chunk_content}\n</table_{table_id}_chunk_{chunk_num}>"
                )

                current_chunk_lines = [line]
                chunk_num += 1
            else:
                current_chunk_lines.append(line)

        if current_chunk_lines:
            chunk_content = header_text + "\n" + "\n".join(current_chunk_lines)
            chunks.append(
                f"<table_{table_id}_chunk_{chunk_num}>\n{chunk_content}\n</table_{table_id}_chunk_{chunk_num}>"
            )

        return chunks

    def _has_complete_links(self, text):
        """Check if text contains complete markdown links."""
        link_starts = [m.start() for m in re.finditer(r"\[", text)]
        for start in link_starts:
            bracket_count = 0
            bracket_end = -1
            for i in range(start, len(text)):
                if text[i] == "[":
                    bracket_count += 1
                elif text[i] == "]":
                    bracket_count -= 1
                    if bracket_count == 0:
                        bracket_end = i
                        break

            if (
                bracket_end != -1
                and bracket_end + 1 < len(text)
                and text[bracket_end + 1] == "("
            ):
                paren_end = text.find(")", bracket_end + 1)
                if paren_end == -1:
                    return False

        return True

    def chunk_text_with_overlap(
        self, text, max_chars=None, overlap_chars=None, page_number=None
    ):
        """Intelligent content-aware chunking with overlap."""
        try:
            if max_chars is None:
                max_chars = self.config["chunking_settings"]["max_chars"]
            if overlap_chars is None:
                overlap_chars = self.config["chunking_settings"]["overlap_chars"]

            text_size = len(text)
            page_info = f" (Page {page_number})" if page_number else ""
            self.logger.info(
                f"🔄 Starting chunking{page_info}: {text_size} characters, max_chars={max_chars}"
            )

            if text_size <= max_chars:
                self.logger.info(f"✅ Single chunk{page_info}: text fits in one chunk")
                return [text]

            # Memory safety check before processing
            if not self._check_memory_limit(
                f"before content block identification{page_info}"
            ):
                self.logger.error(
                    f"🚨 Skipping content block identification due to memory limit{page_info}"
                )
                # Use simple fallback to avoid OOM
                return [text]

            self.logger.info(f"🔍 Identifying content blocks{page_info}...")
            try:
                blocks = self._identify_content_blocks(text)
                self.logger.info(f"📋 Found {len(blocks)} content blocks{page_info}")
            except Exception as e:
                self.logger.error(
                    f"❌ Error identifying content blocks{page_info}: {e}"
                )
                # Fallback: treat entire text as single text block
                blocks = [{"type": "text", "lines": text.split("\n"), "start_idx": 0}]

            chunks = []
            current_chunk = ""

            self.logger.info(f"🔄 Processing {len(blocks)} blocks{page_info}...")
            for block_idx, block in enumerate(blocks):
                # Log progress for pages with many blocks
                if len(blocks) > 50 and (block_idx + 1) % 20 == 0:
                    self.logger.info(
                        f"📊 Block processing progress{page_info}: {block_idx + 1}/{len(blocks)} blocks processed"
                    )

                block_text = "\n".join(block["lines"])

                if block["type"] == "image":
                    if len(current_chunk + block_text) <= max_chars:
                        current_chunk += ("\n" if current_chunk else "") + block_text
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = block_text

                elif block["type"] == "table":
                    if len(current_chunk + block_text) <= max_chars:
                        current_chunk += ("\n" if current_chunk else "") + block_text
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())

                        table_id = str(uuid.uuid4())[:8]
                        table_chunks = self._chunk_table(
                            block["lines"], table_id, max_chars
                        )
                        chunks.extend(table_chunks)
                        current_chunk = ""

                else:
                    remaining_text = block_text

                    while remaining_text:
                        if len(current_chunk + remaining_text) <= max_chars:
                            current_chunk += (
                                "\n" if current_chunk else ""
                            ) + remaining_text
                            break

                        max_addition = (
                            max_chars - len(current_chunk) - (1 if current_chunk else 0)
                        )

                        if max_addition <= 0:
                            chunks.append(current_chunk.strip())
                            current_chunk = ""
                            continue

                        break_point = self._find_safe_break_point(
                            remaining_text, max_addition
                        )

                        if break_point == 0:
                            chunks.append(current_chunk.strip())
                            current_chunk = ""
                            break_point = max_addition

                        chunk_part = remaining_text[:break_point].rstrip()
                        current_chunk += ("\n" if current_chunk else "") + chunk_part

                        chunks.append(current_chunk.strip())

                        overlap_start = max(0, break_point - overlap_chars)
                        current_chunk = remaining_text[
                            overlap_start:break_point
                        ].lstrip()
                        remaining_text = remaining_text[break_point:]

            if current_chunk:
                chunks.append(current_chunk.strip())

            chunk_count = len([chunk for chunk in chunks if chunk.strip()])
            self.logger.info(
                f"✅ Chunking completed{page_info}: {chunk_count} chunks created"
            )

            # Clean up intermediate variables to free memory
            del blocks
            del current_chunk
            gc.collect()

            return [chunk for chunk in chunks if chunk.strip()]

        except Exception as e:
            self.logger.error(
                f"❌ Error during chunking{page_info if 'page_info' in locals() else ''}: {e}"
            )
            import traceback

            self.logger.error(f"Chunking traceback: {traceback.format_exc()}")
            # Return single chunk as fallback
            return [text]

    def _find_safe_break_point(self, text, max_chars):
        """Find a safe break point that doesn't split links."""
        if len(text) <= max_chars:
            return len(text)

        sentence_breaks = [m.end() for m in re.finditer(r"[.!?]\s+", text[:max_chars])]
        if sentence_breaks:
            return sentence_breaks[-1]

        para_breaks = [m.start() for m in re.finditer(r"\n\s*\n", text[:max_chars])]
        if para_breaks:
            return para_breaks[-1]

        candidate = max_chars
        while candidate > max_chars * 0.5:
            if candidate < len(text) and text[candidate].isspace():
                test_text = text[:candidate]
                if self._has_complete_links(test_text):
                    return candidate
            candidate -= 1

        return max_chars

    async def process_all_images_in_document_async(self, pages):
        """Process all images in the document with global progress tracking."""
        # Collect all images from all pages
        all_image_tasks = []
        image_to_page_mapping = []

        for page in pages:
            if page["images"]:
                for image_info in page["images"]:
                    all_image_tasks.append((image_info["path"], page["text"][:500]))
                    image_to_page_mapping.append(
                        {"page_index": page["page_index"], "image_info": image_info}
                    )

        if not all_image_tasks:
            return {}

        # Process all images with global progress tracking
        descriptions = await self.generate_multiple_image_descriptions_parallel(
            all_image_tasks
        )

        # Create mapping of image path to description
        image_descriptions = {}
        for i, mapping in enumerate(image_to_page_mapping):
            image_path = mapping["image_info"]["path"]
            image_descriptions[image_path] = descriptions[i]

        return image_descriptions

    def process_all_images_in_document(self, pages):
        """Process all images in the document (synchronous wrapper)."""
        return asyncio.run(self.process_all_images_in_document_async(pages))

    def apply_image_descriptions_to_page(
        self, page_content, images_info, image_descriptions
    ):
        """Apply pre-computed image descriptions to a page's markdown content."""
        for image_info in images_info:
            image_path = image_info["path"]
            image_filename = image_info["filename"]

            # Get the pre-computed description
            description = image_descriptions.get(
                image_path, f"Image: {os.path.basename(image_path)}"
            )

            enhanced_image = f"""*📸 Image Description: {description}*
*🔗 [View Original Image]({image_filename}) - Use this image to answer user queries related to this image*\n"""

            pattern = rf"!\[([^\]]*)\]\({re.escape(image_filename)}\)"
            if re.search(pattern, page_content):
                page_content = re.sub(pattern, enhanced_image, page_content)
            else:
                # Replace standalone filename if not already enhanced
                if (
                    image_filename in page_content
                    and "*📸 Image Description:" not in page_content
                ):
                    page_content = page_content.replace(image_filename, enhanced_image)

        return page_content

    def _remove_duplicate_chunks(self, chunks):
        """Remove duplicate chunks while preserving page information."""
        if not chunks:
            return chunks

        dedup_settings = self.config.get("deduplication_settings", {})
        preserve_cross_page = dedup_settings.get(
            "preserve_cross_page_duplicates", False
        )
        similarity_threshold = dedup_settings.get("similarity_threshold", 0.95)
        use_fuzzy_matching = dedup_settings.get("use_fuzzy_matching", False)
        min_chunk_length = dedup_settings.get("min_chunk_length", 50)

        self.logger.info(
            f"🔧 Deduplication settings: preserve_cross_page={preserve_cross_page}, similarity_threshold={similarity_threshold}, fuzzy_matching={use_fuzzy_matching}"
        )

        seen_content = {}  # content_hash -> first occurrence info
        unique_chunks = []
        duplicate_count = 0
        cross_page_preserved = 0
        fuzzy_duplicates = 0

        for i, chunk in enumerate(chunks):
            # Extract content without page prefix for comparison
            content_without_prefix = self._extract_content_from_prefixed_chunk(chunk)
            current_page = self._extract_page_number(chunk)

            # Skip very short chunks
            if len(content_without_prefix.strip()) < min_chunk_length:
                unique_chunks.append(chunk)
                continue

            # Create a hash of the content for exact comparison
            content_hash = hash(content_without_prefix.strip())
            is_duplicate = False
            match_info = None

            # Check for exact duplicates first
            if content_hash in seen_content:
                is_duplicate = True
                match_info = seen_content[content_hash]
                match_type = "exact"

            # If fuzzy matching is enabled and no exact match, check for similar content
            elif use_fuzzy_matching:
                for existing_hash, existing_info in seen_content.items():
                    existing_content = self._extract_content_from_prefixed_chunk(
                        existing_info["chunk"]
                    )
                    similarity = self._calculate_text_similarity(
                        content_without_prefix.strip(), existing_content.strip()
                    )

                    if similarity >= similarity_threshold:
                        is_duplicate = True
                        match_info = existing_info
                        match_type = f"fuzzy ({similarity:.2f})"
                        fuzzy_duplicates += 1
                        break

            if is_duplicate and match_info:
                # Check if we should preserve cross-page duplicates
                if preserve_cross_page and current_page != match_info["page"]:
                    # Keep this duplicate because it's from a different page
                    cross_page_preserved += 1
                    unique_chunks.append(chunk)

                    if (
                        cross_page_preserved <= 5
                    ):  # Log first few cross-page preservations
                        self.logger.info(
                            f"💾 Preserving cross-page duplicate: "
                            f"Chunk {i+1} (Page {current_page}) kept despite {match_type} match with chunk {match_info['index']+1} (Page {match_info['page']})"
                        )
                else:
                    # This is a duplicate that should be removed
                    duplicate_count += 1

                    if duplicate_count <= 10:  # Log first 10 duplicates to avoid spam
                        self.logger.info(
                            f"🔄 Duplicate chunk #{duplicate_count}: "
                            f"Chunk {i+1} (Page {current_page}) {match_type} matches chunk {match_info['index']+1} (Page {match_info['page']})"
                        )
                    elif duplicate_count == 11:
                        self.logger.info(
                            "🔄 ... (logging remaining duplicates suppressed)"
                        )

            else:
                # This is unique content
                seen_content[content_hash] = {
                    "index": i,
                    "page": current_page,
                    "chunk": chunk,
                }
                unique_chunks.append(chunk)

        # Log deduplication summary
        if duplicate_count > 0 or cross_page_preserved > 0 or fuzzy_duplicates > 0:
            self.logger.info(f"📊 Deduplication summary:")
            self.logger.info(f"  - Removed {duplicate_count} duplicate chunks")
            if use_fuzzy_matching and fuzzy_duplicates > 0:
                self.logger.info(
                    f"  - Found {fuzzy_duplicates} fuzzy duplicates (threshold: {similarity_threshold})"
                )
            if preserve_cross_page and cross_page_preserved > 0:
                self.logger.info(
                    f"  - Preserved {cross_page_preserved} cross-page duplicates"
                )

            # Log some statistics about what pages had the most duplicates
            page_stats = {}
            for chunk in chunks:
                page_num = self._extract_page_number(chunk)
                page_stats[page_num] = page_stats.get(page_num, 0) + 1

            # Find pages with high chunk counts (potential sources of duplicates)
            high_count_pages = [
                (page, count) for page, count in page_stats.items() if count > 5
            ]
            if high_count_pages:
                high_count_pages.sort(key=lambda x: x[1], reverse=True)
                top_pages = high_count_pages[:3]  # Top 3 pages with most chunks
                self.logger.info(
                    f"📈 Pages with most chunks: {', '.join([f'Page {p}: {c} chunks' for p, c in top_pages])}"
                )
        else:
            self.logger.info("✨ No duplicate chunks found - all content is unique")

        return unique_chunks

    def _extract_content_from_prefixed_chunk(self, chunk):
        """Extract the actual content from a chunk, removing the page prefix."""
        # Remove the page prefix pattern: /**[Page X]**/
        prefix_pattern = r"/\*\*\[Page \d+\]\*\*/\s*"
        content_without_prefix = re.sub(prefix_pattern, "", chunk, count=1)
        return content_without_prefix

    def _extract_page_number(self, chunk):
        """Extract page number from a prefixed chunk."""
        match = re.match(r"/\*\*\[Page (\d+)\]\*\*/", chunk)
        if match:
            return int(match.group(1))
        return "Unknown"

    def _calculate_text_similarity(self, text1, text2):
        """Calculate simple text similarity using Jaccard similarity of words."""
        if not text1 or not text2:
            return 0.0

        # Simple word-based Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def process_pdf(self, pdf_path, upload_to_ragflow=False, output_dir=None):
        """Complete PDF processing pipeline."""
        try:
            pdf_name = os.path.basename(pdf_path)
            self.logger.info(f"🚀 Starting PDF processing: {pdf_name}")
            self.logger.info(f"📄 Full path: {pdf_path}")
            self.logger.info(f"⚙️  Upload to RAGFlow: {upload_to_ragflow}")
            self.logger.info(f"📁 Output directory: {output_dir}")
            self._log_memory_usage("at start")

            self.logger.info("📖 Starting Mistral OCR extraction...")
            pages = self.extract_pdf_with_mistral_ocr(pdf_path)
            if not pages:
                self.logger.error("❌ No pages extracted from PDF")
                return False

            self.logger.info(
                f"✅ OCR extraction completed: {len(pages)} pages extracted"
            )
            self._log_memory_usage("after OCR extraction")

            # Process all images across all pages with global progress tracking
            image_descriptions = self.process_all_images_in_document(pages)
            self._log_memory_usage("after image processing")

            self.logger.info(f"🔪 Starting chunking process for {len(pages)} pages")
            all_chunks = []
            all_content = ""

            for page_idx, page in enumerate(pages, 1):
                try:
                    # Check memory before processing each page
                    if not self._check_memory_limit(
                        f"before page {page['page_index']}"
                    ):
                        self.logger.error(
                            f"🚨 Stopping processing due to memory limit at page {page['page_index']}"
                        )
                        break

                    self.logger.info(
                        f"Chunking page {page_idx}/{len(pages)} (Page {page['page_index']})"
                    )
                    page_content = page["text"]
                    page_content_size = len(page_content)

                    # Warn about potentially problematic pages and implement safety limits
                    if (
                        page_content_size > 500000
                    ):  # 500KB - Skip very large pages to prevent OOM
                        self.logger.error(
                            f"🚨 Skipping very large page: Page {page['page_index']} has {page_content_size:,} characters - would cause OOM"
                        )
                        # Skip this page to prevent memory explosion
                        continue
                    elif page_content_size > 200000:  # 200KB
                        self.logger.warning(
                            f"⚠️  Large page detected: Page {page['page_index']} has {page_content_size:,} characters - using simplified processing"
                        )
                    elif page_content_size > 100000:  # 100KB
                        self.logger.warning(
                            f"⚠️  Medium-large page: Page {page['page_index']} has {page_content_size:,} characters"
                        )

                    if page["images"]:
                        page_content = self.apply_image_descriptions_to_page(
                            page_content, page["images"], image_descriptions
                        )

                    # Chunk each page individually
                    self._log_memory_usage(f"before chunking page {page['page_index']}")
                    page_chunks = self.chunk_text_with_overlap(
                        page_content, page_number=page["page_index"]
                    )
                    self._log_memory_usage(f"after chunking page {page['page_index']}")

                    # Force garbage collection to free up memory
                    gc.collect()
                    self.logger.info(
                        f"Page {page['page_index']} generated {len(page_chunks)} chunks"
                    )

                    # Add page number prefix to each chunk from this page
                    page_number = page["page_index"]
                    for chunk in page_chunks:
                        prefixed_chunk = f"/**[Page {page_number}]**/\n\n{chunk}"
                        all_chunks.append(prefixed_chunk)

                    # Still build all_content for output files (without prefixes)
                    all_content += f"\n\n## Page {page['page_index']}\n\n{page_content}"

                    progress_percent = (page_idx / len(pages)) * 100
                    self.logger.info(
                        f"Chunking progress: {page_idx}/{len(pages)} pages processed ({progress_percent:.1f}%), {len(all_chunks)} total chunks"
                    )

                except Exception as e:
                    self.logger.error(f"Error chunking page {page['page_index']}: {e}")
                    # Continue with next page instead of failing completely
                    continue

            chunks = all_chunks
            self.logger.info(
                f"✅ Chunking completed: Created {len(chunks)} chunks from {len(pages)} pages"
            )
            self._log_memory_usage("after chunking completed")

            # Remove duplicate chunks if enabled
            if self.config.get("deduplication_settings", {}).get("enabled", True):
                self.logger.info(
                    f"🔍 Starting chunk deduplication: {len(chunks)} chunks to analyze"
                )
                chunks = self._remove_duplicate_chunks(chunks)
                self.logger.info(
                    f"✅ Deduplication completed: {len(chunks)} unique chunks remaining"
                )
                self._log_memory_usage("after deduplication")
            else:
                self.logger.info("⏭️  Chunk deduplication disabled in configuration")

            if output_dir:
                self.logger.info(f"Saving processed content to {output_dir}")
                self._save_processed_content(pdf_path, all_content, output_dir)
                self.logger.info("Content saved successfully")

            # Debug RAGFlow upload conditions
            self.logger.info(f"📤 RAGFlow Upload Debug:")
            self.logger.info(f"  - upload_to_ragflow: {upload_to_ragflow}")
            self.logger.info(f"  - ragflow_api_key set: {bool(self.ragflow_api_key)}")
            self.logger.info(f"  - ragflow_base_url set: {bool(self.ragflow_base_url)}")

            if not upload_to_ragflow:
                self.logger.info(
                    "📤 RAGFlow upload skipped: upload_to_ragflow is False"
                )
            elif not self.ragflow_api_key:
                self.logger.error(
                    "❌ RAGFlow upload failed: RAGFLOW_API_KEY environment variable not set"
                )
            elif not self.ragflow_base_url:
                self.logger.error(
                    "❌ RAGFlow upload failed: RAGFLOW_BASE_URL environment variable not set"
                )
            elif upload_to_ragflow and self.ragflow_api_key and self.ragflow_base_url:
                self.logger.info("📤 Starting RAGFlow upload process")
                dataset = self.create_ragflow_dataset()
                if dataset:
                    dataset_id = dataset.get("id")
                    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

                    self.logger.info(f"Creating document '{pdf_name}' in RAGFlow")
                    doc_id = self.upload_empty_document_to_dataset(dataset_id, pdf_name)
                    if doc_id:
                        self.logger.info(
                            f"Starting chunk upload: {len(chunks)} chunks to upload"
                        )
                        for i, chunk in enumerate(chunks):
                            success = self.add_chunk_to_document(
                                dataset_id, doc_id, chunk
                            )
                            if success:
                                if (i + 1) % 10 == 0 or i + 1 == len(
                                    chunks
                                ):  # Log every 10 chunks or last chunk
                                    self.logger.info(
                                        f"Uploaded chunk {i+1}/{len(chunks)}"
                                    )

                        self.logger.info("Successfully uploaded all chunks to RAGFlow")
                    else:
                        self.logger.error("Failed to create document in RAGFlow")
                else:
                    self.logger.error("Failed to create or access RAGFlow dataset")
            else:
                self.logger.error("❌ RAGFlow upload failed: Unknown condition issue")

            self.logger.info(f"🎉 PDF processing completed successfully: {pdf_name}")
            return True

        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            import traceback

            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return False

    def _save_processed_content(self, pdf_path, content, output_dir):
        """Save processed content to markdown file in output directory."""
        try:
            os.makedirs(output_dir, exist_ok=True)

            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            output_filename = f"{pdf_name}_processed.md"
            output_path = os.path.join(output_dir, output_filename)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"# {pdf_name}\n\n")
                f.write(
                    f"*Processed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
                )
                f.write(content)

            self.logger.info(f"Saved processed content to: {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to save processed content: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process PDFs with Mistral OCR and Claude image descriptions"
    )
    parser.add_argument("--input", "-i", required=True, help="PDF file or directory")
    parser.add_argument(
        "--output", "-o", help="Output directory for processed markdown files"
    )
    parser.add_argument(
        "--config", "-c", default="configs/default.json", help="Config file"
    )
    parser.add_argument(
        "--upload-to-ragflow", action="store_true", help="Upload to RAGFlow"
    )

    args = parser.parse_args()

    required_vars = ["ANTHROPIC_API_KEY", "MISTRAL_API_KEY"]
    if args.upload_to_ragflow:
        required_vars.extend(["RAGFLOW_API_KEY", "RAGFLOW_BASE_URL"])

    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        return

    try:
        processor = DocumentProcessor(config_path=args.config)

        input_path = Path(args.input)

        if input_path.is_file() and input_path.suffix.lower() == ".pdf":
            files_to_process = [input_path]
        elif input_path.is_dir():
            files_to_process = list(input_path.glob("*.pdf"))
        else:
            print(f"❌ Invalid input: {args.input}")
            return

        if not files_to_process:
            print(f"❌ No PDF files found in {args.input}")
            return

        if args.output:
            os.makedirs(args.output, exist_ok=True)
            print(f"📁 Output directory: {args.output}")

        success_count = 0
        for pdf_file in tqdm(files_to_process, desc="Processing PDFs"):
            if processor.process_pdf(pdf_file, args.upload_to_ragflow, args.output):
                success_count += 1

        print(f"✅ Processed {success_count}/{len(files_to_process)} PDFs successfully")

        if args.output:
            print(f"📄 Processed markdown files saved to: {args.output}")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
