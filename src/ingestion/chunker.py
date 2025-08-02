"""Intelligent content chunking with overlap and content-aware splitting."""

import re
import uuid
import logging
import tiktoken
from typing import List, Dict, Any, Optional, Tuple

from .config import IngestionConfig


class ContentChunker:
    """Intelligent chunker that respects content boundaries and user-specified rules."""

    def __init__(self, config: IngestionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        self.logger.info(
            f"🔪 ContentChunker initialized with chunk_size={config.chunk_size}, overlap={config.chunk_overlap}"
        )

    def chunk_document_pages(
        self,
        pages: List[Dict[str, Any]],
        image_descriptions: Dict[str, str],
        table_descriptions: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """Chunk all pages of a document with descriptions applied."""
        all_chunks = []

        for page in pages:
            page_content = self._apply_descriptions_to_page(
                page, image_descriptions, table_descriptions
            )

            page_chunks = self.chunk_text_with_overlap(
                page_content, page_number=page["page_index"]
            )

            all_chunks.extend(page_chunks)

        self.logger.info(
            f"✅ Document chunking completed: {len(all_chunks)} total chunks"
        )
        return all_chunks

    def _apply_descriptions_to_page(
        self,
        page: Dict[str, Any],
        image_descriptions: Dict[str, str],
        table_descriptions: Dict[str, str],
    ) -> str:
        """Apply image and table descriptions to page content."""
        content = page["text"]

        # Apply image descriptions
        for image_info in page.get("images", []):
            image_path = image_info["path"]
            image_filename = image_info["filename"]

            description = image_descriptions.get(image_path, f"Image: {image_filename}")

            # Create enhanced image description
            enhanced_description = f"<description_{image_filename}>{description}</description_{image_filename}>"

            # Replace image references
            patterns = [
                rf"!\\[([^\\]]*)\\]\\({re.escape(image_filename)}\\)",
                re.escape(image_filename),
            ]

            for pattern in patterns:
                if re.search(pattern, content):
                    content = re.sub(pattern, enhanced_description, content)
                    break

        # Apply table descriptions
        for i, table_info in enumerate(page.get("tables", [])):
            table_id = table_info.get("id", f"table_{i}")
            table_content = table_info["content"]

            table_key = f"table_{page['page_index']}_{table_id}"
            description = table_descriptions.get(
                table_key, f"Table: {table_content[:100]}..."
            )

            # Create enhanced table description
            enhanced_description = f"<description_table_{table_id}>{description}</description_table_{table_id}>"

            # Replace table content with description
            content = content.replace(table_content, enhanced_description)

        return content

    def chunk_text_with_overlap(
        self, text: str, page_number: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Intelligent content-aware chunking with overlap."""
        try:
            page_info = f" (Page {page_number})" if page_number else ""
            self.logger.info(f"🔄 Starting chunking{page_info}: {len(text)} characters")

            if len(text) <= self.config.chunk_size:
                chunk_data = self._create_chunk_metadata(
                    text, page_number, 1, len(text)
                )
                return [chunk_data]

            # Identify content blocks
            blocks = self._identify_content_blocks(text)
            self.logger.info(f"📋 Found {len(blocks)} content blocks{page_info}")

            chunks = []
            current_chunk = ""

            for block_idx, block in enumerate(blocks):
                block_text = "\\n".join(block["lines"])

                if block["type"] == "image_description":
                    chunks.extend(
                        self._process_image_description_block(
                            block_text, current_chunk, page_number, len(chunks)
                        )
                    )
                    current_chunk = ""

                elif block["type"] == "table_description":
                    chunks.extend(
                        self._process_table_description_block(
                            block_text, current_chunk, page_number, len(chunks)
                        )
                    )
                    current_chunk = ""

                elif block["type"] == "table":
                    chunks.extend(
                        self._process_table_block(
                            block, current_chunk, page_number, len(chunks)
                        )
                    )
                    current_chunk = ""

                else:  # text block
                    current_chunk, new_chunks = self._process_text_block(
                        block_text, current_chunk, page_number, len(chunks)
                    )
                    chunks.extend(new_chunks)

            # Add remaining content
            if current_chunk.strip():
                chunk_data = self._create_chunk_metadata(
                    current_chunk, page_number, len(chunks) + 1, len(current_chunk)
                )
                chunks.append(chunk_data)

            self.logger.info(
                f"✅ Chunking completed{page_info}: {len(chunks)} chunks created"
            )
            return chunks

        except Exception as e:
            self.logger.error(
                f"❌ Error during chunking{page_info if 'page_info' in locals() else ''}: {e}"
            )
            # Fallback to simple chunking
            return self._simple_chunk_fallback(text, page_number)

    def _identify_content_blocks(self, text: str) -> List[Dict[str, Any]]:
        """Identify different types of content blocks."""
        blocks = []
        lines = text.split("\\n")
        current_block = {"type": "text", "lines": [], "start_idx": 0}

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Check for image description blocks
            if (
                line.startswith("<description_")
                and line.endswith(">")
                and "image" in line
            ):
                if current_block["lines"]:
                    blocks.append(current_block)

                desc_block = {"type": "image_description", "lines": [], "start_idx": i}

                # Find the end of description block
                while i < len(lines):
                    desc_block["lines"].append(lines[i])
                    if (
                        lines[i].strip().endswith("</description_")
                        or i == len(lines) - 1
                    ):
                        i += 1
                        break
                    i += 1

                blocks.append(desc_block)
                current_block = {"type": "text", "lines": [], "start_idx": i}
                continue

            # Check for table description blocks
            elif line.startswith("<description_table_"):
                if current_block["lines"]:
                    blocks.append(current_block)

                desc_block = {"type": "table_description", "lines": [], "start_idx": i}

                # Find the end of description block
                while i < len(lines):
                    desc_block["lines"].append(lines[i])
                    if (
                        lines[i].strip().endswith("</description_table_")
                        or i == len(lines) - 1
                    ):
                        i += 1
                        break
                    i += 1

                blocks.append(desc_block)
                current_block = {"type": "text", "lines": [], "start_idx": i}
                continue

            # Check for table blocks
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

    def _process_image_description_block(
        self,
        block_text: str,
        current_chunk: str,
        page_number: Optional[int],
        chunk_count: int,
    ) -> List[Dict[str, Any]]:
        """Process image description block according to chunking rules."""
        chunks = []

        # Check if description fits with current chunk
        combined_length = len(current_chunk) + len(block_text)

        if combined_length <= self.config.chunk_size and current_chunk:
            # Fits with current chunk
            new_chunk = current_chunk + "\\n" + block_text
            chunk_data = self._create_chunk_metadata(
                new_chunk, page_number, chunk_count + 1, len(new_chunk)
            )
            chunks.append(chunk_data)
        else:
            # Add current chunk if exists
            if current_chunk.strip():
                chunk_data = self._create_chunk_metadata(
                    current_chunk, page_number, chunk_count + 1, len(current_chunk)
                )
                chunks.append(chunk_data)
                chunk_count += 1

            # Check if description needs chunking
            if len(block_text) > self.config.chunk_size:
                chunks.extend(
                    self._chunk_large_description(
                        block_text, "image", page_number, chunk_count
                    )
                )
            else:
                chunk_data = self._create_chunk_metadata(
                    block_text, page_number, chunk_count + 1, len(block_text)
                )
                chunks.append(chunk_data)

        return chunks

    def _process_table_description_block(
        self,
        block_text: str,
        current_chunk: str,
        page_number: Optional[int],
        chunk_count: int,
    ) -> List[Dict[str, Any]]:
        """Process table description block according to chunking rules."""
        chunks = []

        # Check if description fits with current chunk
        combined_length = len(current_chunk) + len(block_text)

        if combined_length <= self.config.chunk_size and current_chunk:
            # Fits with current chunk
            new_chunk = current_chunk + "\\n" + block_text
            chunk_data = self._create_chunk_metadata(
                new_chunk, page_number, chunk_count + 1, len(new_chunk)
            )
            chunks.append(chunk_data)
        else:
            # Add current chunk if exists
            if current_chunk.strip():
                chunk_data = self._create_chunk_metadata(
                    current_chunk, page_number, chunk_count + 1, len(current_chunk)
                )
                chunks.append(chunk_data)
                chunk_count += 1

            # Check if description needs chunking
            if len(block_text) > self.config.chunk_size:
                chunks.extend(
                    self._chunk_large_description(
                        block_text, "table", page_number, chunk_count
                    )
                )
            else:
                chunk_data = self._create_chunk_metadata(
                    block_text, page_number, chunk_count + 1, len(block_text)
                )
                chunks.append(chunk_data)

        return chunks

    def _process_table_block(
        self,
        block: Dict[str, Any],
        current_chunk: str,
        page_number: Optional[int],
        chunk_count: int,
    ) -> List[Dict[str, Any]]:
        """Process table block with row-aware chunking."""
        chunks = []
        block_text = "\\n".join(block["lines"])

        # Add current chunk if exists
        if current_chunk.strip():
            chunk_data = self._create_chunk_metadata(
                current_chunk, page_number, chunk_count + 1, len(current_chunk)
            )
            chunks.append(chunk_data)
            chunk_count += 1

        # Check if table needs chunking
        if len(block_text) > self.config.chunk_size:
            chunks.extend(
                self._chunk_large_table(block["lines"], page_number, chunk_count)
            )
        else:
            chunk_data = self._create_chunk_metadata(
                block_text, page_number, chunk_count + 1, len(block_text)
            )
            chunks.append(chunk_data)

        return chunks

    def _process_text_block(
        self,
        block_text: str,
        current_chunk: str,
        page_number: Optional[int],
        chunk_count: int,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Process regular text block with overlap."""
        chunks = []
        remaining_text = block_text

        while remaining_text:
            # Check if remaining text fits with current chunk
            if len(current_chunk + remaining_text) <= self.config.chunk_size:
                current_chunk += ("\\n" if current_chunk else "") + remaining_text
                break

            # Find safe break point
            max_addition = (
                self.config.chunk_size
                - len(current_chunk)
                - (1 if current_chunk else 0)
            )

            if max_addition <= 0:
                # Current chunk is full, finalize it
                chunk_data = self._create_chunk_metadata(
                    current_chunk,
                    page_number,
                    chunk_count + len(chunks) + 1,
                    len(current_chunk),
                )
                chunks.append(chunk_data)
                current_chunk = ""
                continue

            break_point = self._find_safe_break_point(remaining_text, max_addition)

            if break_point == 0:
                # Can't fit any more, finalize current chunk
                chunk_data = self._create_chunk_metadata(
                    current_chunk,
                    page_number,
                    chunk_count + len(chunks) + 1,
                    len(current_chunk),
                )
                chunks.append(chunk_data)
                current_chunk = ""
                break_point = max_addition

            # Extract chunk part
            chunk_part = remaining_text[:break_point].rstrip()
            current_chunk += ("\\n" if current_chunk else "") + chunk_part

            # Finalize chunk
            chunk_data = self._create_chunk_metadata(
                current_chunk,
                page_number,
                chunk_count + len(chunks) + 1,
                len(current_chunk),
            )
            chunks.append(chunk_data)

            # Create overlap for next chunk
            overlap_start = max(0, break_point - self.config.chunk_overlap)
            current_chunk = remaining_text[overlap_start:break_point].lstrip()
            remaining_text = remaining_text[break_point:]

        return current_chunk, chunks

    def _chunk_large_description(
        self,
        description: str,
        desc_type: str,
        page_number: Optional[int],
        chunk_count: int,
    ) -> List[Dict[str, Any]]:
        """Chunk large descriptions into multiple parts."""
        chunks = []

        # Extract reference from description tags
        if desc_type == "image":
            match = re.search(r"<description_([^>]+)>", description)
        else:  # table
            match = re.search(r"<description_table_([^>]+)>", description)

        reference = match.group(1) if match else f"{desc_type}_ref"

        # Remove tags for chunking
        content = re.sub(r"</?description_[^>]*>", "", description).strip()

        # Chunk the content
        chunk_size = self.config.chunk_size - 100  # Account for wrapper tags
        chunk_num = 1

        for i in range(0, len(content), chunk_size):
            chunk_content = content[i : i + chunk_size]

            # Wrap in appropriate tags
            wrapped_chunk = f"<description_{reference}_chunk_{chunk_num}>{chunk_content}</description_{reference}_chunk_{chunk_num}>"

            chunk_data = self._create_chunk_metadata(
                wrapped_chunk, page_number, chunk_count + chunk_num, len(wrapped_chunk)
            )
            chunks.append(chunk_data)
            chunk_num += 1

        return chunks

    def _chunk_large_table(
        self, table_lines: List[str], page_number: Optional[int], chunk_count: int
    ) -> List[Dict[str, Any]]:
        """Chunk large table without breaking rows."""
        chunks = []

        # Extract header rows
        header_lines = []
        data_start = 0

        for i, line in enumerate(table_lines):
            stripped = line.strip()
            if not stripped:
                continue
            if self._is_table_line(stripped):
                header_lines.append(line)
                if re.match(r"^[\\s\\|:\\-]+$", stripped):
                    data_start = i + 1
                    break
            else:
                break

        # Generate table reference
        table_ref = str(uuid.uuid4())[:8]
        header_text = "\\n".join(header_lines)
        data_lines = table_lines[data_start:]

        if not data_lines:
            # Table has only headers
            chunk_data = self._create_chunk_metadata(
                "\\n".join(table_lines),
                page_number,
                chunk_count + 1,
                len("\\n".join(table_lines)),
            )
            return [chunk_data]

        # Chunk data rows while preserving headers
        chunk_num = 1
        current_rows = []

        for line in data_lines:
            test_chunk = header_text + "\\n" + "\\n".join(current_rows + [line])

            if len(test_chunk) > self.config.chunk_size and current_rows:
                # Create chunk with current rows
                chunk_content = header_text + "\\n" + "\\n".join(current_rows)
                wrapped_chunk = f"<table_{table_ref}_chunk_{chunk_num}>\\n{chunk_content}\\n</table_{table_ref}_chunk_{chunk_num}>"

                chunk_data = self._create_chunk_metadata(
                    wrapped_chunk,
                    page_number,
                    chunk_count + chunk_num,
                    len(wrapped_chunk),
                )
                chunks.append(chunk_data)

                current_rows = [line]
                chunk_num += 1
            else:
                current_rows.append(line)

        # Add remaining rows
        if current_rows:
            chunk_content = header_text + "\\n" + "\\n".join(current_rows)
            wrapped_chunk = f"<table_{table_ref}_chunk_{chunk_num}>\\n{chunk_content}\\n</table_{table_ref}_chunk_{chunk_num}>"

            chunk_data = self._create_chunk_metadata(
                wrapped_chunk, page_number, chunk_count + chunk_num, len(wrapped_chunk)
            )
            chunks.append(chunk_data)

        return chunks

    def _find_safe_break_point(self, text: str, max_chars: int) -> int:
        """Find safe break point that doesn't split words or links."""
        if len(text) <= max_chars:
            return len(text)

        # Check if we're breaking inside a link
        if not self._is_safe_link_break(text, max_chars):
            # Find the start of the current link to avoid breaking it
            link_safe_point = self._find_link_safe_point(text, max_chars)
            if link_safe_point is not None:
                max_chars = link_safe_point

        # Check if we're breaking inside a file path
        if not self._is_safe_filepath_break(text, max_chars):
            # Find the start of the current file path to avoid breaking it
            filepath_safe_point = self._find_filepath_safe_point(text, max_chars)
            if filepath_safe_point is not None:
                max_chars = filepath_safe_point

        # Look for sentence breaks
        sentence_breaks = [m.end() for m in re.finditer(r"[.!?]\\s+", text[:max_chars])]
        if sentence_breaks:
            return sentence_breaks[-1]

        # Look for paragraph breaks
        para_breaks = [m.start() for m in re.finditer(r"\\n\\s*\\n", text[:max_chars])]
        if para_breaks:
            return para_breaks[-1]

        # Look for word boundaries
        candidate = max_chars
        while candidate > max_chars * 0.5:
            if candidate < len(text) and text[candidate].isspace():
                # Double-check this doesn't break a link or file path
                if self._is_safe_link_break(
                    text, candidate
                ) and self._is_safe_filepath_break(text, candidate):
                    return candidate
            candidate -= 1

        return max_chars

    def _is_safe_link_break(self, text: str, break_point: int) -> bool:
        """Check if breaking at this point would split a markdown link."""
        # Look for markdown links around the break point
        before_text = text[:break_point]
        after_text = text[break_point:]

        # Check if we're in the middle of a markdown link [text](url)
        open_brackets = before_text.count("[") - before_text.count("]")
        if open_brackets > 0:
            # We might be inside a link text
            return False

        # Check if we have an unclosed parenthesis that might be part of a link
        if (
            "](" in before_text[-50:]
            and ")" not in before_text[-50:]
            and ")" in after_text[:50]
        ):
            return False

        # Check for URL patterns that shouldn't be broken
        url_pattern = r"https?://[^\s\)]*"
        for match in re.finditer(url_pattern, text):
            if match.start() < break_point < match.end():
                return False

        # Check for file paths that shouldn't be broken
        if not self._is_safe_filepath_break(text, break_point):
            return False

        return True

    def _find_link_safe_point(self, text: str, max_chars: int) -> Optional[int]:
        """Find a safe point before a link to avoid breaking it."""
        # Look for the start of a link that encompasses our break point
        link_patterns = [
            r"\[([^\]]*)\]\([^\)]*\)",  # [text](url)
            r"https?://[^\s\)]*",  # Direct URLs
        ]

        for pattern in link_patterns:
            for match in re.finditer(pattern, text):
                if match.start() < max_chars < match.end():
                    # Find a safe break point before this link
                    safe_point = match.start()
                    # Look for a good break point before the link
                    while safe_point > 0 and not text[safe_point - 1].isspace():
                        safe_point -= 1
                    if safe_point > max_chars * 0.5:  # Only if it's not too far back
                        return safe_point

        return None

    def _is_safe_filepath_break(self, text: str, break_point: int) -> bool:
        """Check if breaking at this point would split a file path."""
        # Look for file paths around the break point
        before_text = text[:break_point]
        after_text = text[break_point:]

        # Check for markdown image references ![alt](path)
        try:
            img_pattern = r"!\[[^\]]*\]\([^\)]*\)"
            for match in re.finditer(img_pattern, text):
                if match.start() < break_point < match.end():
                    return False
        except re.error:
            # If regex fails, skip this check
            pass

        # Check for file paths with common patterns
        file_patterns = [
            r"files/[^\s\)\]]+\.[a-zA-Z0-9]+",  # files/filename.ext
            r"data/[^\s\)\]]+\.[a-zA-Z0-9]+",  # data/filename.ext
            r"[^\s\(\)\[\]]+\.[a-zA-Z0-9]+",  # generic filename.ext
        ]

        # Look for file paths that span the break point
        context_before = before_text[-50:] if len(before_text) >= 50 else before_text
        context_after = after_text[:50] if len(after_text) >= 50 else after_text
        context = context_before + context_after

        for pattern in file_patterns:
            try:
                for match in re.finditer(pattern, context):
                    # Calculate actual position in the full text
                    match_start = len(before_text) - len(context_before) + match.start()
                    match_end = len(before_text) - len(context_before) + match.end()

                    if match_start < break_point < match_end:
                        return False
            except re.error:
                # If regex fails, skip this pattern
                continue

        return True

    def _find_filepath_safe_point(self, text: str, max_chars: int) -> Optional[int]:
        """Find a safe point before a file path to avoid breaking it."""
        # Look for file paths that encompass our break point
        file_patterns = [
            r"!\[[^\]]*\]\([^\)]*\)",  # ![alt](path)
            r"files/[^\s\)\]]+\.[a-zA-Z0-9]+",  # files/filename.ext
            r"data/[^\s\)\]]+\.[a-zA-Z0-9]+",  # data/filename.ext
            r"[^\s\(\)\[\]]+\.[a-zA-Z0-9]+",  # generic filename.ext
        ]

        for pattern in file_patterns:
            try:
                for match in re.finditer(pattern, text):
                    if match.start() < max_chars < match.end():
                        # Find a safe break point before this file path
                        safe_point = match.start()
                        # Look for a good break point before the path
                        while safe_point > 0 and not text[safe_point - 1].isspace():
                            safe_point -= 1
                        if (
                            safe_point > max_chars * 0.5
                        ):  # Only if it's not too far back
                            return safe_point
            except re.error:
                # If regex fails, skip this pattern
                continue

        return None

    def _extract_links_from_content(self, content: str) -> List[Dict[str, str]]:
        """Extract all links from content for metadata."""
        links = []

        # Extract markdown links [text](url)
        markdown_links = re.finditer(r"\[([^\]]+)\]\(([^\)]+)\)", content)
        for match in markdown_links:
            links.append(
                {
                    "type": "markdown",
                    "text": match.group(1),
                    "url": match.group(2),
                    "full_match": match.group(0),
                }
            )

        # Extract direct URLs
        url_pattern = r"https?://[^\s\)\]>]*"
        url_links = re.finditer(url_pattern, content)
        for match in url_links:
            url = match.group(0)
            # Skip if it's already part of a markdown link
            if not any(link["url"] == url for link in links):
                links.append(
                    {"type": "url", "text": url, "url": url, "full_match": url}
                )

        # Extract email addresses
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        email_links = re.finditer(email_pattern, content)
        for match in email_links:
            email = match.group(0)
            links.append(
                {
                    "type": "email",
                    "text": email,
                    "url": f"mailto:{email}",
                    "full_match": email,
                }
            )

        return links

    def _extract_filepaths_from_content(self, content: str) -> List[Dict[str, str]]:
        """Extract all file paths from content for metadata."""
        filepaths = []

        # Extract markdown image references ![alt](path)
        img_refs = re.finditer(r"!\[([^\]]*)\]\(([^\)]+)\)", content)
        for match in img_refs:
            filepaths.append(
                {
                    "type": "image_reference",
                    "alt_text": match.group(1),
                    "path": match.group(2),
                    "full_match": match.group(0),
                }
            )

        # Extract file paths with extensions
        file_patterns = [
            (r"files/[^\s\)\]]+\.[a-zA-Z0-9]+", "files_directory"),
            (r"data/[^\s\)\]]+\.[a-zA-Z0-9]+", "data_directory"),
            (r"[^\s\(\)\[\]]+\.[a-zA-Z0-9]{1,4}(?=\s|$|[^\w.])", "generic_file"),
        ]

        for pattern, file_type in file_patterns:
            try:
                for match in re.finditer(pattern, content):
                    filepath = match.group(0)
                    # Skip if it's already captured as an image reference
                    if not any(fp["path"] == filepath for fp in filepaths):
                        filepaths.append(
                            {
                                "type": file_type,
                                "path": filepath,
                                "full_match": filepath,
                            }
                        )
            except re.error:
                # If regex fails, skip this pattern
                continue

        return filepaths

    def _is_table_line(self, line: str) -> bool:
        """Check if line is part of a markdown table."""
        if not line:
            return False
        if re.match(r"^[\\s\\|:\\-]+$", line):
            return True
        return "|" in line and line.count("|") >= 2

    def _create_chunk_metadata(
        self,
        content: str,
        page_number: Optional[int],
        chunk_number: int,
        chunk_size: int,
    ) -> Dict[str, Any]:
        """Create chunk with metadata."""
        # Count tokens
        try:
            token_count = len(self.tokenizer.encode(content))
        except Exception:
            # Fallback token estimation
            token_count = len(content.split()) * 1.3

        metadata = {
            "chunk_number": chunk_number,
            "chunk_size": chunk_size,
            "token_count": int(token_count),
        }

        if page_number is not None:
            metadata["page_number"] = page_number

        # Extract and add links metadata
        links = self._extract_links_from_content(content)
        if links:
            metadata["links"] = links
            metadata["link_count"] = len(links)

        # Extract and add file paths metadata
        filepaths = self._extract_filepaths_from_content(content)
        if filepaths:
            metadata["filepaths"] = filepaths
            metadata["filepath_count"] = len(filepaths)

        # Check for table references in content
        table_matches = re.findall(r"<table_([^>]+)_chunk_([^>]+)>", content)
        if table_matches:
            metadata["table_reference"] = table_matches[0][0]
            metadata["table_chunk_number"] = table_matches[0][1]

        # Check for image/description references
        desc_matches = re.findall(r"<description_([^>]+)(?:_chunk_([^>]+))?>", content)
        if desc_matches:
            metadata["description_reference"] = desc_matches[0][0]
            if desc_matches[0][1]:
                metadata["description_chunk_number"] = desc_matches[0][1]

        return {"content": content, "metadata": metadata}

    def _simple_chunk_fallback(
        self, text: str, page_number: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Simple fallback chunking if main chunking fails."""
        chunks = []
        chunk_size = self.config.chunk_size

        for i in range(0, len(text), chunk_size):
            chunk_content = text[i : i + chunk_size]
            chunk_data = self._create_chunk_metadata(
                chunk_content, page_number, len(chunks) + 1, len(chunk_content)
            )
            chunks.append(chunk_data)

        return chunks
