"""Enhanced Contextual Retrieval implementation for large documents."""

import asyncio
import hashlib
import logging
import re
import time
import tiktoken
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from openai import OpenAI
from langchain_core.documents import Document


@dataclass
class DocumentStructure:
    """Container for document structure analysis."""

    title: str
    sections: List[Dict[str, Any]]
    summary: str
    document_type: str
    total_length: int
    estimated_reading_time: int


@dataclass
class SectionInfo:
    """Container for section information."""

    title: str
    level: int
    start_position: int
    end_position: int
    content_preview: str


@dataclass
class ContextualChunk:
    """Enhanced container for a chunk with its contextual information."""

    original_content: str
    contextualized_content: str
    context: str
    chunk_id: str
    source_document: str
    metadata: Dict
    # Enhanced fields for large documents
    document_summary: Optional[str] = None
    section_context: Optional[str] = None
    document_position: Optional[float] = None  # Position in document (0.0 to 1.0)
    section_info: Optional[SectionInfo] = None
    previous_chunk_context: Optional[str] = None
    next_chunk_context: Optional[str] = None


class PromptCache:
    """Simple in-memory prompt cache for OpenAI requests."""

    def __init__(self, ttl_seconds: int = 3600):
        self.cache: Dict[str, Tuple[str, float]] = {}
        self.ttl_seconds = ttl_seconds
        self.logger = logging.getLogger(__name__)

    def _get_cache_key(self, document_content: str, chunk_content: str) -> str:
        """Generate cache key for document + chunk combination."""
        combined = f"{document_content}||{chunk_content}"
        return hashlib.md5(combined.encode()).hexdigest()

    def get(self, document_content: str, chunk_content: str) -> Optional[str]:
        """Get cached contextual information."""
        key = self._get_cache_key(document_content, chunk_content)
        if key in self.cache:
            context, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                self.logger.debug(f"Cache hit for chunk: {chunk_content[:100]}...")
                return context
            else:
                del self.cache[key]
                self.logger.debug(f"Cache expired for chunk: {chunk_content[:100]}...")
        return None

    def set(self, document_content: str, chunk_content: str, context: str):
        """Cache contextual information."""
        key = self._get_cache_key(document_content, chunk_content)
        self.cache[key] = (context, time.time())
        self.logger.debug(f"Cached context for chunk: {chunk_content[:100]}...")


class EnhancedContextualRetrieval:
    """
    Enhanced Contextual Retrieval implementation for large documents.

    This class generates contextual information for document chunks to improve
    retrieval accuracy by providing chunk-specific explanatory context with
    enhanced document structure awareness and section-based contextualization.
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.client = OpenAI(api_key=config.openai_api_key)

        # Initialize cache if enabled
        self.cache = None
        if config.enable_prompt_caching:
            self.cache = PromptCache(ttl_seconds=config.cache_ttl_seconds)
            self.logger.info("✅ Prompt caching enabled")

        # Semaphore to limit concurrent requests
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)

        # Document structure cache
        self.document_structure_cache = {}

        # Token management for embedding model
        self.embedding_tokenizer = tiktoken.encoding_for_model("text-embedding-3-large")
        self.max_embedding_tokens = min(
            config.max_tokens_per_chunk, 8192
        )  # Use configured limit or API limit
        self.token_safety_margin = (
            100  # Safety margin to account for tokenization differences
        )

    def _analyze_document_structure(
        self, document_content: str, source_document: str
    ) -> DocumentStructure:
        """Analyze document structure to understand sections and overall organization."""
        if source_document in self.document_structure_cache:
            return self.document_structure_cache[source_document]

        # Extract sections using regex patterns
        sections = []

        # Common heading patterns
        heading_patterns = [
            r"^#{1,6}\s+(.+)$",  # Markdown headers
            r"^(.+)\n[=-]{3,}$",  # Underlined headers
            r"^\d+\.\s+(.+)$",  # Numbered sections
            r"^[A-Z][A-Z\s]{5,}$",  # ALL CAPS headers
        ]

        lines = document_content.split("\n")
        current_position = 0

        for i, line in enumerate(lines):
            for pattern in heading_patterns:
                if re.match(pattern, line.strip(), re.MULTILINE):
                    sections.append(
                        {
                            "title": line.strip(),
                            "level": self._determine_heading_level(line),
                            "position": current_position,
                            "line_number": i,
                        }
                    )
                    break
            current_position += len(line) + 1

        # Generate document summary
        summary_prompt = f"""Analyze this document and provide a concise summary in 2-3 sentences that captures the main purpose, topic, and key information:

<document>
{document_content[:3000]}{"..." if len(document_content) > 3000 else ""}
</document>

Provide only the summary, nothing else."""

        try:
            response = self.client.chat.completions.create(
                model=self.config.document_analysis_model,
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=self.config.document_summary_size,
                temperature=0.1,
            )
            summary = response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"Error generating document summary: {e}")
            summary = f"Document from {source_document}"

        # Determine document type
        doc_type = self._determine_document_type(document_content)

        structure = DocumentStructure(
            title=self._extract_document_title(document_content),
            sections=sections,
            summary=summary,
            document_type=doc_type,
            total_length=len(document_content),
            estimated_reading_time=len(document_content)
            // 250,  # ~250 words per minute
        )

        self.document_structure_cache[source_document] = structure
        return structure

    def _determine_heading_level(self, line: str) -> int:
        """Determine the hierarchical level of a heading."""
        line = line.strip()
        if line.startswith("#"):
            return len(line) - len(line.lstrip("#"))
        elif re.match(r"^\d+\.\s+", line):
            return 1
        elif line.isupper() and len(line) > 5:
            return 1
        return 2

    def _extract_document_title(self, document_content: str) -> str:
        """Extract document title from content."""
        lines = document_content.strip().split("\n")
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if line and not line.startswith("#"):
                return line[:100]
            elif line.startswith("# "):
                return line[2:].strip()[:100]
        return "Untitled Document"

    def _determine_document_type(self, document_content: str) -> str:
        """Determine the type of document based on content patterns."""
        content_lower = document_content.lower()

        if any(
            keyword in content_lower
            for keyword in [
                "financial statement",
                "balance sheet",
                "income statement",
                "cash flow",
            ]
        ):
            return "financial_report"
        elif any(
            keyword in content_lower
            for keyword in ["research", "abstract", "methodology", "conclusion"]
        ):
            return "research_paper"
        elif any(
            keyword in content_lower
            for keyword in ["manual", "instructions", "guide", "tutorial"]
        ):
            return "manual"
        elif any(
            keyword in content_lower
            for keyword in ["contract", "agreement", "terms", "conditions"]
        ):
            return "legal_document"
        else:
            return "general_document"

    def _build_contextualization_prompt(
        self, document_content: str, chunk_content: str
    ) -> str:
        """
        Build prompt for contextualizing a chunk within its document.

        Based on Anthropic's contextual retrieval prompt template.
        """
        return f"""<document>
{document_content}
</document>

Here is the chunk we want to situate within the whole document:

<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

    def _build_enhanced_contextualization_prompt(
        self,
        document_structure: DocumentStructure,
        chunk_content: str,
        section_info: Optional[SectionInfo] = None,
        document_position: Optional[float] = None,
        prev_chunk_context: Optional[str] = None,
        next_chunk_context: Optional[str] = None,
    ) -> str:
        """
        Build enhanced prompt for contextualizing a chunk with document structure information.
        """
        context_parts = []

        # Document overview
        context_parts.append(f"Document: {document_structure.title}")
        context_parts.append(f"Type: {document_structure.document_type}")
        context_parts.append(f"Summary: {document_structure.summary}")

        # Position information
        if document_position is not None:
            position_desc = (
                "beginning"
                if document_position < 0.3
                else "middle"
                if document_position < 0.7
                else "end"
            )
            context_parts.append(
                f"Position: {position_desc} of document ({document_position:.1%})"
            )

        # Section information
        if section_info:
            context_parts.append(
                f"Section: {section_info.title} (Level {section_info.level})"
            )

        # Adjacent context
        adjacent_context = []
        if prev_chunk_context:
            adjacent_context.append(f"Previous context: {prev_chunk_context}")
        if next_chunk_context:
            adjacent_context.append(f"Following context: {next_chunk_context}")

        if adjacent_context:
            context_parts.extend(adjacent_context)

        document_context = "\n".join(context_parts)

        return f"""<document_context>
{document_context}
</document_context>

Here is the chunk we want to situate within the document:

<chunk>
{chunk_content}
</chunk>

Based on the document context provided above, give a short succinct context (50-100 tokens) to situate this chunk within the overall document. Include relevant information about the document type, section, and position that would help with search retrieval. Answer only with the context and nothing else."""

    async def _generate_context_async(
        self, document_content: str, chunk_content: str
    ) -> str:
        """Generate contextual information for a chunk asynchronously."""
        async with self.semaphore:
            # Check cache first
            if self.cache:
                cached_context = self.cache.get(document_content, chunk_content)
                if cached_context:
                    return cached_context

            prompt = self._build_contextualization_prompt(
                document_content, chunk_content
            )

            try:
                # Use OpenAI completion
                response = self.client.chat.completions.create(
                    model=self.config.contextual_retrieval_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.config.contextual_context_size,
                    temperature=0.1,
                )

                context = response.choices[0].message.content.strip()

                # Cache the result
                if self.cache:
                    self.cache.set(document_content, chunk_content, context)

                return context

            except Exception as e:
                self.logger.error(f"Error generating context: {e}")
                return f"This chunk is from the document: {chunk_content[:50]}..."

    async def _generate_enhanced_context_async(
        self,
        document_structure: DocumentStructure,
        chunk_content: str,
        section_info: Optional[SectionInfo] = None,
        document_position: Optional[float] = None,
        prev_chunk_context: Optional[str] = None,
        next_chunk_context: Optional[str] = None,
    ) -> str:
        """Generate enhanced contextual information for a chunk with document structure."""
        async with self.semaphore:
            cache_key = f"enhanced_{document_structure.title}_{hash(chunk_content)}"

            # Check cache first
            if self.cache:
                cached_context = self.cache.get(document_structure.title, chunk_content)
                if cached_context:
                    return cached_context

            prompt = self._build_enhanced_contextualization_prompt(
                document_structure,
                chunk_content,
                section_info,
                document_position,
                prev_chunk_context,
                next_chunk_context,
            )

            try:
                # Use enhanced context size for large documents
                max_tokens = (
                    self.config.enhanced_context_size
                    if len(document_structure.summary)
                    > self.config.long_document_threshold
                    else self.config.contextual_context_size
                )

                response = self.client.chat.completions.create(
                    model=self.config.contextual_retrieval_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.1,
                )

                context = response.choices[0].message.content.strip()

                # Cache the result
                if self.cache:
                    self.cache.set(document_structure.title, chunk_content, context)

                return context

            except Exception as e:
                self.logger.error(f"Error generating enhanced context: {e}")
                fallback_context = f"This chunk is from the {document_structure.document_type} '{document_structure.title}'"
                if section_info:
                    fallback_context += f" in section '{section_info.title}'"
                return fallback_context

    def _find_section_for_chunk(
        self,
        chunk_content: str,
        document_structure: DocumentStructure,
        chunk_position: int,
    ) -> Optional[SectionInfo]:
        """Find the section that contains a given chunk."""
        if not document_structure.sections:
            return None

        # Find the section that contains this chunk position
        containing_section = None
        for section in document_structure.sections:
            if section["position"] <= chunk_position:
                containing_section = section
            else:
                break

        if containing_section:
            return SectionInfo(
                title=containing_section["title"],
                level=containing_section["level"],
                start_position=containing_section["position"],
                end_position=containing_section.get(
                    "end_position", document_structure.total_length
                ),
                content_preview=containing_section["title"][:100],
            )

        return None

    def _calculate_document_position(
        self, chunk_position: int, document_length: int
    ) -> float:
        """Calculate the relative position of a chunk in the document (0.0 to 1.0)."""
        if document_length == 0:
            return 0.0
        return min(1.0, max(0.0, chunk_position / document_length))

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the embedding model's tokenizer."""
        return len(self.embedding_tokenizer.encode(text))

    def _create_token_safe_context(
        self,
        chunk_content: str,
        context: str,
        document_summary: str = "",
        section_context: str = "",
    ) -> str:
        """
        Create contextualized content that respects token limits.

        Priority order:
        1. Original chunk content (highest priority)
        2. Context from LLM
        3. Section context
        4. Document summary (lowest priority)
        """
        # Start with the original chunk content (this is mandatory)
        chunk_tokens = self._count_tokens(chunk_content)
        available_tokens = (
            self.max_embedding_tokens - self.token_safety_margin - chunk_tokens
        )

        if available_tokens <= 0:
            # If chunk itself is too large, truncate it
            self.logger.warning(
                f"Chunk too large ({chunk_tokens} tokens), truncating..."
            )
            max_chunk_tokens = (
                self.max_embedding_tokens - self.token_safety_margin - 50
            )  # Reserve 50 for minimal context

            # Truncate chunk content
            chunk_tokens_list = self.embedding_tokenizer.encode(chunk_content)
            truncated_tokens = chunk_tokens_list[:max_chunk_tokens]
            chunk_content = self.embedding_tokenizer.decode(truncated_tokens)
            available_tokens = 50  # Minimal space for context

        # Build context pieces in priority order
        context_pieces = []
        remaining_tokens = available_tokens

        # 1. Context from LLM (highest priority after chunk)
        if context and remaining_tokens > 0:
            context_tokens = self._count_tokens(context)
            if context_tokens <= remaining_tokens:
                context_pieces.append(context)
                remaining_tokens -= context_tokens
            else:
                # Truncate context if needed
                context_tokens_list = self.embedding_tokenizer.encode(context)
                truncated_context_tokens = context_tokens_list[:remaining_tokens]
                truncated_context = self.embedding_tokenizer.decode(
                    truncated_context_tokens
                )
                context_pieces.append(truncated_context)
                remaining_tokens = 0

        # 2. Section context (medium priority)
        if section_context and remaining_tokens > 10:  # Reserve at least 10 tokens
            section_tokens = self._count_tokens(section_context)
            if section_tokens <= remaining_tokens - 10:
                context_pieces.append(f"Section: {section_context}")
                remaining_tokens -= section_tokens
            elif remaining_tokens > 20:  # Only add if we have reasonable space
                section_tokens_list = self.embedding_tokenizer.encode(section_context)
                truncated_section_tokens = section_tokens_list[: remaining_tokens - 10]
                truncated_section = self.embedding_tokenizer.decode(
                    truncated_section_tokens
                )
                context_pieces.append(f"Section: {truncated_section}")
                remaining_tokens = 10

        # 3. Document summary (lowest priority)
        if document_summary and remaining_tokens > 20:
            summary_tokens = self._count_tokens(document_summary)
            if summary_tokens <= remaining_tokens:
                context_pieces.append(f"Document: {document_summary}")
                remaining_tokens -= summary_tokens
            elif remaining_tokens > 30:  # Only add if we have reasonable space
                summary_tokens_list = self.embedding_tokenizer.encode(document_summary)
                truncated_summary_tokens = summary_tokens_list[:remaining_tokens]
                truncated_summary = self.embedding_tokenizer.decode(
                    truncated_summary_tokens
                )
                context_pieces.append(f"Document: {truncated_summary}")

        # Combine everything
        if context_pieces:
            final_context = " ".join(context_pieces)
            contextualized_content = f"{final_context} {chunk_content}"
        else:
            contextualized_content = chunk_content

        # Final safety check and logging
        final_tokens = self._count_tokens(contextualized_content)
        if final_tokens > self.max_embedding_tokens - self.token_safety_margin:
            self.logger.warning(
                f"Final content still too large ({final_tokens} tokens), using chunk only"
            )
            return chunk_content

        # Log token usage for monitoring
        self.logger.debug(
            f"✅ Token-safe context created: {final_tokens}/{self.max_embedding_tokens} tokens ({final_tokens / self.max_embedding_tokens * 100:.1f}%)"
        )

        return contextualized_content

    def generate_context(self, document_content: str, chunk_content: str) -> str:
        """Generate contextual information for a chunk (synchronous wrapper)."""
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # If we're in an async context, we need to handle this differently
            # For now, we'll create a new task
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self._generate_context_async(document_content, chunk_content),
                )
                return future.result()
        except RuntimeError:
            # No event loop running, we can use asyncio.run
            return asyncio.run(
                self._generate_context_async(document_content, chunk_content)
            )

    async def contextualize_chunks_async(
        self,
        document_content: str,
        chunks: List[str],
        source_document: str,
        metadata: Optional[Dict] = None,
    ) -> List[ContextualChunk]:
        """
        Contextualize multiple chunks from a document asynchronously.

        Args:
            document_content: Full content of the source document
            chunks: List of chunk contents to contextualize
            source_document: Source document identifier
            metadata: Additional metadata to attach to chunks

        Returns:
            List of ContextualChunk objects with contextualized content
        """
        if metadata is None:
            metadata = {}

        self.logger.info(
            f"🔄 Contextualizing {len(chunks)} chunks from {source_document}"
        )

        # Create tasks for all chunks
        tasks = []
        for i, chunk in enumerate(chunks):
            task = self._generate_context_async(document_content, chunk)
            tasks.append(task)

        # Execute all tasks concurrently
        contexts = await asyncio.gather(*tasks, return_exceptions=True)

        # Build ContextualChunk objects
        contextual_chunks = []
        for i, chunk in enumerate(chunks):
            context = contexts[i]
            if isinstance(context, Exception):
                self.logger.error(f"Error contextualizing chunk {i}: {context}")
                context = f"This chunk is from the document: {chunk[:50]}..."

            chunk_id = hashlib.md5(
                f"{source_document}_{i}_{chunk[:100]}".encode()
            ).hexdigest()

            # Create token-safe contextualized content
            contextualized_content = self._create_token_safe_context(
                chunk_content=chunk, context=context
            )

            contextual_chunk = ContextualChunk(
                original_content=chunk,
                contextualized_content=contextualized_content,
                context=context,
                chunk_id=chunk_id,
                source_document=source_document,
                metadata={
                    **metadata,
                    "chunk_index": i,
                    "contextualized": True,
                    "context_length": len(context),
                    "original_length": len(chunk),
                    "contextualized_length": len(contextualized_content),
                },
            )

            contextual_chunks.append(contextual_chunk)

        self.logger.info(
            f"✅ Successfully contextualized {len(contextual_chunks)} chunks"
        )
        return contextual_chunks

    def contextualize_chunks(
        self,
        document_content: str,
        chunks: List[str],
        source_document: str,
        metadata: Optional[Dict] = None,
    ) -> List[ContextualChunk]:
        """
        Contextualize multiple chunks from a document (synchronous wrapper).

        Args:
            document_content: Full content of the source document
            chunks: List of chunk contents to contextualize
            source_document: Source document identifier
            metadata: Additional metadata to attach to chunks

        Returns:
            List of ContextualChunk objects with contextualized content
        """
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # If we're in an async context, we need to handle this differently
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.contextualize_chunks_async(
                        document_content, chunks, source_document, metadata
                    ),
                )
                return future.result()
        except RuntimeError:
            # No event loop running, we can use asyncio.run
            return asyncio.run(
                self.contextualize_chunks_async(
                    document_content, chunks, source_document, metadata
                )
            )

    def contextualize_langchain_documents(
        self, documents: List[Document]
    ) -> List[Document]:
        """
        Contextualize a list of LangChain documents.

        Args:
            documents: List of LangChain Document objects

        Returns:
            List of contextualized LangChain Document objects
        """
        if not documents:
            return []

        # Group documents by source
        docs_by_source = {}
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            if source not in docs_by_source:
                docs_by_source[source] = []
            docs_by_source[source].append(doc)

        contextualized_docs = []

        for source, source_docs in docs_by_source.items():
            self.logger.info(
                f"🔄 Processing {len(source_docs)} chunks from source: {source}"
            )

            # Reconstruct document content from chunks
            document_content = "\n\n".join([doc.page_content for doc in source_docs])

            # Extract chunks
            chunks = [doc.page_content for doc in source_docs]

            # Contextualize chunks
            contextual_chunks = self.contextualize_chunks(
                document_content=document_content,
                chunks=chunks,
                source_document=source,
                metadata={"langchain_source": True},
            )

            # Convert back to LangChain documents
            for i, (original_doc, contextual_chunk) in enumerate(
                zip(source_docs, contextual_chunks)
            ):
                new_metadata = {
                    **original_doc.metadata,
                    **contextual_chunk.metadata,
                    "contextual_retrieval": True,
                    "original_content": contextual_chunk.original_content,
                    "context": contextual_chunk.context,
                }

                contextualized_doc = Document(
                    page_content=contextual_chunk.contextualized_content,
                    metadata=new_metadata,
                )

                contextualized_docs.append(contextualized_doc)

        self.logger.info(
            f"✅ Contextualized {len(contextualized_docs)} documents total"
        )
        return contextualized_docs

    async def contextualize_chunks_enhanced_async(
        self,
        document_content: str,
        chunks: List[str],
        source_document: str,
        metadata: Optional[Dict] = None,
    ) -> List[ContextualChunk]:
        """
        Enhanced contextualize multiple chunks from a large document asynchronously.
        Uses document structure analysis for better contextualization.
        """
        if metadata is None:
            metadata = {}

        self.logger.info(
            f"🔄 Enhanced contextualizing {len(chunks)} chunks from {source_document}"
        )

        # Check if document is large enough for enhanced processing
        is_large_document = len(document_content) >= self.config.long_document_threshold

        if is_large_document and self.config.enable_document_structure_analysis:
            # Analyze document structure
            document_structure = self._analyze_document_structure(
                document_content, source_document
            )
            self.logger.info(
                f"📊 Analyzed document structure: {len(document_structure.sections)} sections found"
            )
        else:
            # Fall back to simple contextualization
            return await self.contextualize_chunks_async(
                document_content, chunks, source_document, metadata
            )

        # Calculate chunk positions and find sections
        chunk_positions = []
        current_position = 0
        for chunk in chunks:
            chunk_pos = document_content.find(chunk, current_position)
            if chunk_pos == -1:
                chunk_pos = current_position
            chunk_positions.append(chunk_pos)
            current_position = chunk_pos + len(chunk)

        # Create enhanced contextualization tasks
        tasks = []
        for i, chunk in enumerate(chunks):
            chunk_position = chunk_positions[i]
            document_position = self._calculate_document_position(
                chunk_position, len(document_content)
            )
            section_info = self._find_section_for_chunk(
                chunk, document_structure, chunk_position
            )

            # Get adjacent chunk context
            prev_chunk_context = chunks[i - 1][:100] + "..." if i > 0 else None
            next_chunk_context = (
                chunks[i + 1][:100] + "..." if i < len(chunks) - 1 else None
            )

            task = self._generate_enhanced_context_async(
                document_structure=document_structure,
                chunk_content=chunk,
                section_info=section_info,
                document_position=document_position,
                prev_chunk_context=prev_chunk_context,
                next_chunk_context=next_chunk_context,
            )
            tasks.append(task)

        # Execute all tasks concurrently
        contexts = await asyncio.gather(*tasks, return_exceptions=True)

        # Build enhanced ContextualChunk objects
        contextual_chunks = []
        for i, chunk in enumerate(chunks):
            context = contexts[i]
            if isinstance(context, Exception):
                self.logger.error(f"Error contextualizing chunk {i}: {context}")
                context = f"This chunk is from the {document_structure.document_type} '{document_structure.title}'"

            chunk_id = hashlib.md5(
                f"{source_document}_{i}_{chunk[:100]}".encode()
            ).hexdigest()

            # Create token-safe enhanced context
            section_context_text = section_info.title if section_info else ""
            contextualized_content = self._create_token_safe_context(
                chunk_content=chunk,
                context=context,
                document_summary=document_structure.summary,
                section_context=section_context_text,
            )

            chunk_position = chunk_positions[i]
            document_position = self._calculate_document_position(
                chunk_position, len(document_content)
            )
            section_info = self._find_section_for_chunk(
                chunk, document_structure, chunk_position
            )

            contextual_chunk = ContextualChunk(
                original_content=chunk,
                contextualized_content=contextualized_content,
                context=context,
                chunk_id=chunk_id,
                source_document=source_document,
                metadata={
                    **metadata,
                    "chunk_index": i,
                    "contextualized": True,
                    "enhanced_contextualization": True,
                    "context_length": len(context),
                    "original_length": len(chunk),
                    "contextualized_length": len(contextualized_content),
                    "document_type": document_structure.document_type,
                    "document_title": document_structure.title,
                },
                # Enhanced fields
                document_summary=document_structure.summary,
                section_context=section_info.title if section_info else None,
                document_position=document_position,
                section_info=section_info,
                previous_chunk_context=chunks[i - 1][:100] + "..." if i > 0 else None,
                next_chunk_context=(
                    chunks[i + 1][:100] + "..." if i < len(chunks) - 1 else None
                ),
            )

            contextual_chunks.append(contextual_chunk)

        self.logger.info(
            f"✅ Successfully enhanced contextualized {len(contextual_chunks)} chunks"
        )
        return contextual_chunks

    def contextualize_chunks_enhanced(
        self,
        document_content: str,
        chunks: List[str],
        source_document: str,
        metadata: Optional[Dict] = None,
    ) -> List[ContextualChunk]:
        """
        Enhanced contextualize multiple chunks from a large document (synchronous wrapper).
        """
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.contextualize_chunks_enhanced_async(
                        document_content, chunks, source_document, metadata
                    ),
                )
                return future.result()
        except RuntimeError:
            return asyncio.run(
                self.contextualize_chunks_enhanced_async(
                    document_content, chunks, source_document, metadata
                )
            )

    def contextualize_langchain_documents_enhanced(
        self, documents: List[Document]
    ) -> List[Document]:
        """
        Enhanced contextualize a list of LangChain documents with document structure analysis.
        """
        if not documents:
            return []

        # Group documents by source
        docs_by_source = {}
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            if source not in docs_by_source:
                docs_by_source[source] = []
            docs_by_source[source].append(doc)

        contextualized_docs = []

        for source, source_docs in docs_by_source.items():
            self.logger.info(
                f"🔄 Enhanced processing {len(source_docs)} chunks from source: {source}"
            )

            # Reconstruct document content from chunks
            document_content = "\n\n".join([doc.page_content for doc in source_docs])

            # Check if we should use enhanced processing
            use_enhanced = (
                len(document_content) >= self.config.long_document_threshold
                and self.config.enable_document_structure_analysis
            )

            if use_enhanced:
                # Extract chunks
                chunks = [doc.page_content for doc in source_docs]

                # Enhanced contextualize chunks
                contextual_chunks = self.contextualize_chunks_enhanced(
                    document_content=document_content,
                    chunks=chunks,
                    source_document=source,
                    metadata={"langchain_source": True, "enhanced_processing": True},
                )

                # Convert back to LangChain documents with enhanced metadata
                for i, (original_doc, contextual_chunk) in enumerate(
                    zip(source_docs, contextual_chunks)
                ):
                    new_metadata = {
                        **original_doc.metadata,
                        **contextual_chunk.metadata,
                        "contextual_retrieval": True,
                        "enhanced_contextualization": True,
                        "original_content": contextual_chunk.original_content,
                        "context": contextual_chunk.context,
                        "document_summary": contextual_chunk.document_summary,
                        "section_context": contextual_chunk.section_context,
                        "document_position": contextual_chunk.document_position,
                    }

                    contextualized_doc = Document(
                        page_content=contextual_chunk.contextualized_content,
                        metadata=new_metadata,
                    )

                    contextualized_docs.append(contextualized_doc)
            else:
                # Fall back to regular contextualization
                chunks = [doc.page_content for doc in source_docs]
                contextual_chunks = self.contextualize_chunks(
                    document_content=document_content,
                    chunks=chunks,
                    source_document=source,
                    metadata={"langchain_source": True},
                )

                for i, (original_doc, contextual_chunk) in enumerate(
                    zip(source_docs, contextual_chunks)
                ):
                    new_metadata = {
                        **original_doc.metadata,
                        **contextual_chunk.metadata,
                        "contextual_retrieval": True,
                        "original_content": contextual_chunk.original_content,
                        "context": contextual_chunk.context,
                    }

                    contextualized_doc = Document(
                        page_content=contextual_chunk.contextualized_content,
                        metadata=new_metadata,
                    )

                    contextualized_docs.append(contextualized_doc)

        self.logger.info(
            f"✅ Enhanced contextualized {len(contextualized_docs)} documents total"
        )
        return contextualized_docs

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        if not self.cache:
            return {"cache_enabled": False}

        return {
            "cache_enabled": True,
            "cache_size": len(self.cache.cache),
            "ttl_seconds": self.cache.ttl_seconds,
        }
