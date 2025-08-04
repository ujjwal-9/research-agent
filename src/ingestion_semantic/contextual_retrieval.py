"""Contextual Retrieval implementation based on Anthropic's approach."""

import asyncio
import hashlib
import json
import logging
import time
from typing import Dict, List, Optional, Tuple
from functools import lru_cache
from dataclasses import dataclass

from openai import OpenAI
from langchain_core.documents import Document


@dataclass
class ContextualChunk:
    """Container for a chunk with its contextual information."""

    original_content: str
    contextualized_content: str
    context: str
    chunk_id: str
    source_document: str
    metadata: Dict


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


class ContextualRetrieval:
    """
    Contextual Retrieval implementation following Anthropic's approach.

    This class generates contextual information for document chunks to improve
    retrieval accuracy by providing chunk-specific explanatory context.
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

            # Combine context with original chunk
            contextualized_content = f"{context} {chunk}"

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

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        if not self.cache:
            return {"cache_enabled": False}

        return {
            "cache_enabled": True,
            "cache_size": len(self.cache.cache),
            "ttl_seconds": self.cache.ttl_seconds,
        }
