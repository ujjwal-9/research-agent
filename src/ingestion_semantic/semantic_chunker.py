"""Semantic chunker that creates contextually aware document chunks using embedding-based semantic similarity."""

import logging
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter,
)
from langchain_experimental.text_splitter import (
    SemanticChunker as LangChainSemanticChunker,
)
from langchain_openai import OpenAIEmbeddings
from .contextual_retrieval import ContextualRetrieval, ContextualChunk


class SemanticChunker:
    """
    Semantic document chunker that creates contextually aware chunks.

    This chunker uses LangChain's SemanticChunker for embedding-based semantic splitting
    and then enhances chunks with contextual information using the ContextualRetrieval system.
    """

    def __init__(self, config, contextual_retrieval: ContextualRetrieval):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.contextual_retrieval = contextual_retrieval

        # Initialize semantic and fallback text splitters
        self._init_text_splitters()

    def _init_text_splitters(self):
        """Initialize semantic and fallback text splitters."""

        # Primary semantic chunker using embeddings with gradient method
        try:
            self.semantic_splitter = LangChainSemanticChunker(
                embeddings=OpenAIEmbeddings(
                    model=self.config.embedding_model,
                    openai_api_key=self.config.openai_api_key,
                ),
                breakpoint_threshold_type="gradient",
                breakpoint_threshold_amount=95.0,  # Default for gradient method
            )
            self.logger.info("✅ Semantic chunker initialized with gradient method")
        except Exception as e:
            self.logger.warning(f"Failed to initialize semantic chunker: {e}")
            self.semantic_splitter = None

        # Fallback recursive character splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        # Markdown header splitter for structured markdown documents
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4"),
            ],
            strip_headers=False,
        )

        # HTML header splitter for HTML documents
        self.html_splitter = HTMLHeaderTextSplitter(
            headers_to_split_on=[
                ("h1", "Header 1"),
                ("h2", "Header 2"),
                ("h3", "Header 3"),
                ("h4", "Header 4"),
            ]
        )

        self.logger.info("✅ All text splitters initialized")

    def _detect_document_type(self, content: str, metadata: Dict[str, Any]) -> str:
        """Detect document type to choose appropriate splitter."""

        # Check file extension from metadata
        source = metadata.get("source", "").lower()

        if source.endswith(".md") or source.endswith(".markdown"):
            return "markdown"
        elif source.endswith(".html") or source.endswith(".htm"):
            return "html"

        # Check content patterns
        if content.strip().startswith("#") or "\n#" in content:
            return "markdown"
        elif "<html" in content.lower() or "<body" in content.lower():
            return "html"

        return "text"

    def _chunk_with_semantic_splitter(
        self, content: str, metadata: Dict[str, Any]
    ) -> List[Document]:
        """Primary chunking using semantic splitter with gradient method."""
        try:
            if self.semantic_splitter is None:
                raise ValueError("Semantic splitter not available")

            # Create a document for semantic splitting
            doc = Document(page_content=content, metadata=metadata)

            # Use semantic chunker with gradient method
            chunks = self.semantic_splitter.split_documents([doc])

            self.logger.debug(f"Semantic splitter created {len(chunks)} chunks")
            return chunks

        except Exception as e:
            self.logger.warning(
                f"Semantic splitting failed: {e}, falling back to character splitting"
            )
            return self._fallback_chunk(content, metadata)

    def _fallback_chunk(self, content: str, metadata: Dict[str, Any]) -> List[Document]:
        """Fallback chunking using traditional methods based on content type."""
        doc_type = self._detect_document_type(content, metadata)

        if doc_type == "markdown":
            try:
                header_chunks = self.markdown_splitter.split_text(content)
                documents = []

                for chunk in header_chunks:
                    if len(chunk.page_content) > self.config.chunk_size:
                        sub_chunks = self.text_splitter.split_documents([chunk])
                        documents.extend(sub_chunks)
                    else:
                        documents.append(chunk)

                return documents

            except Exception as e:
                self.logger.warning(f"Markdown splitting failed: {e}")

        elif doc_type == "html":
            try:
                header_chunks = self.html_splitter.split_text(content)
                documents = []

                for chunk in header_chunks:
                    if len(chunk.page_content) > self.config.chunk_size:
                        sub_chunks = self.text_splitter.split_documents([chunk])
                        documents.extend(sub_chunks)
                    else:
                        documents.append(chunk)

                return documents

            except Exception as e:
                self.logger.warning(f"HTML splitting failed: {e}")

        # Final fallback: character-based splitting
        return self.text_splitter.split_documents(
            [Document(page_content=content, metadata=metadata)]
        )

    def chunk_document(
        self,
        content: str,
        metadata: Dict[str, Any],
        apply_contextual_retrieval: bool = True,
    ) -> List[Document]:
        """
        Chunk a document and optionally apply contextual retrieval.

        Args:
            content: Document content to chunk
            metadata: Document metadata
            apply_contextual_retrieval: Whether to apply contextual retrieval

        Returns:
            List of chunked and optionally contextualized documents
        """
        if not content.strip():
            self.logger.warning("Empty content provided for chunking")
            return []

        source = metadata.get("source", "unknown")
        self.logger.info(f"🔪 Chunking document: {source}")

        # Use semantic chunker as primary method
        self.logger.debug("Using semantic chunker with gradient method")
        chunks = self._chunk_with_semantic_splitter(content, metadata)

        self.logger.info(f"📄 Created {len(chunks)} initial chunks from {source}")

        # Apply contextual retrieval if requested
        if apply_contextual_retrieval and chunks:
            self.logger.info(
                f"🧠 Applying contextual retrieval to {len(chunks)} chunks"
            )
            chunks = self.contextual_retrieval.contextualize_langchain_documents(chunks)

        # Add chunking metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update(
                {
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_method": "semantic_gradient",
                    "contextualized": apply_contextual_retrieval,
                    "semantic_chunker_used": self.semantic_splitter is not None,
                }
            )

        self.logger.info(
            f"✅ Successfully processed {len(chunks)} chunks from {source}"
        )
        return chunks

    def chunk_documents(
        self, documents: List[Document], apply_contextual_retrieval: bool = True
    ) -> List[Document]:
        """
        Chunk multiple documents.

        Args:
            documents: List of documents to chunk
            apply_contextual_retrieval: Whether to apply contextual retrieval

        Returns:
            List of all chunks from all documents
        """
        all_chunks = []

        for doc in documents:
            chunks = self.chunk_document(
                content=doc.page_content,
                metadata=doc.metadata,
                apply_contextual_retrieval=apply_contextual_retrieval,
            )
            all_chunks.extend(chunks)

        self.logger.info(
            f"✅ Chunked {len(documents)} documents into {len(all_chunks)} total chunks"
        )
        return all_chunks

    def chunk_text_with_metadata(
        self,
        text: str,
        source: str,
        additional_metadata: Optional[Dict[str, Any]] = None,
        apply_contextual_retrieval: bool = True,
    ) -> List[Document]:
        """
        Chunk raw text with basic metadata.

        Args:
            text: Raw text to chunk
            source: Source identifier for the text
            additional_metadata: Additional metadata to attach
            apply_contextual_retrieval: Whether to apply contextual retrieval

        Returns:
            List of chunked documents
        """
        metadata = {
            "source": source,
            "content_type": "text",
        }

        if additional_metadata:
            metadata.update(additional_metadata)

        return self.chunk_document(
            content=text,
            metadata=metadata,
            apply_contextual_retrieval=apply_contextual_retrieval,
        )

    def get_chunker_stats(self) -> Dict[str, Any]:
        """Get chunker configuration and statistics."""
        return {
            "primary_method": "semantic_gradient",
            "semantic_chunker_available": self.semantic_splitter is not None,
            "embedding_model": self.config.embedding_model,
            "breakpoint_threshold_type": "gradient",
            "breakpoint_threshold_amount": 95.0,
            "fallback_chunk_size": self.config.chunk_size,
            "fallback_chunk_overlap": self.config.chunk_overlap,
            "supported_types": ["text", "markdown", "html"],
            "contextual_retrieval_enabled": True,
            "splitters": {
                "primary": "SemanticChunker (gradient method)",
                "fallback_text": "RecursiveCharacterTextSplitter",
                "fallback_markdown": "MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter",
                "fallback_html": "HTMLHeaderTextSplitter + RecursiveCharacterTextSplitter",
            },
        }
