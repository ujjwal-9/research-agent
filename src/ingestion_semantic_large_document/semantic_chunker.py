"""Enhanced semantic chunker for large documents with advanced contextual awareness."""

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
from .contextual_retrieval import EnhancedContextualRetrieval


class EnhancedSemanticChunker:
    """
    Enhanced semantic document chunker for large documents with advanced contextual awareness.

    This chunker uses LangChain's SemanticChunker for embedding-based semantic splitting
    and then enhances chunks with advanced contextual information using the EnhancedContextualRetrieval system.
    Includes document structure analysis and adaptive chunk sizing for better large document processing.
    """

    def __init__(self, config, contextual_retrieval: EnhancedContextualRetrieval):
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

    def _get_adaptive_chunk_size(
        self, content: str, metadata: Dict[str, Any]
    ) -> tuple[int, int]:
        """Determine adaptive chunk size based on document characteristics."""
        content_length = len(content)

        # Base chunk size from config
        base_chunk_size = self.config.chunk_size
        base_overlap = self.config.chunk_overlap

        if not self.config.adaptive_chunk_sizing:
            return base_chunk_size, base_overlap

        # Adaptive sizing based on document length
        if content_length < 10000:  # Small documents
            return base_chunk_size, base_overlap
        elif content_length < 50000:  # Medium documents
            return int(base_chunk_size * 1.2), int(base_overlap * 1.1)
        elif content_length < 200000:  # Large documents
            return int(base_chunk_size * 1.5), int(base_overlap * 1.2)
        else:  # Very large documents
            return min(
                self.config.max_document_chunk_size, int(base_chunk_size * 2)
            ), int(base_overlap * 1.3)

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
        """Enhanced fallback chunking using traditional methods with adaptive sizing."""
        doc_type = self._detect_document_type(content, metadata)

        # Get adaptive chunk sizes
        adaptive_chunk_size, adaptive_overlap = self._get_adaptive_chunk_size(
            content, metadata
        )

        # Create adaptive text splitter
        adaptive_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=adaptive_chunk_size,
            chunk_overlap=adaptive_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        if doc_type == "markdown":
            try:
                header_chunks = self.markdown_splitter.split_text(content)
                documents = []

                for chunk in header_chunks:
                    if len(chunk.page_content) > adaptive_chunk_size:
                        sub_chunks = adaptive_text_splitter.split_documents([chunk])
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
                    if len(chunk.page_content) > adaptive_chunk_size:
                        sub_chunks = adaptive_text_splitter.split_documents([chunk])
                        documents.extend(sub_chunks)
                    else:
                        documents.append(chunk)

                return documents

            except Exception as e:
                self.logger.warning(f"HTML splitting failed: {e}")

        # Final fallback: adaptive character-based splitting
        return adaptive_text_splitter.split_documents(
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

        # Apply enhanced contextual retrieval for large documents if requested
        if apply_contextual_retrieval and chunks:
            self.logger.info(
                f"🧠 Applying enhanced contextual retrieval to {len(chunks)} chunks"
            )

            # Check if document is large enough for enhanced processing
            is_large_document = len(content) >= self.config.long_document_threshold

            if is_large_document and self.config.enable_document_structure_analysis:
                self.logger.info(
                    f"📊 Using enhanced processing for large document ({len(content):,} chars)"
                )
                chunks = self.contextual_retrieval.contextualize_langchain_documents_enhanced(
                    chunks
                )
            else:
                chunks = self.contextual_retrieval.contextualize_langchain_documents(
                    chunks
                )

        # Add enhanced chunking metadata
        is_large_document = len(content) >= self.config.long_document_threshold
        adaptive_chunk_size, adaptive_overlap = self._get_adaptive_chunk_size(
            content, metadata
        )

        for i, chunk in enumerate(chunks):
            chunk.metadata.update(
                {
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_method": "enhanced_semantic_gradient",
                    "contextualized": apply_contextual_retrieval,
                    "enhanced_contextualization": is_large_document
                    and self.config.enable_document_structure_analysis,
                    "semantic_chunker_used": self.semantic_splitter is not None,
                    "adaptive_chunk_size": adaptive_chunk_size,
                    "adaptive_overlap": adaptive_overlap,
                    "is_large_document": is_large_document,
                    "document_length": len(content),
                    "preserve_structure": self.config.preserve_document_structure,
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
        """Get enhanced chunker configuration and statistics."""
        return {
            "primary_method": "enhanced_semantic_gradient",
            "semantic_chunker_available": self.semantic_splitter is not None,
            "embedding_model": self.config.embedding_model,
            "breakpoint_threshold_type": "gradient",
            "breakpoint_threshold_amount": 95.0,
            "base_chunk_size": self.config.chunk_size,
            "base_chunk_overlap": self.config.chunk_overlap,
            "max_document_chunk_size": self.config.max_document_chunk_size,
            "adaptive_chunk_sizing": self.config.adaptive_chunk_sizing,
            "large_document_threshold": self.config.long_document_threshold,
            "supported_types": ["text", "markdown", "html"],
            "enhanced_features": {
                "contextual_retrieval_enabled": True,
                "document_structure_analysis": self.config.enable_document_structure_analysis,
                "section_based_contextualization": self.config.enable_section_based_contextualization,
                "adaptive_chunk_sizing": self.config.adaptive_chunk_sizing,
                "preserve_document_structure": self.config.preserve_document_structure,
            },
            "splitters": {
                "primary": "Enhanced SemanticChunker (gradient method with document structure analysis)",
                "fallback_text": "Adaptive RecursiveCharacterTextSplitter",
                "fallback_markdown": "MarkdownHeaderTextSplitter + Adaptive RecursiveCharacterTextSplitter",
                "fallback_html": "HTMLHeaderTextSplitter + Adaptive RecursiveCharacterTextSplitter",
            },
        }
