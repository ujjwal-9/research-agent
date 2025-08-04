"""Configuration management for the semantic ingestion pipeline for large documents."""

import os
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class SemanticIngestionLargeDocumentConfig:
    """Configuration management for semantic document ingestion pipeline."""

    def __init__(self):
        self.logger = self._setup_logging()
        self._validate_environment()

    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()

        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(
                    f"logs/semantic_ingestion_large_document_{self._get_timestamp()}.log"
                ),
                logging.StreamHandler(),
            ],
        )

        return logging.getLogger(__name__)

    def _get_timestamp(self) -> str:
        """Get current timestamp for log files."""
        from datetime import datetime

        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _validate_environment(self):
        """Validate required environment variables."""
        required_vars = [
            "OPENAI_API_KEY",
            "OPENAI_CONTEXTUAL_RETRIEVAL_MODEL",
            "OPENAI_INGESTION_EMBEDDING_MODEL",
        ]

        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if missing_vars:
            error_msg = (
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.info("✅ All required environment variables are set")

    @property
    def openai_api_key(self) -> str:
        """Get OpenAI API key."""
        return os.getenv("OPENAI_API_KEY")

    @property
    def contextual_retrieval_model(self) -> str:
        """Get OpenAI model for contextual retrieval."""
        return os.getenv("OPENAI_CONTEXTUAL_RETRIEVAL_MODEL", "gpt-4o")

    @property
    def embedding_model(self) -> str:
        """Get OpenAI embedding model."""
        return os.getenv("OPENAI_INGESTION_EMBEDDING_MODEL", "text-embedding-3-large")

    @property
    def vision_model(self) -> str:
        """Get OpenAI vision model for image processing."""
        return os.getenv("OPENAI_VISION_MODEL", "gpt-4o")

    @property
    def qdrant_url(self) -> str:
        """Get Qdrant URL."""
        return os.getenv("QDRANT_URL", "http://localhost:6333")

    @property
    def qdrant_api_key(self) -> Optional[str]:
        """Get Qdrant API key."""
        return os.getenv("QDRANT_API_KEY")

    @property
    def qdrant_collection_name(self) -> str:
        """Get Qdrant collection name for semantic ingestion of large documents."""
        return os.getenv(
            "QDRANT_SEMANTIC_LARGE_DOCUMENT_COLLECTION_NAME",
            "semantic_large_document_redesign",
        )

    @property
    def chunk_size(self) -> int:
        """Get base chunk size for semantic chunking."""
        return int(os.getenv("SEMANTIC_CHUNK_SIZE", "800"))

    @property
    def chunk_overlap(self) -> int:
        """Get chunk overlap for semantic chunking."""
        return int(os.getenv("SEMANTIC_CHUNK_OVERLAP", "200"))

    @property
    def contextual_context_size(self) -> int:
        """Get context size for contextual retrieval prompt."""
        return int(os.getenv("CONTEXTUAL_CONTEXT_SIZE", "100"))

    # Large Document Specific Configuration
    @property
    def document_analysis_model(self) -> str:
        """Get OpenAI model for document structure analysis."""
        return os.getenv("OPENAI_DOCUMENT_ANALYSIS_MODEL", "gpt-4o")

    @property
    def enable_document_structure_analysis(self) -> bool:
        """Get whether to enable document structure analysis."""
        return os.getenv("ENABLE_DOCUMENT_STRUCTURE_ANALYSIS", "true").lower() == "true"

    @property
    def enable_section_based_contextualization(self) -> bool:
        """Get whether to enable section-based contextualization."""
        return (
            os.getenv("ENABLE_SECTION_BASED_CONTEXTUALIZATION", "true").lower()
            == "true"
        )

    @property
    def enhanced_context_size(self) -> int:
        """Get enhanced context size for large documents (longer contexts)."""
        return int(os.getenv("ENHANCED_CONTEXT_SIZE", "200"))

    @property
    def document_summary_size(self) -> int:
        """Get document summary size for overall context."""
        return int(os.getenv("DOCUMENT_SUMMARY_SIZE", "500"))

    @property
    def section_context_size(self) -> int:
        """Get section context size for section-specific context."""
        return int(os.getenv("SECTION_CONTEXT_SIZE", "150"))

    @property
    def enable_reranking(self) -> bool:
        """Get whether to enable reranking for better retrieval."""
        return os.getenv("ENABLE_RERANKING", "false").lower() == "true"

    @property
    def reranker_model(self) -> str:
        """Get reranker model for hybrid retrieval."""
        return os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    @property
    def enable_hybrid_retrieval(self) -> bool:
        """Get whether to enable hybrid dense-sparse retrieval."""
        return os.getenv("ENABLE_HYBRID_RETRIEVAL", "false").lower() == "true"

    @property
    def long_document_threshold(self) -> int:
        """Get threshold in characters to consider a document as 'long'."""
        return int(os.getenv("LONG_DOCUMENT_THRESHOLD", "50000"))

    @property
    def preserve_document_structure(self) -> bool:
        """Get whether to preserve and use document structure for context."""
        return os.getenv("PRESERVE_DOCUMENT_STRUCTURE", "true").lower() == "true"

    @property
    def max_document_chunk_size(self) -> int:
        """Get maximum chunk size for very long documents (in characters)."""
        return int(os.getenv("MAX_DOCUMENT_CHUNK_SIZE", "2000"))

    @property
    def max_tokens_per_chunk(self) -> int:
        """Get maximum tokens per chunk for embedding (safety limit)."""
        return int(os.getenv("MAX_TOKENS_PER_CHUNK", "6000"))  # Well below 8192 limit

    @property
    def adaptive_chunk_sizing(self) -> bool:
        """Get whether to use adaptive chunk sizing based on document structure."""
        return os.getenv("ADAPTIVE_CHUNK_SIZING", "true").lower() == "true"

    @property
    def max_image_size_mb(self) -> int:
        """Get maximum image size in MB."""
        return int(os.getenv("MAX_IMAGE_SIZE_MB", "20"))

    @property
    def batch_size(self) -> int:
        """Get batch size for processing."""
        return int(os.getenv("SEMANTIC_BATCH_SIZE", "10"))

    @property
    def max_concurrent_requests(self) -> int:
        """Get maximum concurrent API requests."""
        return int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))

    @property
    def api_retry_attempts(self) -> int:
        """Get number of API retry attempts."""
        return int(os.getenv("API_RETRY_ATTEMPTS", "3"))

    @property
    def api_retry_delay(self) -> float:
        """Get API retry delay in seconds."""
        return float(os.getenv("API_RETRY_DELAY", "2.0"))

    @property
    def enable_prompt_caching(self) -> bool:
        """Get whether to enable prompt caching."""
        return os.getenv("ENABLE_PROMPT_CACHING", "true").lower() == "true"

    @property
    def cache_ttl_seconds(self) -> int:
        """Get cache TTL in seconds."""
        return int(os.getenv("CACHE_TTL_SECONDS", "3600"))

    # Parallelization Configuration
    @property
    def max_parallel_documents(self) -> int:
        """Get maximum number of documents to process in parallel."""
        return int(os.getenv("MAX_PARALLEL_DOCUMENTS", "4"))

    @property
    def max_parallel_chunks(self) -> int:
        """Get maximum number of chunks to process in parallel within a document."""
        return int(os.getenv("MAX_PARALLEL_CHUNKS", "8"))

    @property
    def embedding_batch_size(self) -> int:
        """Get batch size for embedding generation."""
        return int(os.getenv("EMBEDDING_BATCH_SIZE", "20"))

    @property
    def enable_parallel_processing(self) -> bool:
        """Get whether to enable parallel processing."""
        return os.getenv("ENABLE_PARALLEL_PROCESSING", "true").lower() == "true"

    @property
    def parallel_chunk_processing(self) -> bool:
        """Get whether to enable parallel chunk processing within documents."""
        return os.getenv("PARALLEL_CHUNK_PROCESSING", "true").lower() == "true"

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "contextual_retrieval_model": self.contextual_retrieval_model,
            "embedding_model": self.embedding_model,
            "vision_model": self.vision_model,
            "qdrant_url": self.qdrant_url,
            "qdrant_collection_name": self.qdrant_collection_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "contextual_context_size": self.contextual_context_size,
            "max_image_size_mb": self.max_image_size_mb,
            "batch_size": self.batch_size,
            "max_concurrent_requests": self.max_concurrent_requests,
            "api_retry_attempts": self.api_retry_attempts,
            "api_retry_delay": self.api_retry_delay,
            "enable_prompt_caching": self.enable_prompt_caching,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "max_parallel_documents": self.max_parallel_documents,
            "max_parallel_chunks": self.max_parallel_chunks,
            "embedding_batch_size": self.embedding_batch_size,
            "enable_parallel_processing": self.enable_parallel_processing,
            "parallel_chunk_processing": self.parallel_chunk_processing,
            # Large Document Specific Configuration
            "document_analysis_model": self.document_analysis_model,
            "enable_document_structure_analysis": self.enable_document_structure_analysis,
            "enable_section_based_contextualization": self.enable_section_based_contextualization,
            "enhanced_context_size": self.enhanced_context_size,
            "document_summary_size": self.document_summary_size,
            "section_context_size": self.section_context_size,
            "enable_reranking": self.enable_reranking,
            "reranker_model": self.reranker_model,
            "enable_hybrid_retrieval": self.enable_hybrid_retrieval,
            "long_document_threshold": self.long_document_threshold,
            "preserve_document_structure": self.preserve_document_structure,
            "max_document_chunk_size": self.max_document_chunk_size,
            "adaptive_chunk_sizing": self.adaptive_chunk_sizing,
        }
