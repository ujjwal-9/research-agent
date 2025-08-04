"""Configuration management for the ingestion pipeline."""

import os
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class IngestionConfig:
    """Configuration management for document ingestion pipeline."""

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
                logging.FileHandler(f"logs/ingestion_{self._get_timestamp()}.log"),
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
        required_vars = ["MISTRAL_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"]

        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if missing_vars:
            error_msg = (
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.info("✅ All required environment variables are set")

    @property
    def mistral_api_key(self) -> str:
        """Get Mistral API key."""
        return os.getenv("MISTRAL_API_KEY")

    @property
    def anthropic_api_key(self) -> str:
        """Get Anthropic API key."""
        return os.getenv("ANTHROPIC_API_KEY")

    @property
    def openai_api_key(self) -> str:
        """Get OpenAI API key."""
        return os.getenv("OPENAI_API_KEY")

    @property
    def mistral_parsing_model(self) -> str:
        """Get Mistral parsing model."""
        return os.getenv("MISTRAL_PARSING_MODEL", "mistral-ocr-latest")

    @property
    def anthropic_description_model(self) -> str:
        """Get Anthropic description model."""
        return os.getenv("ANTHROPIC_DESCRIPTION_MODEL", "claude-sonnet-4-20250514")

    @property
    def openai_embedding_model(self) -> str:
        """Get OpenAI embedding model."""
        return os.getenv("OPENAI_INGESTION_EMBEDDING_MODEL", "text-embedding-3-large")

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
        """Get Qdrant collection name."""
        return os.getenv("QDRANT_COLLECTION_NAME", "redesign")

    @property
    def chunk_size(self) -> int:
        """Get chunk size."""
        return int(os.getenv("CHUNK_SIZE", "1000"))

    @property
    def chunk_overlap(self) -> int:
        """Get chunk overlap."""
        return int(os.getenv("CHUNK_OVERLAP", "150"))

    @property
    def max_image_size_mb(self) -> int:
        """Get maximum image size in MB."""
        return int(os.getenv("MAX_IMAGE_SIZE_MB", "20"))

    @property
    def batch_size(self) -> int:
        """Get batch size for processing."""
        return int(os.getenv("BATCH_SIZE", "10"))

    @property
    def memory_limit_mb(self) -> int:
        """Get memory limit in MB."""
        return int(os.getenv("MEMORY_LIMIT_MB", "8000"))

    @property
    def parallel_description_calls(self) -> int:
        """Get maximum parallel description calls."""
        return int(os.getenv("PARALLEL_DESCRIPTION_CALLS", "3"))

    @property
    def api_retry_attempts(self) -> int:
        """Get number of API retry attempts."""
        return int(os.getenv("API_RETRY_ATTEMPTS", "3"))

    @property
    def api_retry_delay(self) -> float:
        """Get API retry delay in seconds."""
        return float(os.getenv("API_RETRY_DELAY", "2.0"))

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "mistral_parsing_model": self.mistral_parsing_model,
            "anthropic_description_model": self.anthropic_description_model,
            "openai_embedding_model": self.openai_embedding_model,
            "qdrant_url": self.qdrant_url,
            "qdrant_collection_name": self.qdrant_collection_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "max_image_size_mb": self.max_image_size_mb,
            "batch_size": self.batch_size,
            "memory_limit_mb": self.memory_limit_mb,
            "parallel_description_calls": self.parallel_description_calls,
            "api_retry_attempts": self.api_retry_attempts,
            "api_retry_delay": self.api_retry_delay,
        }
