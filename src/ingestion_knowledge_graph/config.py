"""Configuration for Knowledge Graph Ingestion."""

import os
import logging
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class KnowledgeGraphConfig:
    """Configuration class for knowledge graph ingestion."""

    # Neo4j Configuration
    neo4j_uri: str = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    neo4j_username: str = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")

    # LLM Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    knowledge_graph_model: str = os.getenv("OPENAPI_KNOWLEDGE_GRAPH_MODEL", "gpt-4o")

    # Processing Configuration
    max_chunk_size: int = int(os.getenv("KG_MAX_CHUNK_SIZE", "4000"))
    batch_size: int = int(os.getenv("KG_BATCH_SIZE", "5"))
    max_concurrent_requests: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "3"))

    # Entity Extraction Configuration
    max_entities_per_chunk: int = int(os.getenv("MAX_ENTITIES_PER_CHUNK", "20"))
    max_relationships_per_chunk: int = int(
        os.getenv("MAX_RELATIONSHIPS_PER_CHUNK", "15")
    )
    entity_types: list = None

    # Logging Configuration
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    def __post_init__(self):
        """Post-initialization validation and setup."""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        if not self.neo4j_password or self.neo4j_password == "password":
            logging.warning(
                "Using default Neo4j password. Please change for production."
            )

        # Default entity types if not provided
        if self.entity_types is None:
            self.entity_types = [
                "PERSON",
                "ORGANIZATION",
                "LOCATION",
                "EVENT",
                "CONCEPT",
                "PRODUCT",
                "SERVICE",
                "TECHNOLOGY",
                "DOCUMENT",
                "DATE",
                "METRIC",
                "CURRENCY",
                "PERCENTAGE",
            ]

        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def validate(self) -> bool:
        """Validate the configuration."""
        required_fields = [
            "neo4j_uri",
            "neo4j_username",
            "neo4j_password",
            "openai_api_key",
            "knowledge_graph_model",
        ]

        for field in required_fields:
            value = getattr(self, field)
            if not value or (isinstance(value, str) and not value.strip()):
                logging.error(f"Required field '{field}' is missing or empty")
                return False

        return True
