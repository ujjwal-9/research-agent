"""Configuration management for the research system."""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4-turbo-preview", env="OPENAI_MODEL")

    # Vector Database Configuration
    qdrant_host: str = Field("localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(6333, env="QDRANT_PORT")
    qdrant_collection_name: str = Field("documents", env="QDRANT_COLLECTION_NAME")
    qdrant_storage_path: str = Field("./qdrant_storage", env="QDRANT_STORAGE_PATH")
    embedding_model: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL")

    # Knowledge Graph Configuration
    neo4j_uri: str = Field("neo4j://127.0.0.1:7687", env="NEO4J_URI")
    neo4j_username: str = Field("neo4j", env="NEO4J_USERNAME")
    neo4j_password: str = Field("password", env="NEO4J_PASSWORD")
    neo4j_database: str = Field("neo4j", env="NEO4J_DATABASE")

    # Entity Extraction Configuration
    spacy_model: str = Field("en_core_web_sm", env="SPACY_MODEL")
    min_entity_confidence: float = Field(0.7, env="MIN_ENTITY_CONFIDENCE")
    max_entities_per_chunk: int = Field(20, env="MAX_ENTITIES_PER_CHUNK")

    # LLM Entity Extraction Configuration
    use_llm_extraction: bool = Field(True, env="USE_LLM_EXTRACTION")
    llm_extraction_model: str = Field("gpt-4-turbo-preview", env="LLM_EXTRACTION_MODEL")
    llm_max_text_length: int = Field(4000, env="LLM_MAX_TEXT_LENGTH")
    llm_extraction_temperature: float = Field(0.1, env="LLM_EXTRACTION_TEMPERATURE")

    # Document Processing Configuration
    document_chunk_size: int = Field(1000, env="DOCUMENT_CHUNK_SIZE")
    document_chunk_overlap: int = Field(200, env="DOCUMENT_CHUNK_OVERLAP")

    # System Configuration
    log_level: str = Field("INFO", env="LOG_LEVEL")
    max_concurrent_tasks: int = Field(5, env="MAX_CONCURRENT_TASKS")

    # Web Search Configuration
    duckduckgo_max_results: int = Field(10, env="DUCKDUCKGO_MAX_RESULTS")

    # API Configuration
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")

    # MCP Configuration
    mcp_server_name: str = Field("research-system", env="MCP_SERVER_NAME")
    mcp_server_version: str = Field("1.0.0", env="MCP_SERVER_VERSION")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
