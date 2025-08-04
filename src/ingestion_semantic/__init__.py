"""
Semantic Document Ingestion Pipeline

This package provides a comprehensive semantic document ingestion system that:
- Uses LangChain for document processing
- Implements contextual retrieval for better chunk semantics
- Uses OpenAI GPT-4.1 for contextual chunk generation with prompt caching
- Uses OpenAI text-embedding-3-large for vector embeddings
- Processes images/charts/tables with vision models using contextual retrieval
- Stores enhanced vectors in Qdrant with rich semantic metadata
"""

from .config import SemanticIngestionConfig
from .contextual_retrieval import ContextualRetrieval
from .semantic_chunker import SemanticChunker
from .vision_processor import VisionProcessor
from .document_processor import SemanticDocumentProcessor
from .embeddings import SemanticEmbeddings
from .pipeline import SemanticIngestionPipeline

__all__ = [
    "SemanticIngestionConfig",
    "ContextualRetrieval",
    "SemanticChunker",
    "VisionProcessor",
    "SemanticDocumentProcessor",
    "SemanticEmbeddings",
    "SemanticIngestionPipeline",
]
