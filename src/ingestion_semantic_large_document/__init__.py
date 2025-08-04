"""
Enhanced Semantic Document Ingestion Pipeline for Large Documents

This package provides a comprehensive enhanced semantic document ingestion system optimized for large documents that:
- Uses LangChain for document processing
- Implements enhanced contextual retrieval with document structure analysis
- Provides section-based contextualization for better chunk semantics
- Uses adaptive chunk sizing based on document characteristics
- Uses OpenAI GPT-4o for contextual chunk generation with prompt caching
- Uses OpenAI text-embedding-3-large for vector embeddings
- Processes images/charts/tables with vision models using contextual retrieval
- Stores enhanced vectors in Qdrant with rich semantic metadata
- Optimized for processing large documents (>50k characters) efficiently
"""

from .config import SemanticIngestionLargeDocumentConfig
from .contextual_retrieval import EnhancedContextualRetrieval
from .semantic_chunker import EnhancedSemanticChunker
from .vision_processor import VisionProcessor
from .document_processor import EnhancedSemanticDocumentProcessor
from .embeddings import SemanticEmbeddings
from .pipeline import EnhancedSemanticIngestionPipeline

__all__ = [
    "SemanticIngestionLargeDocumentConfig",
    "EnhancedContextualRetrieval",
    "EnhancedSemanticChunker",
    "VisionProcessor",
    "EnhancedSemanticDocumentProcessor",
    "SemanticEmbeddings",
    "EnhancedSemanticIngestionPipeline",
]
