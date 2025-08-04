"""
Document Ingestion Pipeline

This package provides a comprehensive document ingestion system that:
- Processes documents with Mistral OCR
- Generates descriptions for images and tables using Anthropic Claude
- Creates intelligent chunks with overlap
- Stores vectors in Qdrant with rich metadata
"""

from .config import IngestionConfig
from .document_processor import DocumentProcessor
from .chunker import ContentChunker
from .vectorizer import VectorStore
from .pipeline import IngestionPipeline

__all__ = [
    "IngestionConfig",
    "DocumentProcessor",
    "ContentChunker",
    "VectorStore",
    "IngestionPipeline",
]
