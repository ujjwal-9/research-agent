"""Tests for document processing functionality."""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch

from src.ingestion.document_processor import DocumentProcessor, ProcessedDocument


@pytest.fixture
def processor():
    """Create a document processor instance."""
    return DocumentProcessor(chunk_size=100, chunk_overlap=20)


@pytest.mark.asyncio
async def test_text_file_processing(processor, tmp_path):
    """Test processing of text files."""
    # Create a test text file
    test_file = tmp_path / "test.txt"
    test_content = "This is a test document with some content for processing."
    test_file.write_text(test_content)
    
    # Process the document
    result = await processor.process_document(test_file)
    
    # Assertions
    assert result is not None
    assert isinstance(result, ProcessedDocument)
    assert result.file_type == "text"
    assert result.content == test_content
    assert result.title == "test"
    assert len(result.chunks) > 0


@pytest.mark.asyncio
async def test_unsupported_file_type(processor, tmp_path):
    """Test handling of unsupported file types."""
    # Create a file with unsupported extension
    test_file = tmp_path / "test.xyz"
    test_file.write_text("content")
    
    # Process the document
    result = await processor.process_document(test_file)
    
    # Should return None for unsupported types
    assert result is None


def test_text_chunking(processor):
    """Test text chunking functionality."""
    text = "A" * 150  # Text longer than chunk_size
    
    chunks = processor._chunk_text(text)
    
    assert len(chunks) > 1
    assert all(len(chunk) <= processor.chunk_size for chunk in chunks)


def test_empty_text_chunking(processor):
    """Test chunking of empty text."""
    chunks = processor._chunk_text("")
    assert chunks == []
    
    chunks = processor._chunk_text("   ")
    assert chunks == []


@pytest.mark.asyncio
async def test_processing_error_handling(processor, tmp_path):
    """Test error handling during document processing."""
    # Create a file that will cause an error
    test_file = tmp_path / "nonexistent.txt"
    
    # Process the document
    result = await processor.process_document(test_file)
    
    # Should handle error gracefully
    assert result is None