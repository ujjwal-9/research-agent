#!/usr/bin/env python3
"""Test token management in enhanced contextual retrieval."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingestion_semantic_large_document.config import (
    SemanticIngestionLargeDocumentConfig,
)
from ingestion_semantic_large_document.contextual_retrieval import (
    EnhancedContextualRetrieval,
)


def test_token_counting():
    """Test basic token counting functionality."""
    config = SemanticIngestionLargeDocumentConfig()
    retrieval = EnhancedContextualRetrieval(config)

    # Test token counting
    short_text = "Hello world"
    tokens = retrieval._count_tokens(short_text)
    assert tokens > 0
    assert tokens < 10  # Should be small number for short text

    # Test longer text
    long_text = "This is a much longer text that contains many more words and should result in a higher token count than the previous example."
    long_tokens = retrieval._count_tokens(long_text)
    assert long_tokens > tokens
    print(
        f"✅ Token counting works: '{short_text}' = {tokens} tokens, longer text = {long_tokens} tokens"
    )


def test_token_safe_context_creation():
    """Test that contextualized content respects token limits."""
    config = SemanticIngestionLargeDocumentConfig()
    retrieval = EnhancedContextualRetrieval(config)

    # Create a chunk that's close to the limit
    large_chunk = "Large document content. " * 2000  # ~6000 characters
    context = (
        "This chunk discusses important business metrics and financial data analysis."
    )
    document_summary = "This is a comprehensive financial report covering Q3 2023 business performance, revenue analysis, and strategic planning initiatives."
    section_context = "Financial Analysis Section"

    # Test token-safe creation
    result = retrieval._create_token_safe_context(
        chunk_content=large_chunk,
        context=context,
        document_summary=document_summary,
        section_context=section_context,
    )

    # Verify result doesn't exceed token limit
    final_tokens = retrieval._count_tokens(result)
    max_allowed = retrieval.max_embedding_tokens - retrieval.token_safety_margin

    assert final_tokens <= max_allowed, (
        f"Result has {final_tokens} tokens, max allowed is {max_allowed}"
    )

    # If chunk was truncated, check that result contains beginning of original chunk
    # If not truncated, original chunk should be fully preserved
    chunk_tokens = retrieval._count_tokens(large_chunk)
    if chunk_tokens > max_allowed:
        # For truncated chunks, check that some of the original content is preserved
        original_start = large_chunk[:200]  # First 200 characters
        assert original_start in result or any(
            word in result for word in original_start.split()[:10]
        ), "Truncated chunk should preserve some original content"
    else:
        assert large_chunk in result, (
            "Original chunk content should be preserved when not truncated"
        )

    print(
        f"✅ Token-safe context: {final_tokens}/{max_allowed} tokens ({final_tokens / max_allowed * 100:.1f}%)"
    )


def test_extremely_large_chunk_handling():
    """Test handling of chunks that are too large even without context."""
    config = SemanticIngestionLargeDocumentConfig()
    retrieval = EnhancedContextualRetrieval(config)

    # Create an extremely large chunk
    huge_chunk = "Extremely large content. " * 5000  # ~125,000 characters
    context = "This is contextual information."

    # Should truncate the chunk itself
    result = retrieval._create_token_safe_context(
        chunk_content=huge_chunk, context=context
    )

    final_tokens = retrieval._count_tokens(result)
    max_allowed = retrieval.max_embedding_tokens - retrieval.token_safety_margin

    assert final_tokens <= max_allowed, (
        f"Result has {final_tokens} tokens, max allowed is {max_allowed}"
    )
    assert len(result) < len(huge_chunk), "Huge chunk should be truncated"

    print(
        f"✅ Large chunk handled: {final_tokens}/{max_allowed} tokens, truncated from {len(huge_chunk)} to {len(result)} chars"
    )


def test_priority_system():
    """Test that context prioritization works correctly."""
    config = SemanticIngestionLargeDocumentConfig()
    retrieval = EnhancedContextualRetrieval(config)

    chunk = "Medium sized chunk content with business data. " * 100  # ~4500 characters
    context = "High priority contextual information that should be preserved."
    document_summary = (
        "Lower priority document summary that might be truncated or omitted."
    )
    section_context = "Medium priority section information."

    result = retrieval._create_token_safe_context(
        chunk_content=chunk,
        context=context,
        document_summary=document_summary,
        section_context=section_context,
    )

    # Original chunk should always be present (or most of it if truncated)
    chunk_tokens = retrieval._count_tokens(chunk)
    max_allowed = retrieval.max_embedding_tokens - retrieval.token_safety_margin

    if chunk_tokens <= max_allowed:
        assert chunk in result, "Chunk should be fully preserved when not too large"
    else:
        # If chunk was truncated, verify some content is preserved
        assert any(word in result for word in chunk.split()[:20]), (
            "Some chunk content should be preserved"
        )

    # High priority context should be present if there's space
    context_tokens = retrieval._count_tokens(context)
    if chunk_tokens + context_tokens <= max_allowed:
        assert context in result, (
            "High priority context should be preserved when space allows"
        )

    final_tokens = retrieval._count_tokens(result)
    max_allowed = retrieval.max_embedding_tokens - retrieval.token_safety_margin
    assert final_tokens <= max_allowed

    print(
        f"✅ Priority system works: {final_tokens}/{max_allowed} tokens, all high priority content preserved"
    )


if __name__ == "__main__":
    print("🧪 Testing Enhanced Contextual Retrieval Token Management")
    print("=" * 60)

    test_token_counting()
    test_token_safe_context_creation()
    test_extremely_large_chunk_handling()
    test_priority_system()

    print("=" * 60)
    print("✅ All token management tests passed!")
