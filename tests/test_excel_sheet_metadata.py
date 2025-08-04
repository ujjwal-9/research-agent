"""Test Excel sheet name metadata handling in semantic ingestion."""

import tempfile
import pandas as pd
import os
import logging
from unittest.mock import Mock

# Set up logging for testing
logging.basicConfig(level=logging.INFO)


def test_excel_sheet_metadata():
    """Test that Excel sheet names are properly extracted and stored in metadata."""

    # Create a test Excel file with multiple sheets
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        temp_path = f.name

    try:
        # Create Excel file with multiple sheets
        with pd.ExcelWriter(temp_path, engine="openpyxl") as writer:
            df1 = pd.DataFrame({"Name": ["Alice", "Bob"], "Age": [25, 30]})
            df2 = pd.DataFrame({"Product": ["Widget", "Gadget"], "Price": [10, 20]})
            df1.to_excel(writer, sheet_name="Employees", index=False)
            df2.to_excel(writer, sheet_name="Products", index=False)

        print(f"✅ Created test Excel file: {temp_path}")

        # Test the metadata extraction logic
        import sys

        sys.path.append(".")
        from src.ingestion_semantic.embeddings import SemanticEmbeddings

        # Create a mock config
        mock_config = Mock()
        mock_config.openai_api_key = "test-key"
        mock_config.embedding_model = "text-embedding-3-large"
        mock_config.qdrant_url = "http://localhost:6333"
        mock_config.qdrant_api_key = None
        mock_config.qdrant_collection_name = "test_collection"
        mock_config.max_concurrent_requests = 5

        # Initialize embeddings class (without actual connections)
        try:
            embedder = SemanticEmbeddings(mock_config)
        except Exception as e:
            print(f"⚠️  Skipping actual embedding initialization: {e}")
            # Test the helper methods directly
            embedder = Mock()
            embedder._is_excel_file = SemanticEmbeddings._is_excel_file.__get__(
                embedder
            )
            embedder._extract_sheet_name = (
                SemanticEmbeddings._extract_sheet_name.__get__(embedder)
            )
            embedder.logger = logging.getLogger(__name__)

        # Test Excel file detection
        is_excel = embedder._is_excel_file(temp_path)
        print(f"✅ Excel file detection: {is_excel}")
        assert is_excel, "Should detect Excel file"

        # Test sheet name extraction from different metadata formats
        test_cases = [
            {"page_name": "Employees", "source": temp_path},
            {"page_name": "Products", "source": temp_path, "page_number": 2},
            {"sheet_name": "DataSheet", "source": "test.xlsx"},
            {"source": "test_sheet_CustomSheet.html"},
            {"page_title": "Summary", "source": "report.xlsx"},
        ]

        expected_results = [
            "Employees",
            "Products",
            "DataSheet",
            "CustomSheet",
            "Summary",
        ]

        for i, metadata in enumerate(test_cases):
            sheet_name = embedder._extract_sheet_name(metadata)
            expected = expected_results[i]
            print(
                f"✅ Test case {i + 1}: {metadata} -> '{sheet_name}' (expected: '{expected}')"
            )
            assert sheet_name == expected, f"Expected '{expected}', got '{sheet_name}'"

        print("✅ All sheet name extraction tests passed!")

        # Test no sheet name case
        no_sheet_metadata = {"source": "document.pdf", "page": 1}
        no_sheet_name = embedder._extract_sheet_name(no_sheet_metadata)
        print(
            f"✅ No sheet case: {no_sheet_metadata} -> '{no_sheet_name}' (expected: None)"
        )
        assert no_sheet_name is None, "Should return None for non-Excel metadata"

        print("🎉 All tests passed!")

    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)
            print(f"🧹 Cleaned up test file: {temp_path}")


if __name__ == "__main__":
    test_excel_sheet_metadata()
