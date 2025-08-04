"""
Test script for processing a single Excel file.

This script tests the complete pipeline with header detection on one file.
"""

import sys
from pathlib import Path
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.excel_processor import ExcelProcessor


def test_single_file():
    """Test processing a single Excel file."""

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting single file processing test")

    try:
        # Initialize processor with OpenAI API key
        processor = ExcelProcessor(log_dir=Path("logs"))

        # Find a test file
        test_file = Path(
            "data/documents/0. Workplans - To-Dos/Saki Workplan_082023.xlsx"
        )
        output_dir = Path("data/processed_sheet")

        if not test_file.exists():
            logger.error(f"Test file does not exist: {test_file}")
            return False

        logger.info(f"Processing test file: {test_file}")

        # Process single file
        result = processor.process_single_file(test_file, output_dir)

        # Check results
        if result["success"]:
            logger.info("✓ Single file processing completed successfully!")
            logger.info(f"HTML files: {len(result['html_files'])}")
            logger.info(f"Knowledge files: {len(result['knowledge_files'])}")
            logger.info(f"Table files: {len(result['table_files'])}")

            if result["errors"]:
                logger.warning(f"Errors encountered: {result['errors']}")

            return True
        else:
            logger.error("✗ Processing failed")
            logger.error(f"Errors: {result['errors']}")
            return False

    except Exception as e:
        logger.error(f"✗ Test failed with error: {str(e)}")
        return False


if __name__ == "__main__":
    success = test_single_file()
    sys.exit(0 if success else 1)
