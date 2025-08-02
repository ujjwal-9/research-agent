"""
Test script for parallel processing functionality.

This script tests the parallel processing of multiple sheets.
"""

import sys
import os
from pathlib import Path
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.excel_processor import ExcelProcessor


def test_parallel_processing():
    """Test the parallel processing functionality."""

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting parallel processing test")

    try:
        # Initialize processor with parallel processing
        processor = ExcelProcessor(log_dir=Path("logs"))

        # Check parallelization setting
        logger.info(f"Parallel sheets setting: {processor.parallel_sheets}")

        # Find a complex test file with multiple sheets
        test_files = [
            Path(
                "data/documents/7. Product (Saki)/Discovery Sprints/Marketplace Discovery Phase Learning Plan & Sprint Reviews.xlsx"
            ),
            Path("data/documents/11. Financial Model/Saki_Financial Model_v26.xlsx"),
            Path(
                "data/documents/0. Workplans - To-Dos/Copy of Quartz_UPMC Pilot Study Planning Tracker.xlsx"
            ),
        ]

        # Find the first available test file
        test_file = None
        for file_path in test_files:
            if file_path.exists():
                test_file = file_path
                break

        if not test_file:
            logger.error("No test files found for parallel processing test")
            return False

        output_dir = Path("data/processed_sheet")

        logger.info(f"Testing parallel processing with: {test_file}")
        logger.info(f"Parallel sheets: {processor.parallel_sheets}")

        # Process single file with parallel processing
        import time

        start_time = time.time()

        result = processor.process_single_file(test_file, output_dir)

        end_time = time.time()
        processing_time = end_time - start_time

        # Check results
        if result["success"]:
            logger.info("✓ Parallel processing test completed successfully!")
            logger.info(f"Processing time: {processing_time:.2f} seconds")
            logger.info(f"HTML files: {len(result['html_files'])}")
            logger.info(f"Knowledge files: {len(result['knowledge_files'])}")
            logger.info(f"Table files: {len(result['table_files'])}")

            if result["errors"]:
                logger.warning(f"Errors encountered: {result['errors']}")

            return True
        else:
            logger.error("✗ Parallel processing failed")
            logger.error(f"Errors: {result['errors']}")
            return False

    except Exception as e:
        logger.error(f"✗ Test failed with error: {str(e)}")
        return False


if __name__ == "__main__":
    success = test_parallel_processing()
    sys.exit(0 if success else 1)
