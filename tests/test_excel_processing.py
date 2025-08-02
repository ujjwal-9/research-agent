"""
Test script for Excel processing functionality.

This script tests the Excel processing pipeline without requiring OpenAI API key.
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


def test_excel_processing():
    """Test the Excel processing pipeline."""

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting Excel processing test")

    try:
        # Initialize processor without OpenAI API key (will use fallback for knowledge extraction)
        processor = ExcelProcessor(log_dir=Path("logs"))

        # Define directories
        input_dir = Path("data/documents")
        output_dir = Path("data/processed_sheet")

        if not input_dir.exists():
            logger.error(f"Input directory does not exist: {input_dir}")
            return False

        logger.info(f"Processing Excel files from: {input_dir}")
        logger.info(f"Output will be saved to: {output_dir}")

        # Process all Excel files in the directory
        results = processor.process_directory(input_dir, output_dir)

        # Generate and display report
        report = processor.generate_processing_report(
            results, output_path=Path("logs/processing_report.txt")
        )

        print("\n" + "=" * 80)
        print("PROCESSING COMPLETED")
        print("=" * 80)
        print(report)

        # Check if we had any successful processing
        if results["successful"] > 0:
            logger.info("✓ Test completed successfully!")
            return True
        else:
            logger.warning("⚠ No files were processed successfully")
            return False

    except Exception as e:
        logger.error(f"✗ Test failed with error: {str(e)}")
        return False


if __name__ == "__main__":
    success = test_excel_processing()
    sys.exit(0 if success else 1)
