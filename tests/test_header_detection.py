"""
Test script for header detection functionality.

This script tests the AI-powered header detection without requiring large Excel files.
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

from src.preprocessing.header_detector import HeaderDetector


def test_header_detection():
    """Test the header detection functionality."""

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting header detection test")

    # Sample data to test with
    test_data = [
        ["Financial Report Q4 2023", "", "", ""],
        ["", "", "", ""],
        ["Product Name", "Sales", "Profit", "Region"],
        ["Widget A", "1000", "200", "North"],
        ["Widget B", "1500", "300", "South"],
        ["Widget C", "800", "150", "East"],
    ]

    try:
        # Test without API key (fallback method)
        logger.info("Testing fallback header detection...")
        try:
            detector = HeaderDetector()
        except ValueError:
            logger.info("No API key available, testing fallback method directly")
            detector = HeaderDetector.__new__(HeaderDetector)
            detector.header_check_rows = 10
            header_row = detector._fallback_header_detection(test_data)
            logger.info(f"Fallback detected header row: {header_row}")

            headers, data_rows = [], []
            if 0 <= header_row < len(test_data):
                headers = test_data[header_row]
                data_rows = test_data[header_row + 1 :]

            logger.info(f"Headers: {headers}")
            logger.info(f"Data rows: {len(data_rows)}")
            return True

        # Test with API key if available
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            logger.info("Testing AI-powered header detection...")
            header_row = detector.detect_header_row(test_data, "Test Sheet")
            logger.info(f"AI detected header row: {header_row}")

            headers, data_rows = detector.get_headers_and_data(test_data, "Test Sheet")
            logger.info(f"Headers: {headers}")
            logger.info(f"Data rows: {len(data_rows)}")

            logger.info("✓ Header detection test completed successfully!")
        else:
            logger.info("✓ Fallback header detection test completed successfully!")

        return True

    except Exception as e:
        logger.error(f"✗ Test failed with error: {str(e)}")
        return False


if __name__ == "__main__":
    success = test_header_detection()
    sys.exit(0 if success else 1)
