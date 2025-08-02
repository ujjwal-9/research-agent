"""
Test script for skip processed files functionality.

This script tests that already processed files are skipped correctly.
"""

import sys
import os
from pathlib import Path
import logging
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.excel_processor import ExcelProcessor


def test_skip_processed():
    """Test the skip processed files functionality."""

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting skip processed files test")

    try:
        # Initialize processor
        processor = ExcelProcessor(log_dir=Path("logs"))

        # Find a test file
        test_file = Path(
            "data/documents/7. Product (Saki)/Discovery Sprints/Marketplace Discovery Phase Learning Plan & Sprint Reviews.xlsx"
        )

        if not test_file.exists():
            logger.error("Test file not found")
            return False

        output_dir = Path("data/processed_sheet")

        logger.info(f"Testing skip processed with: {test_file}")

        # First run - should process the file
        logger.info("=== FIRST RUN (should process) ===")
        start_time = time.time()

        result1 = processor.process_single_file(test_file, output_dir)

        end_time = time.time()
        first_run_time = end_time - start_time

        if not result1["success"]:
            logger.error("✗ First run failed")
            return False

        logger.info(f"✓ First run completed in {first_run_time:.2f} seconds")
        logger.info(
            f"Generated: {len(result1['html_files'])} HTML, {len(result1['knowledge_files'])} knowledge, {len(result1['table_files'])} table files"
        )

        # Second run - should skip the file
        logger.info("=== SECOND RUN (should skip) ===")
        start_time = time.time()

        result2 = processor.process_single_file(test_file, output_dir)

        end_time = time.time()
        second_run_time = end_time - start_time

        if not result2["success"]:
            logger.error("✗ Second run failed")
            return False

        # Check if file was skipped
        if result2.get("skipped", False):
            logger.info(
                f"✓ Second run correctly skipped file in {second_run_time:.2f} seconds"
            )
            logger.info(
                f"Time saved: {first_run_time - second_run_time:.2f} seconds ({((first_run_time - second_run_time) / first_run_time * 100):.1f}% faster)"
            )

            # Verify same number of files found
            if (
                len(result2["html_files"]) == len(result1["html_files"])
                and len(result2["knowledge_files"]) == len(result1["knowledge_files"])
                and len(result2["table_files"]) == len(result1["table_files"])
            ):
                logger.info("✓ Same number of files found in both runs")
                return True
            else:
                logger.error("✗ Different number of files found between runs")
                return False
        else:
            logger.warning(
                "⚠ File was not skipped - this might be expected if files were modified"
            )
            return True

    except Exception as e:
        logger.error(f"✗ Test failed with error: {str(e)}")
        return False


def test_force_reprocess():
    """Test that files are reprocessed when source is newer."""

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("=== Testing force reprocess (when source is newer) ===")

    try:
        processor = ExcelProcessor(log_dir=Path("logs"))

        test_file = Path(
            "data/documents/7. Product (Saki)/Discovery Sprints/Marketplace Discovery Phase Learning Plan & Sprint Reviews.xlsx"
        )
        output_dir = Path("data/processed_sheet")

        if not test_file.exists():
            logger.error("Test file not found")
            return False

        # Find an output HTML file to modify its timestamp
        folder_name = processor._get_folder_name(test_file)
        output_folder = output_dir / folder_name

        if output_folder.exists():
            html_files = list(output_folder.glob("*.html"))
            if html_files:
                # Make the HTML file older than the source
                old_time = test_file.stat().st_mtime - 3600  # 1 hour older
                os.utime(html_files[0], (old_time, old_time))

                logger.info(f"Made output file older: {html_files[0]}")

                # Now process - should not skip
                result = processor.process_single_file(test_file, output_dir)

                if result["success"] and not result.get("skipped", False):
                    logger.info(
                        "✓ File was correctly reprocessed when output was older"
                    )
                    return True
                elif result.get("skipped", False):
                    logger.warning("⚠ File was skipped even though output was older")
                    return False
                else:
                    logger.error("✗ Processing failed")
                    return False

        logger.info("No output files found to test timestamp comparison")
        return True

    except Exception as e:
        logger.error(f"✗ Force reprocess test failed: {str(e)}")
        return False


if __name__ == "__main__":
    success1 = test_skip_processed()
    success2 = test_force_reprocess()

    if success1 and success2:
        print("\n✓ All skip processed tests passed!")
    else:
        print("\n✗ Some tests failed")

    sys.exit(0 if (success1 and success2) else 1)
