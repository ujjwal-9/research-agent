"""
Main Excel processing orchestrator.

This module coordinates the entire Excel processing pipeline:
1. Convert Excel files to HTML
2. Extract knowledge using OpenAI
3. Extract and save individual tables as HTML
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

from .excel_converter import ExcelToHTMLConverter
from .knowledge_extractor import KnowledgeExtractor
from .table_extractor import TableExtractor
from .header_detector import HeaderDetector

# Set up logging
logger = logging.getLogger(__name__)


class ExcelProcessor:
    """Main orchestrator for processing Excel files."""

    def __init__(
        self, openai_api_key: Optional[str] = None, log_dir: Optional[Path] = None
    ):
        """
        Initialize the Excel processor.

        Args:
            openai_api_key: OpenAI API key for knowledge extraction
            log_dir: Directory for logging files
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.log_dir = log_dir or Path("logs")

        # Initialize components
        self.excel_converter = ExcelToHTMLConverter(openai_api_key=self.openai_api_key)
        self.knowledge_extractor = (
            KnowledgeExtractor(api_key=self.openai_api_key)
            if self.openai_api_key
            else None
        )
        self.table_extractor = TableExtractor()
        self.header_detector = (
            HeaderDetector(api_key=self.openai_api_key) if self.openai_api_key else None
        )

        # Set up logging
        self._setup_logging()

        if not self.openai_api_key:
            logger.warning(
                "OpenAI API key not provided. Knowledge extraction will use fallback method."
            )

    def _setup_logging(self):
        """Set up logging for the processor."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.log_dir / "excel_processing.log"

        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

        logger.info("Excel processor initialized")

    def process_single_file(self, excel_path: Path, output_base_dir: Path) -> Dict:
        """
        Process a single Excel file through the complete pipeline.

        Args:
            excel_path: Path to Excel file
            output_base_dir: Base directory for outputs

        Returns:
            Dictionary containing processing results
        """
        logger.info(f"Starting processing of Excel file: {excel_path}")

        try:
            # Determine folder structure
            folder_name = excel_path.parent.name
            output_folder = output_base_dir / folder_name

            results = {
                "excel_file": str(excel_path),
                "folder_name": folder_name,
                "html_files": {},
                "knowledge_files": [],
                "table_files": [],
                "success": False,
                "errors": [],
            }

            # Step 1: Convert Excel to HTML
            logger.info("Step 1: Converting Excel to HTML")
            try:
                html_files = self.excel_converter.convert_excel_to_html(
                    excel_path, output_folder
                )
                results["html_files"] = html_files
                logger.info(f"Successfully converted {len(html_files)} sheets to HTML")
            except Exception as e:
                error_msg = f"Error converting Excel to HTML: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
                return results

            # Step 2: Extract knowledge from HTML files
            logger.info("Step 2: Extracting knowledge using OpenAI")
            try:
                if self.knowledge_extractor:
                    knowledge_files = {}
                    for sheet_name, html_path in html_files.items():
                        try:
                            html_file_path = Path(html_path)
                            knowledge_path = (
                                self.knowledge_extractor.extract_knowledge_from_file(
                                    html_file_path, output_folder, 0
                                )
                            )
                            knowledge_files[sheet_name] = str(knowledge_path)
                            # Add delay between API calls
                            time.sleep(0.5)
                        except Exception as e:
                            error_msg = f"Error extracting knowledge from {sheet_name}: {str(e)}"
                            logger.error(error_msg)
                            results["errors"].append(error_msg)

                    results["knowledge_files"] = knowledge_files
                    logger.info(
                        f"Successfully extracted knowledge from {len(knowledge_files)} sheets"
                    )
                else:
                    logger.warning(
                        "Knowledge extraction skipped - no OpenAI API key provided"
                    )
            except Exception as e:
                error_msg = f"Error in knowledge extraction: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)

            # Step 3: Extract tables and save as separate HTML files
            logger.info("Step 3: Extracting tables as separate HTML files")
            try:
                all_table_files = []
                for sheet_name, html_path in html_files.items():
                    try:
                        html_file_path = Path(html_path)
                        table_files = self.table_extractor.extract_tables_from_file(
                            html_file_path, output_folder, excel_path.name
                        )
                        all_table_files.extend([str(path) for path in table_files])
                    except Exception as e:
                        error_msg = (
                            f"Error extracting tables from {sheet_name}: {str(e)}"
                        )
                        logger.error(error_msg)
                        results["errors"].append(error_msg)

                results["table_files"] = all_table_files
                logger.info(f"Successfully extracted {len(all_table_files)} tables")
            except Exception as e:
                error_msg = f"Error in table extraction: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)

            # Mark as successful if no critical errors
            results["success"] = len(results["html_files"]) > 0

            logger.info(
                f"Completed processing of {excel_path}. Success: {results['success']}"
            )
            return results

        except Exception as e:
            error_msg = f"Unexpected error processing {excel_path}: {str(e)}"
            logger.error(error_msg)
            return {
                "excel_file": str(excel_path),
                "folder_name": excel_path.parent.name,
                "html_files": {},
                "knowledge_files": [],
                "table_files": [],
                "success": False,
                "errors": [error_msg],
            }

    def process_directory(self, input_dir: Path, output_base_dir: Path = None) -> Dict:
        """
        Process all Excel files in a directory.

        Args:
            input_dir: Directory containing Excel files
            output_base_dir: Base directory for outputs (default: data/processed_sheet)

        Returns:
            Dictionary containing processing results for all files
        """
        if output_base_dir is None:
            output_base_dir = Path("data/processed_sheet")

        logger.info(f"Starting batch processing of directory: {input_dir}")

        # Find all Excel files
        excel_files = []
        for pattern in ["*.xlsx", "*.xls"]:
            excel_files.extend(input_dir.rglob(pattern))

        logger.info(f"Found {len(excel_files)} Excel files to process")

        if not excel_files:
            logger.warning("No Excel files found in the directory")
            return {
                "total_files": 0,
                "successful": 0,
                "failed": 0,
                "results": [],
                "summary": {
                    "total_html_files": 0,
                    "total_knowledge_files": 0,
                    "total_table_files": 0,
                },
            }

        # Process each file
        results = []
        successful = 0
        failed = 0
        total_html_files = 0
        total_knowledge_files = 0
        total_table_files = 0

        for excel_path in excel_files:
            try:
                logger.info(
                    f"Processing file {len(results) + 1}/{len(excel_files)}: {excel_path}"
                )

                result = self.process_single_file(excel_path, output_base_dir)
                results.append(result)

                if result["success"]:
                    successful += 1
                    total_html_files += len(result["html_files"])
                    total_knowledge_files += (
                        len(result["knowledge_files"])
                        if isinstance(result["knowledge_files"], dict)
                        else len(result["knowledge_files"])
                    )
                    total_table_files += len(result["table_files"])
                else:
                    failed += 1

                # Log progress
                logger.info(
                    f"Progress: {len(results)}/{len(excel_files)} files processed"
                )

            except Exception as e:
                logger.error(f"Unexpected error processing {excel_path}: {str(e)}")
                failed += 1
                results.append(
                    {
                        "excel_file": str(excel_path),
                        "folder_name": excel_path.parent.name,
                        "html_files": {},
                        "knowledge_files": [],
                        "table_files": [],
                        "success": False,
                        "errors": [f"Unexpected error: {str(e)}"],
                    }
                )

        # Create summary
        batch_results = {
            "total_files": len(excel_files),
            "successful": successful,
            "failed": failed,
            "results": results,
            "summary": {
                "total_html_files": total_html_files,
                "total_knowledge_files": total_knowledge_files,
                "total_table_files": total_table_files,
            },
        }

        logger.info(
            f"Batch processing completed. Success: {successful}, Failed: {failed}"
        )
        logger.info(
            f"Generated: {total_html_files} HTML files, {total_knowledge_files} knowledge files, {total_table_files} table files"
        )

        return batch_results

    def generate_processing_report(
        self, results: Dict, output_path: Optional[Path] = None
    ) -> str:
        """
        Generate a detailed processing report.

        Args:
            results: Processing results from process_directory
            output_path: Optional path to save the report

        Returns:
            Report as string
        """
        report_lines = [
            "=" * 80,
            "EXCEL PROCESSING REPORT",
            "=" * 80,
            "",
            f"Total files processed: {results['total_files']}",
            f"Successful: {results['successful']}",
            f"Failed: {results['failed']}",
            (
                f"Success rate: {(results['successful'] / results['total_files'] * 100):.1f}%"
                if results["total_files"] > 0
                else "N/A"
            ),
            "",
            "SUMMARY:",
            f"  - HTML files generated: {results['summary']['total_html_files']}",
            f"  - Knowledge files generated: {results['summary']['total_knowledge_files']}",
            f"  - Table files generated: {results['summary']['total_table_files']}",
            "",
            "DETAILED RESULTS:",
            "-" * 40,
        ]

        for i, result in enumerate(results["results"], 1):
            status = "✓ SUCCESS" if result["success"] else "✗ FAILED"
            report_lines.extend(
                [
                    f"{i}. {result['excel_file']} - {status}",
                    f"   Folder: {result['folder_name']}",
                    f"   HTML files: {len(result['html_files'])}",
                    f"   Knowledge files: {len(result['knowledge_files']) if isinstance(result['knowledge_files'], dict) else len(result['knowledge_files'])}",
                    f"   Table files: {len(result['table_files'])}",
                ]
            )

            if result["errors"]:
                report_lines.append("   Errors:")
                for error in result["errors"]:
                    report_lines.append(f"     - {error}")

            report_lines.append("")

        report = "\n".join(report_lines)

        # Save report if path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info(f"Processing report saved to: {output_path}")

        return report


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process Excel files to HTML, knowledge, and table extracts"
    )
    parser.add_argument("input_dir", type=str, help="Directory containing Excel files")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed_sheet",
        help="Output directory (default: data/processed_sheet)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenAI API key (optional, can use OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs", help="Log directory (default: logs)"
    )
    parser.add_argument("--report", type=str, help="Path to save processing report")

    args = parser.parse_args()

    # Initialize processor
    processor = ExcelProcessor(openai_api_key=args.api_key, log_dir=Path(args.log_dir))

    # Process directory
    results = processor.process_directory(Path(args.input_dir), Path(args.output_dir))

    # Generate and print report
    report_path = Path(args.report) if args.report else None
    report = processor.generate_processing_report(results, report_path)
    print(report)


if __name__ == "__main__":
    main()
