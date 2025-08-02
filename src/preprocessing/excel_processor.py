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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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

        # Get parallelization setting
        try:
            self.parallel_sheets = int(os.getenv("PARALLEL_SHEETS", "3"))
        except ValueError:
            self.parallel_sheets = 3
            logger.warning("Invalid PARALLEL_SHEETS value, using default: 3")

        # Thread lock for thread-safe logging and API calls
        self.thread_lock = threading.Lock()

        # Set up logging
        self._setup_logging()

        if not self.openai_api_key:
            logger.warning(
                "OpenAI API key not provided. Knowledge extraction will use fallback method."
            )

        logger.info(
            f"Parallel processing enabled: {self.parallel_sheets} sheets at a time"
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

            # Check if file has already been processed
            if self._is_already_processed(excel_path, output_folder):
                logger.info(f"✓ File already processed, skipping: {excel_path}")
                return self._get_existing_results(excel_path, output_folder)

            results = {
                "excel_file": str(excel_path),
                "folder_name": folder_name,
                "html_files": {},
                "knowledge_files": [],
                "table_files": [],
                "success": False,
                "errors": [],
            }

            # Step 1: Convert Excel to HTML (parallelized)
            logger.info("Step 1: Converting Excel to HTML (parallel processing)")
            try:
                html_files = self._convert_excel_parallel(excel_path, output_folder)
                results["html_files"] = html_files
                logger.info(f"Successfully converted {len(html_files)} sheets to HTML")
            except Exception as e:
                error_msg = f"Error converting Excel to HTML: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
                return results

            # Step 2: Extract knowledge from HTML files (parallelized)
            logger.info(
                "Step 2: Extracting knowledge using OpenAI (parallel processing)"
            )
            try:
                if self.knowledge_extractor:
                    knowledge_files = self._extract_knowledge_parallel(
                        html_files, output_folder
                    )
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

            # Step 3: Extract tables and save as separate HTML files (parallelized)
            logger.info(
                "Step 3: Extracting tables as separate HTML files (parallel processing)"
            )
            try:
                all_table_files = self._extract_tables_parallel(
                    html_files, output_folder, excel_path.name
                )
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
            if result["success"] and result.get("skipped", False):
                status = "⏭ SKIPPED (already processed)"
            elif result["success"]:
                status = "✓ SUCCESS"
            else:
                status = "✗ FAILED"

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

    def _convert_excel_parallel(
        self, excel_path: Path, output_folder: Path
    ) -> Dict[str, str]:
        """
        Convert Excel sheets to HTML in parallel.

        Args:
            excel_path: Path to Excel file
            output_folder: Output directory

        Returns:
            Dictionary mapping sheet names to HTML file paths
        """
        import pandas as pd

        try:
            # Read all sheet names first
            excel_file = pd.ExcelFile(excel_path)
            sheet_names = excel_file.sheet_names

            logger.info(
                f"Processing {len(sheet_names)} sheets in parallel (max {self.parallel_sheets} at a time)"
            )

            html_files = {}

            def process_single_sheet(sheet_name):
                """Process a single sheet."""
                try:
                    with self.thread_lock:
                        logger.info(f"Processing sheet: {sheet_name}")

                    # Read the sheet without assuming headers
                    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)

                    # Use header detector to properly set headers if available
                    if self.excel_converter.header_detector and not df.empty:
                        try:
                            df = self.excel_converter.header_detector.process_dataframe_with_detected_headers(
                                df, sheet_name
                            )
                        except Exception as e:
                            with self.thread_lock:
                                logger.warning(
                                    f"Header detection failed for sheet {sheet_name}: {str(e)}, using original data"
                                )

                    # Convert to HTML
                    html_content = self.excel_converter._dataframe_to_html(
                        df, sheet_name
                    )

                    # Create output file path
                    safe_sheet_name = self.excel_converter._sanitize_filename(
                        sheet_name
                    )
                    html_filename = f"{excel_path.stem}_{safe_sheet_name}.html"
                    html_path = output_folder / html_filename

                    # Ensure output directory exists
                    html_path.parent.mkdir(parents=True, exist_ok=True)

                    # Write HTML file
                    with open(html_path, "w", encoding="utf-8") as f:
                        f.write(html_content)

                    with self.thread_lock:
                        logger.info(
                            f"Saved HTML for sheet '{sheet_name}' to: {html_path}"
                        )

                    return sheet_name, str(html_path)

                except Exception as e:
                    with self.thread_lock:
                        logger.error(f"Error processing sheet {sheet_name}: {str(e)}")
                    return sheet_name, None

            # Process sheets in parallel
            with ThreadPoolExecutor(max_workers=self.parallel_sheets) as executor:
                # Submit all tasks
                future_to_sheet = {
                    executor.submit(process_single_sheet, sheet_name): sheet_name
                    for sheet_name in sheet_names
                }

                # Collect results
                for future in as_completed(future_to_sheet):
                    sheet_name, html_path = future.result()
                    if html_path:
                        html_files[sheet_name] = html_path

            return html_files

        except Exception as e:
            logger.error(f"Error in parallel Excel conversion: {str(e)}")
            raise

    def _extract_knowledge_parallel(
        self, html_files: Dict[str, str], output_folder: Path
    ) -> Dict[str, str]:
        """
        Extract knowledge from HTML files in parallel.

        Args:
            html_files: Dictionary mapping sheet names to HTML file paths
            output_folder: Output directory

        Returns:
            Dictionary mapping sheet names to knowledge file paths
        """
        logger.info(f"Extracting knowledge from {len(html_files)} sheets in parallel")

        knowledge_files = {}

        def extract_single_knowledge(sheet_name, html_path):
            """Extract knowledge from a single HTML file."""
            try:
                html_file_path = Path(html_path)
                knowledge_path = self.knowledge_extractor.extract_knowledge_from_file(
                    html_file_path, output_folder, 0
                )

                with self.thread_lock:
                    logger.info(f"Extracted knowledge for sheet: {sheet_name}")

                return sheet_name, str(knowledge_path)

            except Exception as e:
                with self.thread_lock:
                    logger.error(
                        f"Error extracting knowledge from {sheet_name}: {str(e)}"
                    )
                return sheet_name, None

        # Process in parallel with rate limiting
        with ThreadPoolExecutor(max_workers=self.parallel_sheets) as executor:
            # Submit all tasks
            future_to_sheet = {
                executor.submit(
                    extract_single_knowledge, sheet_name, html_path
                ): sheet_name
                for sheet_name, html_path in html_files.items()
            }

            # Collect results with rate limiting delay
            for i, future in enumerate(as_completed(future_to_sheet)):
                sheet_name, knowledge_path = future.result()
                if knowledge_path:
                    knowledge_files[sheet_name] = knowledge_path

                # Add small delay to respect API rate limits
                if i < len(future_to_sheet) - 1:  # Don't delay after the last one
                    time.sleep(0.5)

        return knowledge_files

    def _extract_tables_parallel(
        self, html_files: Dict[str, str], output_folder: Path, original_file_name: str
    ) -> List[str]:
        """
        Extract tables from HTML files in parallel.

        Args:
            html_files: Dictionary mapping sheet names to HTML file paths
            output_folder: Output directory
            original_file_name: Name of the original Excel file

        Returns:
            List of paths to extracted table HTML files
        """
        logger.info(f"Extracting tables from {len(html_files)} sheets in parallel")

        all_table_files = []

        def extract_single_tables(sheet_name, html_path):
            """Extract tables from a single HTML file."""
            try:
                html_file_path = Path(html_path)
                table_files = self.table_extractor.extract_tables_from_file(
                    html_file_path, output_folder, original_file_name
                )

                with self.thread_lock:
                    logger.info(
                        f"Extracted {len(table_files)} tables from sheet: {sheet_name}"
                    )

                return [str(path) for path in table_files]

            except Exception as e:
                with self.thread_lock:
                    logger.error(f"Error extracting tables from {sheet_name}: {str(e)}")
                return []

        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.parallel_sheets) as executor:
            # Submit all tasks
            futures = [
                executor.submit(extract_single_tables, sheet_name, html_path)
                for sheet_name, html_path in html_files.items()
            ]

            # Collect results
            for future in as_completed(futures):
                table_files = future.result()
                all_table_files.extend(table_files)

        return all_table_files

    def _is_already_processed(self, excel_path: Path, output_folder: Path) -> bool:
        """
        Check if an Excel file has already been processed.

        Args:
            excel_path: Path to the Excel file
            output_folder: Output directory for this file

        Returns:
            True if file has been processed, False otherwise
        """
        try:
            logger.info(f"Checking if file already processed: {excel_path}")
            logger.info(f"Output folder: {output_folder}")

            # Check if output folder exists
            if not output_folder.exists():
                logger.info("Output folder does not exist - needs processing")
                return False

            # Get modification time of source file
            source_mtime = excel_path.stat().st_mtime
            logger.info(f"Source file modification time: {source_mtime}")

            # Check for HTML files (primary indicator)
            html_files = list(output_folder.glob("*.html"))
            logger.info(f"Found {len(html_files)} HTML files in output folder")

            if not html_files:
                logger.info("No HTML files found - needs processing")
                return False

            # Check knowledge files as additional indicator
            knowledge_files = list(output_folder.glob("*__knowledge_table_*.txt"))
            logger.info(f"Found {len(knowledge_files)} knowledge files")

            # Require both HTML and knowledge files to consider it processed
            if len(knowledge_files) == 0:
                logger.info("No knowledge files found - needs processing")
                return False

            # Check if any HTML file is newer than source
            newest_html_time = 0
            for html_file in html_files:
                html_mtime = html_file.stat().st_mtime
                newest_html_time = max(newest_html_time, html_mtime)
                logger.info(f"HTML file {html_file.name}: {html_mtime}")

            if newest_html_time > source_mtime:
                # File has been processed and is up to date
                logger.info(
                    f"✓ Files are up to date (newest: {newest_html_time} > source: {source_mtime})"
                )
                return True
            else:
                # If HTML files exist but are older than source, reprocess
                logger.info(
                    f"Output files are older than source file - needs reprocessing"
                )
                return False

        except Exception as e:
            logger.warning(f"Error checking if file is processed: {str(e)}")
            return False

    def _get_existing_results(
        self, excel_path: Path, output_folder: Path
    ) -> Dict[str, any]:
        """
        Get results for an already processed file.

        Args:
            excel_path: Path to the Excel file
            output_folder: Output directory for this file

        Returns:
            Dictionary containing existing results
        """
        try:
            folder_name = excel_path.parent.name

            # Find existing files
            html_files = {}
            knowledge_files = {}
            table_files = []

            # Get HTML files
            for html_file in output_folder.glob("*.html"):
                if "_table_" not in html_file.name:  # Skip table files
                    # Extract sheet name from filename
                    filename = html_file.stem
                    # Remove the Excel filename prefix to get sheet name
                    sheet_name = filename.replace(f"{excel_path.stem}_", "", 1)
                    html_files[sheet_name] = str(html_file)
                else:
                    table_files.append(str(html_file))

            # Get knowledge files
            for knowledge_file in output_folder.glob("*__knowledge_table_*.txt"):
                # Extract sheet name from filename
                filename = knowledge_file.stem
                # Parse: <sheet_name>__knowledge_table_<number>
                if "__knowledge_table_" in filename:
                    sheet_name = filename.split("__knowledge_table_")[0]
                    knowledge_files[sheet_name] = str(knowledge_file)

            results = {
                "excel_file": str(excel_path),
                "folder_name": folder_name,
                "html_files": html_files,
                "knowledge_files": knowledge_files,
                "table_files": table_files,
                "success": True,
                "errors": [],
                "skipped": True,  # Add flag to indicate this was skipped
            }

            logger.info(
                f"Found existing results: {len(html_files)} HTML, {len(knowledge_files)} knowledge, {len(table_files)} table files"
            )
            return results

        except Exception as e:
            logger.error(f"Error reading existing results: {str(e)}")
            # Return empty results to trigger reprocessing
            return {
                "excel_file": str(excel_path),
                "folder_name": excel_path.parent.name,
                "html_files": {},
                "knowledge_files": {},
                "table_files": [],
                "success": False,
                "errors": [f"Error reading existing results: {str(e)}"],
            }


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
