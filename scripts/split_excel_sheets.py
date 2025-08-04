"""
Script to split Excel sheets into separate Excel files and move originals to backup location.

This script:
1. Finds all Excel files in the data/documents directory (recursively)
2. Splits each Excel file into separate Excel files for each sheet
3. Saves the split sheets maintaining the directory structure
4. Moves the original Excel files to data/original_sheets (maintaining directory structure)
"""

import logging
import os
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)


class ExcelSheetSplitter:
    """Splits Excel files into separate files for each sheet and manages file movement."""

    def __init__(
        self, source_dir: Path, split_dir: Path, backup_dir: Path, log_dir: Path
    ):
        """
        Initialize the Excel sheet splitter.

        Args:
            source_dir: Directory containing Excel files to split
            split_dir: Directory to save split Excel files
            backup_dir: Directory to move original Excel files
            log_dir: Directory for log files
        """
        self.source_dir = Path(source_dir)
        self.split_dir = Path(split_dir)
        self.backup_dir = Path(backup_dir)
        self.log_dir = Path(log_dir)

        # Ensure directories exist
        self.split_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        self._setup_logging()

        self.supported_formats = [".xlsx", ".xls"]
        self.processed_files = []
        self.failed_files = []

    def _setup_logging(self):
        """Set up logging for the splitter."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"excel_sheet_splitting_{timestamp}.log"

        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
            force=True,
        )

        logger.info("Excel sheet splitter initialized")
        logger.info(f"Source directory: {self.source_dir}")
        logger.info(f"Split directory: {self.split_dir}")
        logger.info(f"Backup directory: {self.backup_dir}")

    def find_excel_files(self) -> List[Path]:
        """
        Find all Excel files in the source directory recursively.

        Returns:
            List of paths to Excel files
        """
        excel_files = []

        for ext in self.supported_formats:
            pattern = f"**/*{ext}"
            files = list(self.source_dir.glob(pattern))
            excel_files.extend(files)

        logger.info(f"Found {len(excel_files)} Excel files to process")

        for file in excel_files:
            logger.info(f"  - {file}")

        return excel_files

    def get_relative_path(self, file_path: Path) -> Path:
        """
        Get relative path from source directory.

        Args:
            file_path: Absolute path to file

        Returns:
            Relative path from source directory
        """
        return file_path.relative_to(self.source_dir)

    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to be safe for filesystem.

        Args:
            filename: Original filename

        Returns:
            Sanitized filename
        """
        # Replace problematic characters
        invalid_chars = ["<", ">", ":", '"', "/", "\\", "|", "?", "*"]
        sanitized = filename

        for char in invalid_chars:
            sanitized = sanitized.replace(char, "_")

        # Remove leading/trailing spaces and dots
        sanitized = sanitized.strip(" .")

        # Truncate if too long (leave room for extension)
        if len(sanitized) > 200:
            sanitized = sanitized[:200]

        return sanitized

    def split_excel_file(self, excel_path: Path) -> Dict[str, str]:
        """
        Split an Excel file into separate files for each sheet.

        Args:
            excel_path: Path to the Excel file

        Returns:
            Dictionary mapping sheet names to created file paths
        """
        logger.info(f"Splitting Excel file: {excel_path}")

        try:
            # Get relative path to maintain directory structure
            relative_path = self.get_relative_path(excel_path)
            output_base_dir = self.split_dir / relative_path.parent
            output_base_dir.mkdir(parents=True, exist_ok=True)

            # Read Excel file to get sheet names
            excel_file = pd.ExcelFile(excel_path)
            sheet_names = excel_file.sheet_names

            logger.info(f"Found {len(sheet_names)} sheets: {sheet_names}")

            created_files = {}

            for sheet_name in sheet_names:
                try:
                    # Read the specific sheet
                    df = pd.read_excel(excel_path, sheet_name=sheet_name)

                    # Skip empty sheets
                    if df.empty:
                        logger.warning(f"Sheet '{sheet_name}' is empty, skipping")
                        continue

                    # Create filename for the split sheet
                    sanitized_sheet_name = self.sanitize_filename(sheet_name)
                    base_filename = excel_path.stem
                    new_filename = f"{base_filename}_{sanitized_sheet_name}.xlsx"
                    output_path = output_base_dir / new_filename

                    # Save as new Excel file
                    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                        df.to_excel(writer, sheet_name=sheet_name, index=False)

                    created_files[sheet_name] = str(output_path)
                    logger.info(f"Created: {output_path}")

                except Exception as e:
                    logger.error(f"Error processing sheet '{sheet_name}': {str(e)}")
                    continue

            return created_files

        except Exception as e:
            logger.error(f"Error splitting Excel file {excel_path}: {str(e)}")
            raise

    def move_original_file(self, excel_path: Path) -> Path:
        """
        Move original Excel file to backup directory maintaining directory structure.

        Args:
            excel_path: Path to the original Excel file

        Returns:
            Path where the file was moved
        """
        try:
            # Get relative path to maintain directory structure
            relative_path = self.get_relative_path(excel_path)
            backup_path = self.backup_dir / relative_path

            # Create backup directory structure
            backup_path.parent.mkdir(parents=True, exist_ok=True)

            # Move the file
            shutil.move(str(excel_path), str(backup_path))
            logger.info(f"Moved original file: {excel_path} -> {backup_path}")

            return backup_path

        except Exception as e:
            logger.error(f"Error moving file {excel_path}: {str(e)}")
            raise

    def process_single_file(self, excel_path: Path) -> Dict:
        """
        Process a single Excel file: split sheets and move original.

        Args:
            excel_path: Path to Excel file

        Returns:
            Dictionary containing processing results
        """
        result = {
            "original_file": str(excel_path),
            "split_files": {},
            "backup_location": None,
            "success": False,
            "errors": [],
        }

        try:
            # Split the Excel file
            split_files = self.split_excel_file(excel_path)
            result["split_files"] = split_files

            # Move original file to backup
            backup_location = self.move_original_file(excel_path)
            result["backup_location"] = str(backup_location)

            result["success"] = True
            logger.info(f"Successfully processed: {excel_path}")

        except Exception as e:
            error_msg = f"Error processing {excel_path}: {str(e)}"
            logger.error(error_msg)
            result["errors"].append(error_msg)

        return result

    def process_all_files(self) -> Dict:
        """
        Process all Excel files in the source directory.

        Returns:
            Dictionary containing summary of processing results
        """
        logger.info("Starting processing of all Excel files")

        # Find all Excel files
        excel_files = self.find_excel_files()

        if not excel_files:
            logger.warning("No Excel files found to process")
            return {
                "total_files": 0,
                "successful": 0,
                "failed": 0,
                "results": [],
                "summary": {"total_sheets_created": 0, "files_backed_up": 0},
            }

        # Process each file
        results = []
        successful = 0
        failed = 0
        total_sheets_created = 0
        files_backed_up = 0

        for i, excel_path in enumerate(excel_files, 1):
            logger.info(f"Processing file {i}/{len(excel_files)}: {excel_path}")

            result = self.process_single_file(excel_path)
            results.append(result)

            if result["success"]:
                successful += 1
                total_sheets_created += len(result["split_files"])
                if result["backup_location"]:
                    files_backed_up += 1
            else:
                failed += 1

        # Create summary
        summary = {
            "total_files": len(excel_files),
            "successful": successful,
            "failed": failed,
            "results": results,
            "summary": {
                "total_sheets_created": total_sheets_created,
                "files_backed_up": files_backed_up,
            },
        }

        logger.info("=" * 80)
        logger.info("PROCESSING COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Total files processed: {len(excel_files)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Total sheets created: {total_sheets_created}")
        logger.info(f"Files backed up: {files_backed_up}")

        return summary

    def generate_report(self, results: Dict, output_path: Path = None) -> str:
        """
        Generate a detailed processing report.

        Args:
            results: Processing results
            output_path: Optional path to save the report

        Returns:
            Report as string
        """
        report_lines = [
            "=" * 80,
            "EXCEL SHEET SPLITTING REPORT",
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
            f"  - Total sheets created: {results['summary']['total_sheets_created']}",
            f"  - Files backed up: {results['summary']['files_backed_up']}",
            "",
            "DETAILED RESULTS:",
            "-" * 40,
        ]

        for i, result in enumerate(results["results"], 1):
            status = "✓ SUCCESS" if result["success"] else "✗ FAILED"

            report_lines.extend(
                [
                    f"{i}. {result['original_file']} - {status}",
                    f"   Sheets created: {len(result['split_files'])}",
                    f"   Backup location: {result['backup_location'] or 'N/A'}",
                ]
            )

            if result["split_files"]:
                report_lines.append("   Split files:")
                for sheet_name, file_path in result["split_files"].items():
                    report_lines.append(f"     - {sheet_name}: {file_path}")

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
    """Main function to run the Excel sheet splitter."""
    print("=" * 80)
    print("EXCEL SHEET SPLITTER")
    print("=" * 80)
    print()

    # Define directories
    source_dir = Path("data/documents")
    split_dir = Path("data/split_sheets")
    backup_dir = Path("data/original_sheets")
    log_dir = Path("logs")

    print(f"Source directory: {source_dir}")
    print(f"Split directory: {split_dir}")
    print(f"Backup directory: {backup_dir}")
    print(f"Log directory: {log_dir}")
    print()

    if not source_dir.exists():
        print(f"❌ Error: Source directory '{source_dir}' does not exist!")
        return 1

    try:
        # Initialize splitter
        print("Initializing Excel sheet splitter...")
        splitter = ExcelSheetSplitter(source_dir, split_dir, backup_dir, log_dir)

        # Process all files
        print("Starting processing...")
        print("-" * 40)

        results = splitter.process_all_files()

        # Generate report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = log_dir / f"sheet_splitting_report_{timestamp}.txt"
        report = splitter.generate_report(results, report_path)

        print()
        print("=" * 80)
        print("PROCESSING COMPLETED")
        print("=" * 80)
        print(report)

        # Summary
        if results["successful"] > 0:
            print("✅ Processing completed successfully!")
            print(f"📁 Check split files in: {split_dir}")
            print(f"📁 Original files backed up to: {backup_dir}")
            print(f"📋 Full report saved to: {report_path}")
        else:
            print("⚠ No files were processed successfully.")
            print("Check the logs for error details.")

        return 0

    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        return 1


if __name__ == "__main__":
    import sys

    exit_code = main()
    sys.exit(exit_code)
