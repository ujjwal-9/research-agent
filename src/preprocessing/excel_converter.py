"""
Excel to HTML converter module.

This module provides functionality to convert Excel files to HTML format,
preserving tables and formatting for further processing.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from bs4 import BeautifulSoup
import os
from .header_detector import HeaderDetector

# Set up logging
logger = logging.getLogger(__name__)


class ExcelToHTMLConverter:
    """Converts Excel files to HTML format."""

    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the converter."""
        self.supported_formats = [".xlsx", ".xls"]

        # Initialize header detector if API key is available
        try:
            self.header_detector = (
                HeaderDetector(api_key=openai_api_key) if openai_api_key else None
            )
            if self.header_detector:
                logger.info(
                    "Header detector initialized - will use AI for header detection"
                )
            else:
                logger.info("No OpenAI API key provided - using basic header detection")
        except Exception as e:
            logger.warning(f"Could not initialize header detector: {str(e)}")
            self.header_detector = None

    def is_supported_file(self, file_path: Path) -> bool:
        """
        Check if the file format is supported.

        Args:
            file_path: Path to the file

        Returns:
            True if file format is supported, False otherwise
        """
        return file_path.suffix.lower() in self.supported_formats

    def convert_excel_to_html(
        self, excel_path: Path, output_dir: Path
    ) -> Dict[str, str]:
        """
        Convert an Excel file to HTML format.

        Args:
            excel_path: Path to the Excel file
            output_dir: Directory to save HTML files

        Returns:
            Dictionary mapping sheet names to HTML file paths
        """
        if not self.is_supported_file(excel_path):
            raise ValueError(f"Unsupported file format: {excel_path.suffix}")

        logger.info(f"Converting Excel file: {excel_path}")

        try:
            # Read all sheets from the Excel file
            excel_file = pd.ExcelFile(excel_path)
            sheet_html_paths = {}

            for sheet_name in excel_file.sheet_names:
                logger.info(f"Processing sheet: {sheet_name}")

                # Read the sheet without assuming headers
                df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)

                # Use header detector to properly set headers if available
                if self.header_detector and not df.empty:
                    try:
                        df = self.header_detector.process_dataframe_with_detected_headers(
                            df, sheet_name
                        )
                    except Exception as e:
                        logger.warning(
                            f"Header detection failed for sheet {sheet_name}: {str(e)}, using original data"
                        )

                # Convert to HTML
                html_content = self._dataframe_to_html(df, sheet_name)

                # Create output file path
                safe_sheet_name = self._sanitize_filename(sheet_name)
                html_filename = f"{excel_path.stem}_{safe_sheet_name}.html"
                html_path = output_dir / html_filename

                # Ensure output directory exists
                html_path.parent.mkdir(parents=True, exist_ok=True)

                # Write HTML file
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(html_content)

                sheet_html_paths[sheet_name] = str(html_path)
                logger.info(f"Saved HTML for sheet '{sheet_name}' to: {html_path}")

            return sheet_html_paths

        except Exception as e:
            logger.error(f"Error converting Excel file {excel_path}: {str(e)}")
            raise

    def _dataframe_to_html(self, df: pd.DataFrame, sheet_name: str) -> str:
        """
        Convert DataFrame to well-formatted HTML.

        Args:
            df: DataFrame to convert
            sheet_name: Name of the sheet

        Returns:
            HTML string with proper formatting
        """
        # Convert DataFrame to HTML
        # If we have proper column names (not just numbers), include them as headers
        has_proper_headers = not all(
            str(col).startswith("Column_") or str(col).isdigit() for col in df.columns
        )

        df_html = df.to_html(
            index=False,
            table_id=f"sheet_{self._sanitize_filename(sheet_name)}",
            classes="excel-table",
            escape=False,
            na_rep="",
            # Note: pandas to_html doesn't have 'thead' parameter
            # Headers are automatically included based on column names
        )

        # Create complete HTML document
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sheet: {sheet_name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f9f9f9;
        }}
        .excel-table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .excel-table th, .excel-table td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
            vertical-align: top;
        }}
        .excel-table th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        .excel-table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .sheet-title {{
            color: #333;
            margin-bottom: 10px;
        }}
    </style>
</head>
<body>
    <h1 class="sheet-title">Sheet: {sheet_name}</h1>
    {df_html}
</body>
</html>
        """

        return html_template

    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to be filesystem-safe.

        Args:
            filename: Original filename

        Returns:
            Sanitized filename
        """
        # Replace invalid characters with underscores
        invalid_chars = '<>:"/\\|?*'
        sanitized = filename
        for char in invalid_chars:
            sanitized = sanitized.replace(char, "_")

        # Remove multiple consecutive underscores and strip
        sanitized = "_".join(filter(None, sanitized.split("_")))

        return sanitized

    def batch_convert_directory(
        self, input_dir: Path, output_base_dir: Path
    ) -> Dict[str, Dict[str, str]]:
        """
        Convert all Excel files in a directory to HTML.

        Args:
            input_dir: Directory containing Excel files
            output_base_dir: Base directory for HTML outputs

        Returns:
            Dictionary mapping file paths to sheet HTML paths
        """
        logger.info(f"Starting batch conversion of directory: {input_dir}")

        results = {}

        # Find all Excel files recursively
        for excel_path in input_dir.rglob("*"):
            if excel_path.is_file() and self.is_supported_file(excel_path):
                try:
                    # Create output directory maintaining folder structure
                    relative_path = excel_path.relative_to(input_dir)
                    output_dir = output_base_dir / relative_path.parent

                    # Convert the Excel file
                    sheet_paths = self.convert_excel_to_html(excel_path, output_dir)
                    results[str(excel_path)] = sheet_paths

                except Exception as e:
                    logger.error(f"Failed to convert {excel_path}: {str(e)}")
                    results[str(excel_path)] = {}

        logger.info(f"Batch conversion completed. Processed {len(results)} files.")
        return results


def setup_logging(log_file: Optional[Path] = None):
    """
    Set up logging for the excel converter.

    Args:
        log_file: Optional path to log file
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)
