"""
Table extraction module for Excel-converted HTML files.

This module identifies and extracts individual tables from Excel sheets,
saving each table as a separate HTML file with proper formatting.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from bs4 import BeautifulSoup

# Set up logging
logger = logging.getLogger(__name__)


class TableExtractor:
    """Extracts and saves individual tables from Excel sheets."""

    def __init__(self):
        """Initialize the table extractor."""
        self.min_table_rows = 2  # Minimum rows to consider as a table
        self.min_table_cols = 2  # Minimum columns to consider as a table

    def extract_tables_from_html(
        self, html_content: str, sheet_name: str
    ) -> List[Dict]:
        """
        Extract individual tables from HTML content.

        Args:
            html_content: HTML content containing tables
            sheet_name: Name of the original sheet

        Returns:
            List of dictionaries containing table information
        """
        try:
            soup = BeautifulSoup(html_content, "lxml")
            tables = []

            # Find the main table in the HTML
            main_table = soup.find("table")
            if not main_table:
                logger.warning(
                    f"No table found in HTML content for sheet: {sheet_name}"
                )
                return tables

            # Extract table data
            rows = main_table.find_all("tr")
            if len(rows) < self.min_table_rows:
                logger.warning(
                    f"Table too small in sheet {sheet_name}: {len(rows)} rows"
                )
                return tables

            # Convert table to pandas DataFrame for easier manipulation
            table_data = []
            for row in rows:
                cells = row.find_all(["td", "th"])
                row_data = [cell.get_text(strip=True) for cell in cells]
                table_data.append(row_data)

            if not table_data:
                return tables

            # Identify distinct tables within the sheet based on empty rows/columns
            detected_tables = self._identify_separate_tables(table_data)

            # If no separate tables detected, treat the whole thing as one table
            if not detected_tables:
                detected_tables = [
                    (0, len(table_data), 0, len(table_data[0]) if table_data else 0)
                ]

            # Process each detected table
            for i, (start_row, end_row, start_col, end_col) in enumerate(
                detected_tables
            ):
                table_subset = self._extract_table_subset(
                    table_data, start_row, end_row, start_col, end_col
                )

                if self._is_valid_table(table_subset):
                    table_info = {
                        "table_number": i,
                        "data": table_subset,
                        "start_row": start_row,
                        "end_row": end_row,
                        "start_col": start_col,
                        "end_col": end_col,
                        "sheet_name": sheet_name,
                    }
                    tables.append(table_info)

            logger.info(f"Extracted {len(tables)} tables from sheet: {sheet_name}")
            return tables

        except Exception as e:
            logger.error(f"Error extracting tables from HTML: {str(e)}")
            return []

    def _identify_separate_tables(
        self, table_data: List[List[str]]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Identify separate tables within the data based on empty rows/columns.

        Args:
            table_data: 2D list of table cell data

        Returns:
            List of tuples (start_row, end_row, start_col, end_col) for each table
        """
        if not table_data:
            return []

        tables = []

        # For now, implement a simple approach - treat the whole sheet as one table
        # In a more sophisticated implementation, we could detect empty rows/columns
        # that separate distinct tables

        max_cols = max(len(row) for row in table_data) if table_data else 0
        non_empty_rows = []

        # Find rows with at least some content
        for i, row in enumerate(table_data):
            if any(cell.strip() for cell in row):
                non_empty_rows.append(i)

        if non_empty_rows:
            # Find continuous blocks of non-empty rows
            current_start = non_empty_rows[0]
            current_end = non_empty_rows[0]

            for row_idx in non_empty_rows[1:]:
                if (
                    row_idx == current_end + 1 or row_idx <= current_end + 2
                ):  # Allow 1-2 empty rows
                    current_end = row_idx
                else:
                    # Found a gap, create a table for the current block
                    if current_end - current_start >= self.min_table_rows - 1:
                        tables.append((current_start, current_end + 1, 0, max_cols))
                    current_start = row_idx
                    current_end = row_idx

            # Add the final table
            if current_end - current_start >= self.min_table_rows - 1:
                tables.append((current_start, current_end + 1, 0, max_cols))

        return tables

    def _extract_table_subset(
        self,
        table_data: List[List[str]],
        start_row: int,
        end_row: int,
        start_col: int,
        end_col: int,
    ) -> List[List[str]]:
        """
        Extract a subset of the table data.

        Args:
            table_data: Original table data
            start_row: Starting row index
            end_row: Ending row index (exclusive)
            start_col: Starting column index
            end_col: Ending column index (exclusive)

        Returns:
            Subset of table data
        """
        subset = []
        for i in range(start_row, min(end_row, len(table_data))):
            row = table_data[i]
            subset_row = row[start_col : min(end_col, len(row))]
            # Pad row if necessary
            while len(subset_row) < (end_col - start_col):
                subset_row.append("")
            subset.append(subset_row)
        return subset

    def _is_valid_table(self, table_data: List[List[str]]) -> bool:
        """
        Check if table data represents a valid table.

        Args:
            table_data: Table data to validate

        Returns:
            True if valid table, False otherwise
        """
        if len(table_data) < self.min_table_rows:
            return False

        # Check if there's at least some content
        has_content = False
        for row in table_data:
            if any(cell.strip() for cell in row):
                has_content = True
                break

        return has_content

    def save_table_as_html(self, table_info: Dict, output_path: Path) -> Path:
        """
        Save a table as an HTML file.

        Args:
            table_info: Dictionary containing table information
            output_path: Path where to save the HTML file

        Returns:
            Path to the saved HTML file
        """
        try:
            table_data = table_info["data"]
            table_number = table_info["table_number"]
            sheet_name = table_info["sheet_name"]

            # Create DataFrame from table data
            df = pd.DataFrame(table_data)

            # Try to identify headers (first row with content)
            header_row_idx = 0
            for i, row in enumerate(table_data):
                if any(cell.strip() for cell in row):
                    header_row_idx = i
                    break

            # Set headers if we found a good header row
            if header_row_idx < len(table_data):
                headers = table_data[header_row_idx]
                df.columns = [
                    f"Col_{i}" if not header.strip() else header
                    for i, header in enumerate(headers)
                ]
                df = df.iloc[header_row_idx + 1 :]  # Remove header row from data

            # Convert to HTML
            table_html = df.to_html(
                index=False,
                table_id=f"table_{table_number}",
                classes="extracted-table",
                escape=False,
                na_rep="",
            )

            # Create complete HTML document
            html_content = self._create_table_html_document(
                table_html, sheet_name, table_number
            )

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write HTML file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            logger.info(f"Saved table {table_number} to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error saving table as HTML: {str(e)}")
            raise

    def _create_table_html_document(
        self, table_html: str, sheet_name: str, table_number: int
    ) -> str:
        """
        Create a complete HTML document for a table.

        Args:
            table_html: HTML content of the table
            sheet_name: Name of the source sheet
            table_number: Number of the table

        Returns:
            Complete HTML document
        """
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Table {table_number} from Sheet: {sheet_name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f9f9f9;
        }}
        .extracted-table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .extracted-table th, .extracted-table td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
            vertical-align: top;
        }}
        .extracted-table th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        .extracted-table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .table-title {{
            color: #333;
            margin-bottom: 10px;
        }}
        .table-meta {{
            color: #666;
            font-size: 0.9em;
            margin-bottom: 15px;
        }}
    </style>
</head>
<body>
    <h1 class="table-title">Table {table_number}</h1>
    <div class="table-meta">
        <p><strong>Source Sheet:</strong> {sheet_name}</p>
        <p><strong>Table Number:</strong> {table_number}</p>
    </div>
    {table_html}
</body>
</html>
        """
        return html_template

    def extract_tables_from_file(
        self, html_file_path: Path, output_dir: Path, original_file_name: str
    ) -> List[Path]:
        """
        Extract tables from an HTML file and save each as separate HTML files.

        Args:
            html_file_path: Path to HTML file
            output_dir: Directory to save table HTML files
            original_file_name: Name of the original Excel file

        Returns:
            List of paths to saved table HTML files
        """
        try:
            # Read HTML file
            with open(html_file_path, "r", encoding="utf-8") as f:
                html_content = f.read()

            # Extract sheet name from file path
            sheet_name = html_file_path.stem
            if sheet_name.startswith(f"{Path(original_file_name).stem}_"):
                sheet_name = sheet_name[len(f"{Path(original_file_name).stem}_") :]

            # Extract tables
            tables = self.extract_tables_from_html(html_content, sheet_name)

            saved_paths = []
            for table_info in tables:
                table_number = table_info["table_number"]

                # Create output filename following the required format
                # data/processed_sheet/<folder name>/<sheet_name>_table_<table_number>.html
                table_filename = f"{sheet_name}_table_{table_number}.html"
                table_path = output_dir / table_filename

                # Save table
                saved_path = self.save_table_as_html(table_info, table_path)
                saved_paths.append(saved_path)

            return saved_paths

        except Exception as e:
            logger.error(
                f"Error extracting tables from file {html_file_path}: {str(e)}"
            )
            return []

    def batch_extract_tables(
        self, html_files: Dict[str, Dict[str, str]], output_base_dir: Path
    ) -> Dict[str, List[Path]]:
        """
        Extract tables from multiple HTML files.

        Args:
            html_files: Dictionary mapping original file paths to sheet HTML file paths
            output_base_dir: Base directory for table outputs

        Returns:
            Dictionary mapping original file paths to table HTML file paths
        """
        logger.info(f"Starting batch table extraction for {len(html_files)} files")

        results = {}

        for original_path, sheet_paths in html_files.items():
            try:
                original_file_path = Path(original_path)
                relative_path = original_file_path.parent.name  # Get folder name
                output_dir = output_base_dir / relative_path

                all_table_paths = []

                # Process each sheet HTML file
                for sheet_name, html_path in sheet_paths.items():
                    html_file_path = Path(html_path)
                    table_paths = self.extract_tables_from_file(
                        html_file_path, output_dir, original_file_path.name
                    )
                    all_table_paths.extend(table_paths)

                results[original_path] = all_table_paths

            except Exception as e:
                logger.error(f"Failed to extract tables from {original_path}: {str(e)}")
                results[original_path] = []

        logger.info(
            f"Batch table extraction completed. Processed {len(results)} files."
        )
        return results


def setup_logging(log_file: Optional[Path] = None):
    """
    Set up logging for the table extractor.

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
