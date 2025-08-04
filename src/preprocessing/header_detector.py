"""
Header detection module using OpenAI GPT-4.

This module analyzes the first few rows of Excel data to intelligently
identify which row contains the column headers.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
from openai import OpenAI

# Set up logging
logger = logging.getLogger(__name__)


class HeaderDetector:
    """Detects header rows in Excel data using OpenAI."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the header detector.

        Args:
            api_key: OpenAI API key. If not provided, will use OPENAI_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        self.client = OpenAI(api_key=self.api_key)

        # Get model from environment variable, with fallback
        self.model = os.getenv("OPENAI_EXCEL_PROCESSING_MODEL", "gpt-4-turbo-preview")

        # Get number of rows to check from environment variable, with fallback
        try:
            self.header_check_rows = int(os.getenv("HEADER_CHECK_ROWS", "10"))
        except ValueError:
            self.header_check_rows = 10
            logger.warning("Invalid HEADER_CHECK_ROWS value, using default: 10")

        logger.info(f"Header detector initialized with model: {self.model}")
        logger.info(f"Will check first {self.header_check_rows} rows for headers")

        # Header detection prompt template
        self.header_prompt = """
You are an expert data analyst specializing in analyzing spreadsheet data structure.

Your task is to analyze the first few rows of an Excel spreadsheet and determine which row contains the column headers.

Rules for identifying headers:
1. Headers are typically descriptive text that describes what each column contains
2. Headers often contain words like "Name", "Date", "Amount", "ID", "Description", etc.
3. Headers usually don't contain purely numeric values or dates
4. Headers are often in the first few rows of the spreadsheet
5. There may be title rows or empty rows before the actual headers
6. Headers should be more descriptive than the data rows that follow

Here are the first {num_rows} rows of the spreadsheet:

{rows_data}

Please analyze these rows and respond with ONLY the row number (0-indexed) that contains the column headers. If no clear headers are found, respond with "-1".

Examples:
- If row 0 contains headers, respond: 0
- If row 2 contains headers, respond: 2  
- If no headers found, respond: -1

Row number containing headers:"""

    def detect_header_row(self, data: List[List[str]], sheet_name: str = "") -> int:
        """
        Detect which row contains the column headers.

        Args:
            data: 2D list representing the spreadsheet data
            sheet_name: Name of the sheet (for context)

        Returns:
            Row index (0-based) containing headers, or -1 if not found
        """
        if not data:
            logger.warning("No data provided for header detection")
            return -1

        try:
            # Limit to the specified number of rows
            rows_to_check = min(len(data), self.header_check_rows)
            sample_data = data[:rows_to_check]

            # Format the data for the prompt
            formatted_rows = []
            for i, row in enumerate(sample_data):
                # Convert row to string, handling empty cells
                row_str = " | ".join(str(cell) if cell else "" for cell in row)
                formatted_rows.append(f"Row {i}: {row_str}")

            rows_data = "\n".join(formatted_rows)

            # Prepare the prompt
            prompt = self.header_prompt.format(
                num_rows=rows_to_check, rows_data=rows_data
            )

            if sheet_name:
                prompt += f"\n\nSheet name for context: {sheet_name}"

            logger.info(
                f"Sending header detection request to OpenAI for sheet: {sheet_name}"
            )

            # Make API call to OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing spreadsheet structure and identifying header rows. Respond only with the row number.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=50,
                temperature=0.1,  # Low temperature for consistent results
            )

            result = response.choices[0].message.content.strip()

            # Parse the result
            try:
                header_row = int(result)
                if 0 <= header_row < len(data):
                    logger.info(
                        f"Detected header row: {header_row} for sheet: {sheet_name}"
                    )
                    return header_row
                elif header_row == -1:
                    logger.info(f"No clear headers detected for sheet: {sheet_name}")
                    return -1
                else:
                    logger.warning(
                        f"Invalid header row {header_row} detected, using fallback"
                    )
                    return self._fallback_header_detection(data)
            except ValueError:
                logger.warning(
                    f"Could not parse OpenAI response: {result}, using fallback"
                )
                return self._fallback_header_detection(data)

        except Exception as e:
            logger.error(f"Error in header detection: {str(e)}")
            return self._fallback_header_detection(data)

    def _fallback_header_detection(self, data: List[List[str]]) -> int:
        """
        Fallback header detection without AI.

        Args:
            data: 2D list representing the spreadsheet data

        Returns:
            Row index containing headers, or 0 as default
        """
        logger.info("Using fallback header detection method")

        if not data:
            return 0

        # Simple heuristic: look for the first row with mostly non-empty, non-numeric content
        for i, row in enumerate(data[: min(5, len(data))]):  # Check first 5 rows max
            if not row:  # Skip empty rows
                continue

            # Count non-empty cells that look like headers (text, not pure numbers)
            header_like_cells = 0
            total_cells = len(row)

            for cell in row:
                cell_str = str(cell).strip()
                if cell_str:  # Non-empty
                    # Check if it looks like a header (not a pure number or date)
                    try:
                        float(cell_str)
                        # If it's a number, it's less likely to be a header
                        continue
                    except ValueError:
                        # If it's not a number, it's more likely to be a header
                        header_like_cells += 1

            # If more than half the cells look like headers, consider this the header row
            if total_cells > 0 and header_like_cells / total_cells > 0.5:
                logger.info(f"Fallback detected header row: {i}")
                return i

        # Default to first row if no clear headers found
        logger.info("Fallback defaulting to row 0")
        return 0

    def get_headers_and_data(
        self, data: List[List[str]], sheet_name: str = ""
    ) -> Tuple[List[str], List[List[str]]]:
        """
        Get headers and data separately based on detected header row.

        Args:
            data: 2D list representing the spreadsheet data
            sheet_name: Name of the sheet

        Returns:
            Tuple of (headers, data_rows)
        """
        if not data:
            return [], []

        header_row_idx = self.detect_header_row(data, sheet_name)

        if header_row_idx == -1 or header_row_idx >= len(data):
            # No headers detected, use column numbers as headers
            max_cols = max(len(row) for row in data) if data else 0
            headers = [f"Column_{i + 1}" for i in range(max_cols)]
            data_rows = data
        else:
            # Use detected header row
            headers = data[header_row_idx]
            data_rows = data[header_row_idx + 1 :]  # Data starts after header row

        # Clean up headers
        headers = [
            str(header).strip() if header else f"Column_{i + 1}"
            for i, header in enumerate(headers)
        ]

        return headers, data_rows

    def process_dataframe_with_detected_headers(
        self, df: pd.DataFrame, sheet_name: str = ""
    ) -> pd.DataFrame:
        """
        Process a DataFrame to properly set headers based on AI detection.

        Args:
            df: Input DataFrame
            sheet_name: Name of the sheet

        Returns:
            DataFrame with properly set headers
        """
        if df.empty:
            return df

        # Convert DataFrame to list format for header detection
        data = df.values.tolist()

        # Detect headers
        headers, data_rows = self.get_headers_and_data(data, sheet_name)

        if not data_rows:
            # If no data rows after header, return original DataFrame
            return df

        # Create new DataFrame with detected headers
        try:
            new_df = pd.DataFrame(data_rows, columns=headers)
            logger.info(
                f"Created DataFrame with {len(headers)} columns and {len(data_rows)} rows for sheet: {sheet_name}"
            )
            return new_df
        except Exception as e:
            logger.warning(
                f"Error creating DataFrame with detected headers: {str(e)}, returning original"
            )
            return df


def setup_logging(log_file: Optional[Path] = None):
    """
    Set up logging for the header detector.

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
