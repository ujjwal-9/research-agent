"""
Knowledge extraction module using OpenAI GPT-4.1.

This module analyzes HTML content from converted Excel files and extracts
meaningful knowledge that can be stored as text files.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from bs4 import BeautifulSoup
import openai
from openai import OpenAI
import time

# Set up logging
logger = logging.getLogger(__name__)


class KnowledgeExtractor:
    """Extracts knowledge from HTML content using OpenAI GPT-4.1."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the knowledge extractor.

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

        # Knowledge extraction prompt template
        self.knowledge_prompt = """
You are an expert knowledge analyst tasked with extracting meaningful insights and knowledge from Excel spreadsheet content that has been converted to HTML.

Your task is to:
1. Analyze the provided HTML content from an Excel sheet
2. Identify and extract key information, insights, data patterns, and knowledge
3. Focus on extracting actionable information, trends, relationships, and important findings
4. Ignore purely formatting elements and focus on substantive content
5. Structure the knowledge in a clear, readable format

Please extract knowledge in the following format:
- **Summary**: Brief overview of what this data represents
- **Key Insights**: Main findings, patterns, or important information
- **Data Points**: Specific important values, metrics, or measurements
- **Relationships**: Connections between different data elements
- **Actionable Information**: Any recommendations, next steps, or decisions that can be derived

HTML Content to analyze:
{html_content}

Extract meaningful knowledge from this content:"""

    def extract_knowledge_from_html(self, html_content: str, context: str = "") -> str:
        """
        Extract knowledge from HTML content using OpenAI.

        Args:
            html_content: HTML content to analyze
            context: Additional context about the content

        Returns:
            Extracted knowledge as formatted text
        """
        try:
            # Clean and prepare HTML content
            cleaned_html = self._clean_html_content(html_content)

            # Prepare the prompt
            prompt = self.knowledge_prompt.format(html_content=cleaned_html)
            if context:
                prompt += f"\n\nAdditional context: {context}"

            logger.info("Sending request to OpenAI for knowledge extraction")

            # Make API call to OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert data analyst specializing in extracting meaningful insights from spreadsheet data. Focus on identifying patterns, trends, key metrics, and actionable information.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2000,
                temperature=0.3,
            )

            knowledge = response.choices[0].message.content
            logger.info("Successfully extracted knowledge from HTML content")

            return knowledge

        except Exception as e:
            logger.error(f"Error extracting knowledge from HTML: {str(e)}")
            # Return a fallback knowledge extraction
            return self._fallback_knowledge_extraction(html_content)

    def _clean_html_content(self, html_content: str) -> str:
        """
        Clean HTML content for better processing.

        Args:
            html_content: Raw HTML content

        Returns:
            Cleaned HTML content
        """
        try:
            soup = BeautifulSoup(html_content, "lxml")

            # Remove style and script tags
            for tag in soup(["style", "script"]):
                tag.decompose()

            # Get the table content specifically
            table = soup.find("table")
            if table:
                # Convert table to a more readable format
                rows = []
                for tr in table.find_all("tr"):
                    cells = [
                        td.get_text(strip=True) for td in tr.find_all(["td", "th"])
                    ]
                    if any(cell for cell in cells):  # Skip empty rows
                        rows.append(" | ".join(cells))

                return "\n".join(rows)
            else:
                # If no table found, return clean text
                return soup.get_text(separator="\n", strip=True)

        except Exception as e:
            logger.warning(f"Error cleaning HTML content: {str(e)}")
            # Return raw content if cleaning fails
            return html_content

    def _fallback_knowledge_extraction(self, html_content: str) -> str:
        """
        Fallback knowledge extraction without AI.

        Args:
            html_content: HTML content to analyze

        Returns:
            Basic knowledge extraction
        """
        try:
            soup = BeautifulSoup(html_content, "lxml")

            # Extract basic information
            title = soup.find("title")
            title_text = title.get_text() if title else "Unknown Sheet"

            table = soup.find("table")
            if table:
                rows = table.find_all("tr")
                row_count = len(rows)

                # Try to identify headers
                headers = []
                first_row = rows[0] if rows else None
                if first_row:
                    headers = [
                        th.get_text(strip=True)
                        for th in first_row.find_all(["th", "td"])
                    ]

                knowledge = f"""
**Summary**: {title_text}
- Data table with {row_count} rows
- Columns: {', '.join(headers) if headers else 'Not clearly identified'}

**Key Insights**: 
- This appears to be a structured data table
- Manual review recommended for detailed analysis

**Data Points**: 
- Total rows: {row_count}
- Estimated columns: {len(headers)}

**Note**: This is a fallback extraction. For detailed insights, please ensure OpenAI API is properly configured.
"""
                return knowledge.strip()
            else:
                return f"**Summary**: {title_text}\n**Note**: No structured data table found in this content."

        except Exception as e:
            logger.error(f"Error in fallback knowledge extraction: {str(e)}")
            return "**Error**: Unable to extract knowledge from this content."

    def extract_knowledge_from_file(
        self, html_file_path: Path, output_dir: Path, table_number: int = 0
    ) -> Path:
        """
        Extract knowledge from an HTML file and save as text.

        Args:
            html_file_path: Path to HTML file
            output_dir: Directory to save knowledge file
            table_number: Table number for filename

        Returns:
            Path to the saved knowledge file
        """
        try:
            # Read HTML file
            with open(html_file_path, "r", encoding="utf-8") as f:
                html_content = f.read()

            # Extract knowledge
            knowledge = self.extract_knowledge_from_html(
                html_content, f"Source file: {html_file_path.name}"
            )

            # Create output filename
            base_name = html_file_path.stem
            if base_name.endswith(".html"):
                base_name = base_name[:-5]  # Remove .html extension

            knowledge_filename = f"{base_name}__knowledge_table_{table_number}.txt"
            knowledge_path = output_dir / knowledge_filename

            # Ensure output directory exists
            knowledge_path.parent.mkdir(parents=True, exist_ok=True)

            # Save knowledge to file
            with open(knowledge_path, "w", encoding="utf-8") as f:
                f.write(knowledge)

            logger.info(f"Saved knowledge to: {knowledge_path}")
            return knowledge_path

        except Exception as e:
            logger.error(f"Error processing file {html_file_path}: {str(e)}")
            raise

    def batch_extract_knowledge(
        self, html_files: Dict[str, str], output_base_dir: Path
    ) -> Dict[str, List[Path]]:
        """
        Extract knowledge from multiple HTML files.

        Args:
            html_files: Dictionary mapping original file paths to HTML file paths
            output_base_dir: Base directory for knowledge outputs

        Returns:
            Dictionary mapping original file paths to knowledge file paths
        """
        logger.info(f"Starting batch knowledge extraction for {len(html_files)} files")

        results = {}

        for original_path, html_path in html_files.items():
            try:
                html_file_path = Path(html_path)

                # Determine output directory maintaining folder structure
                original_file_path = Path(original_path)
                relative_path = original_file_path.parent.name  # Get folder name
                output_dir = output_base_dir / relative_path

                # Extract knowledge
                knowledge_path = self.extract_knowledge_from_file(
                    html_file_path, output_dir, 0
                )

                if original_path not in results:
                    results[original_path] = []
                results[original_path].append(knowledge_path)

                # Add a small delay to avoid rate limiting
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Failed to extract knowledge from {html_path}: {str(e)}")
                if original_path not in results:
                    results[original_path] = []

        logger.info(
            f"Batch knowledge extraction completed. Processed {len(results)} files."
        )
        return results


def setup_logging(log_file: Optional[Path] = None):
    """
    Set up logging for the knowledge extractor.

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
