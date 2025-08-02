"""
Excel preprocessing package.

This package provides modules for processing Excel files:
- Converting Excel to HTML
- Extracting knowledge using OpenAI
- Extracting individual tables as HTML files
"""

from .excel_converter import ExcelToHTMLConverter
from .knowledge_extractor import KnowledgeExtractor
from .table_extractor import TableExtractor
from .excel_processor import ExcelProcessor
from .header_detector import HeaderDetector

__all__ = [
    "ExcelToHTMLConverter",
    "KnowledgeExtractor",
    "TableExtractor",
    "ExcelProcessor",
    "HeaderDetector",
]

__version__ = "1.0.0"
