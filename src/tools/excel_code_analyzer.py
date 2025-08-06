"""
Excel Code Analyzer tool for running code analysis on Excel files from retrieved chunks.

This tool identifies Excel files from Qdrant search results and uses the code interpreter
to perform analysis on the relevant Excel files based on user queries.
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from collections import defaultdict

from .document_retriever import DocumentRetriever, SearchResult
from src.code_interpreter.excel_analyzer import ExcelAnalyzer
from src.code_interpreter.config import CodeInterpreterConfig


@dataclass
class ExcelAnalysisResult:
    """Results from Excel code analysis."""

    query: str
    excel_files_analyzed: List[str]
    analysis_result: Dict[str, Any]
    source_chunks: List[SearchResult]
    success: bool
    error_message: Optional[str] = None


class ExcelCodeAnalyzer:
    """Tool for running code analysis on Excel files identified from search results."""

    def __init__(
        self,
        document_retriever: Optional[DocumentRetriever] = None,
        config: Optional[CodeInterpreterConfig] = None,
    ):
        """Initialize the Excel Code Analyzer.

        Args:
            document_retriever: DocumentRetriever instance to use for searches
            config: Configuration for the code interpreter
        """
        self.logger = logging.getLogger(__name__)
        self.document_retriever = document_retriever or DocumentRetriever()
        self.config = config or CodeInterpreterConfig()
        self.excel_analyzer = ExcelAnalyzer(self.config)

        # Excel file extensions
        self.excel_extensions = {".xlsx", ".xls", ".xlsm", ".xlsb", ".csv"}

    def analyze_excel_from_search_results(
        self,
        search_results: List[SearchResult],
        user_query: str,
        additional_context: Optional[str] = None,
    ) -> ExcelAnalysisResult:
        """Analyze Excel files identified from search results.

        Args:
            search_results: List of search results from document retrieval
            user_query: User's query for the Excel analysis
            additional_context: Additional context for the analysis

        Returns:
            ExcelAnalysisResult containing the analysis results
        """
        try:
            # Identify Excel files from search results
            excel_files = self._identify_excel_files_from_results(search_results)

            if not excel_files:
                return ExcelAnalysisResult(
                    query=user_query,
                    excel_files_analyzed=[],
                    analysis_result={
                        "success": False,
                        "error": "No Excel files found in search results",
                    },
                    source_chunks=search_results,
                    success=False,
                    error_message="No Excel files found in search results",
                )

            self.logger.info(
                f"🔍 Identified {len(excel_files)} Excel files from search results"
            )
            for file_path in excel_files:
                self.logger.info(f"  - {file_path}")

            # Build enhanced context from search results
            enhanced_context = self._build_enhanced_context(
                search_results, additional_context
            )

            # Prepare files for Docker analysis
            docker_accessible_files = self._prepare_files_for_docker(list(excel_files))

            # Run the Excel analysis
            analysis_result = self.excel_analyzer.analyze_excel_files(
                excel_files=docker_accessible_files,
                user_query=user_query,
                additional_context=enhanced_context,
            )

            return ExcelAnalysisResult(
                query=user_query,
                excel_files_analyzed=list(excel_files),
                analysis_result=analysis_result,
                source_chunks=search_results,
                success=analysis_result.get("success", False),
                error_message=analysis_result.get("error")
                if not analysis_result.get("success", False)
                else None,
            )

        except Exception as e:
            self.logger.error(f"Excel analysis failed: {str(e)}")
            return ExcelAnalysisResult(
                query=user_query,
                excel_files_analyzed=[],
                analysis_result={"success": False, "error": str(e)},
                source_chunks=search_results,
                success=False,
                error_message=str(e),
            )

    def analyze_excel_from_query(
        self,
        query: str,
        user_query: str,
        limit: int = 50,
        score_threshold: float = 0.3,
        additional_context: Optional[str] = None,
    ) -> ExcelAnalysisResult:
        """Search for relevant chunks and analyze Excel files.

        Args:
            query: Search query to find relevant documents
            user_query: User's query for the Excel analysis
            limit: Maximum number of search results
            score_threshold: Minimum similarity score for results
            additional_context: Additional context for the analysis

        Returns:
            ExcelAnalysisResult containing the analysis results
        """
        try:
            # Search for relevant documents
            search_results = self.document_retriever.search_documents(
                query=query,
                limit=limit,
                score_threshold=score_threshold,
                include_context=True,
            )

            if not search_results:
                return ExcelAnalysisResult(
                    query=user_query,
                    excel_files_analyzed=[],
                    analysis_result={
                        "success": False,
                        "error": "No relevant documents found",
                    },
                    source_chunks=[],
                    success=False,
                    error_message="No relevant documents found",
                )

            return self.analyze_excel_from_search_results(
                search_results=search_results,
                user_query=user_query,
                additional_context=additional_context,
            )

        except Exception as e:
            self.logger.error(f"Excel analysis from query failed: {str(e)}")
            return ExcelAnalysisResult(
                query=user_query,
                excel_files_analyzed=[],
                analysis_result={"success": False, "error": str(e)},
                source_chunks=[],
                success=False,
                error_message=str(e),
            )

    def _identify_excel_files_from_results(
        self, search_results: List[SearchResult]
    ) -> Set[str]:
        """Identify Excel files from search results.

        Args:
            search_results: List of search results

        Returns:
            Set of Excel file paths
        """
        excel_files = set()

        for result in search_results:
            # Check the source field (document name)
            source = result.source
            if source and self._is_excel_file(source):
                # Try to find the actual file path
                file_path = self._resolve_excel_file_path(source)
                if file_path:
                    excel_files.add(file_path)

            # Also check metadata for file paths
            metadata = result.metadata or {}

            # Check common metadata fields that might contain file paths
            potential_paths = [
                metadata.get("document_name"),
                metadata.get("file_name"),
                metadata.get("file_path"),
                metadata.get("source_file"),
                metadata.get("original_filename"),
            ]

            for path in potential_paths:
                if path and self._is_excel_file(path):
                    file_path = self._resolve_excel_file_path(path)
                    if file_path:
                        excel_files.add(file_path)

        return excel_files

    def _is_excel_file(self, filename: str) -> bool:
        """Check if a filename represents an Excel file.

        Args:
            filename: Filename to check

        Returns:
            True if the file is an Excel file
        """
        if not filename:
            return False

        # Get file extension
        file_path = Path(filename)
        extension = file_path.suffix.lower()

        return extension in self.excel_extensions

    def _resolve_excel_file_path(self, filename: str) -> Optional[str]:
        """Resolve the actual file path for an Excel file.

        Args:
            filename: Filename or path from metadata

        Returns:
            Resolved file path if it exists, None otherwise
        """
        if not filename:
            return None

        # If it's already an absolute path and exists, return it
        if os.path.isabs(filename) and os.path.exists(filename):
            return filename

        # Try common directories where Excel files might be stored
        search_directories = [
            ".",  # Current directory
            "outputs",
            "data",
            "inputs",
            "files",
            "../data",
            "../outputs",
            "../inputs",
            # Add the specific path structure from your data
            "data/split_sheets",
            "data/split_sheets/11. Financial Model",
            "data/split_sheets/11. Financial Model/Archive",
        ]

        for directory in search_directories:
            # Try the filename as-is in each directory
            full_path = os.path.join(directory, filename)
            if os.path.exists(full_path):
                return os.path.abspath(full_path)

            # Try just the basename in each directory
            basename = os.path.basename(filename)
            full_path = os.path.join(directory, basename)
            if os.path.exists(full_path):
                return os.path.abspath(full_path)

        # If we can't find it, try searching recursively from current directory
        try:
            basename = os.path.basename(filename)
            for root, dirs, files in os.walk("."):
                if basename in files:
                    return os.path.abspath(os.path.join(root, basename))
        except Exception as e:
            self.logger.debug(f"Error during recursive search: {e}")

        self.logger.warning(f"Could not resolve file path for: {filename}")
        return None

    def _build_enhanced_context(
        self,
        search_results: List[SearchResult],
        additional_context: Optional[str] = None,
    ) -> str:
        """Build enhanced context from search results.

        Args:
            search_results: List of search results
            additional_context: Additional context provided by user

        Returns:
            Enhanced context string
        """
        context_parts = []

        if additional_context:
            context_parts.append(f"User Context: {additional_context}")

        # Group results by source
        source_chunks = defaultdict(list)
        for result in search_results:
            source_chunks[result.source].append(result)

        # Build context from chunks
        if source_chunks:
            context_parts.append("\nRelevant Context from Retrieved Documents:")

            for source, chunks in source_chunks.items():
                if self._is_excel_file(source):
                    context_parts.append(f"\nFrom Excel file '{source}':")
                    for i, chunk in enumerate(
                        chunks[:3]
                    ):  # Limit to top 3 chunks per source
                        context_parts.append(
                            f"  Chunk {i + 1}: {chunk.content[:200]}..."
                        )

        return "\n".join(context_parts)

    def _prepare_files_for_docker(self, excel_files: List[str]) -> List[str]:
        """Prepare Excel files for Docker analysis by copying them to an accessible location.

        Args:
            excel_files: List of Excel file paths

        Returns:
            List of Docker-accessible file paths
        """
        import shutil

        docker_accessible_files = []
        docker_temp_dir = "outputs/temp_excel_analysis"

        # Create temp directory for Docker-accessible files
        os.makedirs(docker_temp_dir, exist_ok=True)

        for file_path in excel_files:
            try:
                if os.path.exists(file_path):
                    # Copy file to Docker-accessible location
                    filename = os.path.basename(file_path)
                    # Sanitize filename for Docker
                    safe_filename = re.sub(r"[^\w\-_\.]", "_", filename)
                    docker_path = os.path.join(docker_temp_dir, safe_filename)

                    shutil.copy2(file_path, docker_path)
                    docker_accessible_files.append(docker_path)
                    self.logger.info(
                        f"📁 Copied for Docker analysis: {file_path} -> {docker_path}"
                    )
                else:
                    self.logger.warning(f"⚠️ File not found, skipping: {file_path}")
            except Exception as e:
                self.logger.error(
                    f"❌ Failed to prepare file for Docker: {file_path} - {e}"
                )

        return docker_accessible_files

    def get_excel_files_summary(
        self, search_results: List[SearchResult]
    ) -> Dict[str, Any]:
        """Get a summary of Excel files found in search results.

        Args:
            search_results: List of search results

        Returns:
            Dictionary containing Excel files summary
        """
        excel_files = self._identify_excel_files_from_results(search_results)

        if not excel_files:
            return {
                "excel_files_found": 0,
                "files": [],
                "message": "No Excel files found in search results",
            }

        # Get file summaries
        summary_result = self.excel_analyzer.get_file_summary(list(excel_files))

        return {
            "excel_files_found": len(excel_files),
            "files": list(excel_files),
            "summary": summary_result,
            "message": f"Found {len(excel_files)} Excel files",
        }
