"""Enhanced hybrid retriever integrating structured data analysis with RAG and Knowledge Graph."""

import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from loguru import logger
from dataclasses import dataclass

from src.knowledge_graph.hybrid_retriever import HybridRetriever, HybridResult
from src.tools.hybrid_excel_processor import HybridExcelProcessor, analyze_excel_query
from src.tools.web_search import WebSearchTool


@dataclass
class EnhancedHybridResult:
    """Extended result object combining traditional RAG+KG with structured data analysis."""

    content: str
    metadata: Dict[str, Any]
    source_type: str  # "rag", "graph", "structured_data", "web"
    relevance_score: float
    entities: List[str] = None
    relationships: List[str] = None

    # New fields for structured data
    schema_info: Dict[str, Any] = None
    code_executed: str = None
    data_analysis: Dict[str, Any] = None
    confidence_score: float = None


class EnhancedHybridRetriever:
    """
    Enhanced retriever combining:
    1. Traditional RAG (vector similarity)
    2. Knowledge Graph (entity relationships)
    3. Structured Data Analysis (code execution on Excel/JSON)
    4. Web Search (real-time information)
    """

    def __init__(self):
        self.hybrid_retriever = HybridRetriever()
        self.excel_processor = HybridExcelProcessor()
        self.web_search = WebSearchTool()

    async def enhanced_search(
        self,
        query: str,
        n_results: int = 10,
        include_rag: bool = True,
        include_graph: bool = True,
        include_structured_data: bool = True,
        include_web_search: bool = True,
        excel_files: List[Union[Path, str]] = None,
        json_files: List[Union[Path, str]] = None,
        context_sources: List[str] = None,
    ) -> List[EnhancedHybridResult]:
        """
        Perform enhanced search combining all available data sources.

        Args:
            query: The search query
            n_results: Maximum results to return
            include_rag: Include RAG vector search
            include_graph: Include Knowledge Graph search
            include_structured_data: Include Excel/JSON analysis
            include_web_search: Include web search
            excel_files: List of Excel files to analyze
            json_files: List of JSON files to analyze
            context_sources: Additional context sources (PDFs, etc.)
        """

        logger.info(f"Enhanced search for query: {query}")

        # Collect all search tasks
        search_tasks = []

        # Task 1: Traditional RAG + Knowledge Graph
        if include_rag or include_graph:
            search_tasks.append(
                self._get_traditional_results(
                    query, n_results, include_rag, include_graph
                )
            )

        # Task 2: Structured data analysis
        if include_structured_data and (excel_files or json_files):
            search_tasks.append(
                self._get_structured_data_results(
                    query, excel_files, json_files, context_sources
                )
            )

        # Task 3: Web search
        if include_web_search:
            search_tasks.append(self._get_web_search_results(query, n_results // 2))

        # Execute all searches in parallel
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Combine and rank results
        all_results = []

        # Process traditional RAG+KG results
        result_idx = 0
        if include_rag or include_graph:
            if result_idx < len(search_results) and not isinstance(
                search_results[result_idx], Exception
            ):
                traditional_results = search_results[result_idx]
                all_results.extend(
                    self._convert_traditional_results(traditional_results)
                )
            result_idx += 1

        # Process structured data results
        if include_structured_data and (excel_files or json_files):
            if result_idx < len(search_results) and not isinstance(
                search_results[result_idx], Exception
            ):
                structured_results = search_results[result_idx]
                all_results.extend(self._convert_structured_results(structured_results))
            result_idx += 1

        # Process web search results
        if include_web_search:
            if result_idx < len(search_results) and not isinstance(
                search_results[result_idx], Exception
            ):
                web_results = search_results[result_idx]
                all_results.extend(self._convert_web_results(web_results))

        # Rank and filter results
        ranked_results = self._rank_enhanced_results(all_results, query)

        return ranked_results[:n_results]

    async def _get_traditional_results(
        self, query: str, n_results: int, include_rag: bool, include_graph: bool
    ) -> List[HybridResult]:
        """Get results from traditional RAG + Knowledge Graph."""

        try:
            return await self.hybrid_retriever.search(
                query=query,
                n_results=n_results,
                include_rag=include_rag,
                include_graph=include_graph,
            )
        except Exception as e:
            logger.error(f"Traditional search failed: {e}")
            return []

    async def _get_structured_data_results(
        self,
        query: str,
        excel_files: List[Union[Path, str]] = None,
        json_files: List[Union[Path, str]] = None,
        context_sources: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get results from structured data analysis."""

        results = []

        try:
            # Process Excel files
            if excel_files:
                excel_tasks = [
                    analyze_excel_query(
                        query=query,
                        file_source=file_path,
                        context_sources=context_sources,
                        enable_web_search=False,  # Avoid duplicate web searches
                        enable_rag=False,  # Avoid duplicate RAG searches
                    )
                    for file_path in excel_files
                ]

                excel_results = await asyncio.gather(
                    *excel_tasks, return_exceptions=True
                )

                for result in excel_results:
                    if not isinstance(result, Exception) and result.get(
                        "success", False
                    ):
                        results.append(result)

            # TODO: Process JSON files similarly
            if json_files:
                logger.info(f"JSON file processing not yet implemented: {json_files}")

            return results

        except Exception as e:
            logger.error(f"Structured data analysis failed: {e}")
            return []

    async def _get_web_search_results(
        self, query: str, max_results: int
    ) -> List[Dict[str, Any]]:
        """Get results from web search."""

        try:
            return await self.web_search.search(query, max_results=max_results)
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []

    def _convert_traditional_results(
        self, results: List[HybridResult]
    ) -> List[EnhancedHybridResult]:
        """Convert traditional RAG+KG results to enhanced format."""

        enhanced_results = []

        for result in results:
            enhanced_result = EnhancedHybridResult(
                content=result.content,
                metadata=result.metadata,
                source_type=result.source_type,
                relevance_score=result.relevance_score,
                entities=getattr(result, "entities", []),
                relationships=getattr(result, "relationships", []),
                confidence_score=result.relevance_score,
            )
            enhanced_results.append(enhanced_result)

        return enhanced_results

    def _convert_structured_results(
        self, results: List[Dict[str, Any]]
    ) -> List[EnhancedHybridResult]:
        """Convert structured data analysis results to enhanced format."""

        enhanced_results = []

        for result in results:
            if not result.get("success", False):
                continue

            excel_analysis = result.get("excel_analysis", {})
            direct_answer = excel_analysis.get("direct_answer", "")
            schema_summary = excel_analysis.get("schema_summary", {})
            code_executed = excel_analysis.get("code_executed", "")

            # Create content summary
            content_parts = []
            if direct_answer:
                content_parts.append(f"Direct Answer: {direct_answer}")

            if schema_summary.get("main_tables"):
                tables_info = []
                for table in schema_summary["main_tables"][:3]:  # Limit to 3 tables
                    tables_info.append(f"{table['name']} ({table['columns']} columns)")
                content_parts.append(f"Data Tables: {', '.join(tables_info)}")

            content = (
                "\n".join(content_parts)
                if content_parts
                else "Structured data analysis completed"
            )

            enhanced_result = EnhancedHybridResult(
                content=content,
                metadata={
                    "file_source": result.get("file_source", "Unknown"),
                    "query": result.get("query", ""),
                    "execution_successful": excel_analysis.get(
                        "execution_successful", False
                    ),
                },
                source_type="structured_data",
                relevance_score=result.get("confidence_score", 0.5),
                schema_info=schema_summary,
                code_executed=code_executed,
                data_analysis=excel_analysis,
                confidence_score=result.get("confidence_score", 0.5),
            )

            enhanced_results.append(enhanced_result)

        return enhanced_results

    def _convert_web_results(
        self, results: List[Dict[str, Any]]
    ) -> List[EnhancedHybridResult]:
        """Convert web search results to enhanced format."""

        enhanced_results = []

        for result in results:
            enhanced_result = EnhancedHybridResult(
                content=f"{result.get('title', '')}\n{result.get('snippet', '')}",
                metadata={
                    "url": result.get("url", ""),
                    "title": result.get("title", ""),
                    "snippet": result.get("snippet", ""),
                },
                source_type="web",
                relevance_score=0.6,  # Default web relevance score
                confidence_score=0.6,
            )
            enhanced_results.append(enhanced_result)

        return enhanced_results

    def _rank_enhanced_results(
        self, results: List[EnhancedHybridResult], query: str
    ) -> List[EnhancedHybridResult]:
        """Rank results using enhanced scoring that considers source type and confidence."""

        # Apply source type weights
        source_weights = {
            "structured_data": 1.2,  # Boost structured data analysis
            "rag": 1.0,
            "graph": 1.1,  # Slightly boost knowledge graph
            "web": 0.8,  # Lower weight for web results
        }

        # Calculate enhanced scores
        for result in results:
            base_score = result.relevance_score
            source_weight = source_weights.get(result.source_type, 1.0)
            confidence_boost = (result.confidence_score or 0.5) * 0.2

            # Boost structured data results that executed successfully
            execution_boost = 0.0
            if result.source_type == "structured_data" and result.metadata.get(
                "execution_successful", False
            ):
                execution_boost = 0.3

            result.relevance_score = min(
                base_score * source_weight + confidence_boost + execution_boost, 1.0
            )

        # Sort by enhanced relevance score
        return sorted(results, key=lambda x: x.relevance_score, reverse=True)


class StructuredDataDetector:
    """Detects and categorizes structured data files for processing."""

    @staticmethod
    def detect_structured_files(directory: Path) -> Dict[str, List[Path]]:
        """Detect structured data files in a directory."""

        structured_files = {"excel": [], "json": [], "csv": []}

        try:
            # Find Excel files
            for pattern in ["**/*.xlsx", "**/*.xls"]:
                structured_files["excel"].extend(directory.glob(pattern))

            # Find JSON files
            structured_files["json"].extend(directory.glob("**/*.json"))

            # Find CSV files
            structured_files["csv"].extend(directory.glob("**/*.csv"))

            logger.info(
                f"Detected structured files: {[(k, len(v)) for k, v in structured_files.items()]}"
            )

        except Exception as e:
            logger.error(f"Error detecting structured files: {e}")

        return structured_files

    @staticmethod
    def is_url_structured_data(url: str) -> bool:
        """Check if URL points to structured data."""

        structured_extensions = [".xlsx", ".xls", ".json", ".csv"]
        return any(url.lower().endswith(ext) for ext in structured_extensions)


# Convenience functions for integration


async def enhanced_query_search(
    query: str,
    data_directory: Optional[Path] = None,
    excel_files: List[Union[Path, str]] = None,
    include_web: bool = True,
    n_results: int = 15,
    schema_file: Optional[Path] = None,
    use_multi_file_analysis: bool = True,
) -> List[EnhancedHybridResult]:
    """
    Comprehensive search function that automatically detects and processes all data types.
    Now supports multi-file analysis using preprocessed schemas.

    Args:
        query: Search query
        data_directory: Directory to scan for structured data files
        excel_files: Specific Excel files to analyze
        include_web: Whether to include web search
        n_results: Maximum results to return
        schema_file: Path to preprocessed schema file (for multi-file analysis)
        use_multi_file_analysis: Whether to use the new multi-file analyst
    """

    # Use multi-file analysis if schema file is available
    if (
        use_multi_file_analysis
        and schema_file
        and schema_file.exists()
        and data_directory
    ):
        try:
            from src.tools.multi_file_analyst import analyze_multi_file_query

            logger.info("Using multi-file analysis approach")
            multi_result = await analyze_multi_file_query(
                query=query,
                data_directory=data_directory,
                schema_file=schema_file,
                max_files=3,  # Limit to prevent context overflow
            )

            if "error" not in multi_result:
                # Convert multi-file result to enhanced format
                enhanced_result = EnhancedHybridResult(
                    content=str(multi_result.get("result", {}).get("analysis", "")),
                    metadata={
                        "analysis_type": "multi_file",
                        "selected_files": multi_result.get("selected_files", []),
                        "file_scores": multi_result.get("file_relevance_scores", []),
                    },
                    source_type="structured_data",
                    relevance_score=0.9,  # High relevance for multi-file analysis
                    confidence_score=0.9,
                    data_analysis=multi_result.get("result", {}),
                )

                return [enhanced_result]

        except Exception as e:
            logger.warning(
                f"Multi-file analysis failed, falling back to single-file: {e}"
            )

    # Fallback to original approach
    retriever = EnhancedHybridRetriever()

    # Auto-detect structured files if directory provided
    detected_files = {"excel": excel_files or []}

    if data_directory and data_directory.exists():
        detected = StructuredDataDetector.detect_structured_files(data_directory)
        if not excel_files:
            detected_files["excel"] = detected["excel"][
                :3
            ]  # Limit files to prevent context overflow
        detected_files["json"] = detected["json"][:3]

    # Perform enhanced search
    results = await retriever.enhanced_search(
        query=query,
        n_results=n_results,
        include_rag=True,
        include_graph=True,
        include_structured_data=bool(
            detected_files["excel"] or detected_files.get("json")
        ),
        include_web_search=include_web,
        excel_files=detected_files["excel"],
        json_files=detected_files.get("json", []),
    )

    return results


async def analyze_structured_data_query(
    query: str, file_source: Union[Path, str], context_sources: List[str] = None
) -> EnhancedHybridResult:
    """
    Analyze a single structured data file with comprehensive context.

    This function focuses specifically on structured data analysis while
    still incorporating RAG and web context for completeness.
    """

    processor = HybridExcelProcessor()

    result = await processor.process_excel_query(
        query=query,
        file_source=file_source,
        context_sources=context_sources,
        web_search_enabled=True,
        rag_enabled=True,
    )

    # Convert to enhanced result format
    if result.get("success", False):
        excel_analysis = result.get("excel_analysis", {})

        enhanced_result = EnhancedHybridResult(
            content=excel_analysis.get("direct_answer", "Analysis completed"),
            metadata={
                "file_source": str(file_source),
                "query": query,
                "execution_successful": excel_analysis.get(
                    "execution_successful", False
                ),
            },
            source_type="structured_data",
            relevance_score=result.get("confidence_score", 0.5),
            schema_info=excel_analysis.get("schema_summary", {}),
            code_executed=excel_analysis.get("code_executed", ""),
            data_analysis=excel_analysis,
            confidence_score=result.get("confidence_score", 0.5),
        )

        return enhanced_result

    else:
        # Return error result
        return EnhancedHybridResult(
            content=f"Analysis failed: {result.get('error', 'Unknown error')}",
            metadata={"file_source": str(file_source), "query": query},
            source_type="structured_data",
            relevance_score=0.0,
            confidence_score=0.0,
        )
