"""Hybrid Excel processor integrating schema detection, code execution, and RAG+KG retrieval."""

import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from loguru import logger
import tempfile
import aiohttp
import aiofiles
from urllib.parse import urlparse

from src.tools.schema_detector import SchemaDetector
from src.tools.code_executor import ExcelDataAnalyzer
from src.tools.web_search import WebSearchTool
from src.knowledge_graph.hybrid_retriever import HybridRetriever
from src.ingestion.document_store import DocumentStore


class HybridExcelProcessor:
    """Processes Excel files using multi-modal approach: schema detection + code execution + RAG+KG."""

    def __init__(self):
        self.schema_detector = SchemaDetector()
        self.excel_analyzer = ExcelDataAnalyzer()
        self.web_search = WebSearchTool()
        self.hybrid_retriever = HybridRetriever()
        self.document_store = DocumentStore()

    async def process_excel_query(
        self,
        query: str,
        file_source: Union[Path, str],  # Local path or URL
        context_sources: List[str] = None,
        web_search_enabled: bool = True,
        rag_enabled: bool = True,
    ) -> Dict[str, Any]:
        """
        Process Excel query using hybrid approach:
        1. Download/load Excel file
        2. Detect schema with context
        3. Execute code analysis
        4. Retrieve RAG+KG context
        5. Perform web search
        6. Synthesize comprehensive response
        """

        try:
            # Step 1: Prepare Excel file
            file_path, web_source_url = await self._prepare_excel_file(file_source)

            # Step 2: Gather contextual information
            context_info = await self._gather_comprehensive_context(
                file_path,
                query,
                context_sources,
                web_source_url,
                web_search_enabled,
                rag_enabled,
            )

            # Step 3: Analyze Excel with code execution
            excel_analysis = await self.excel_analyzer.analyze_excel_with_query(
                file_path=file_path,
                query=query,
                context_pdfs=context_info.get("pdf_files", []),
                web_source_url=web_source_url,
                additional_context=self._format_context_for_analysis(context_info),
            )

            # Step 4: Synthesize comprehensive response
            comprehensive_response = await self._synthesize_response(
                query=query,
                excel_analysis=excel_analysis,
                context_info=context_info,
                file_source=file_source,
            )

            return comprehensive_response

        except Exception as e:
            logger.error(f"Hybrid Excel processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "file_source": str(file_source),
            }

        finally:
            # Cleanup temporary files if created
            if isinstance(file_source, str) and file_source.startswith("http"):
                try:
                    if file_path.exists():
                        file_path.unlink()
                except:
                    pass

    async def _prepare_excel_file(
        self, file_source: Union[Path, str]
    ) -> tuple[Path, Optional[str]]:
        """Download Excel file if URL, or validate local path."""

        if isinstance(file_source, str) and file_source.startswith("http"):
            # Download from URL
            logger.info(f"Downloading Excel file from: {file_source}")

            parsed_url = urlparse(file_source)
            filename = Path(parsed_url.path).name or "downloaded_excel.xlsx"

            # Create temporary file
            temp_file = Path(tempfile.gettempdir()) / filename

            async with aiohttp.ClientSession() as session:
                async with session.get(file_source) as response:
                    if response.status == 200:
                        async with aiofiles.open(temp_file, "wb") as f:
                            async for chunk in response.content.iter_chunked(8192):
                                await f.write(chunk)

                        logger.info(f"Downloaded Excel file to: {temp_file}")
                        return temp_file, file_source
                    else:
                        raise Exception(
                            f"Failed to download file: HTTP {response.status}"
                        )

        else:
            # Local file
            file_path = Path(file_source)
            if not file_path.exists():
                raise FileNotFoundError(f"Excel file not found: {file_path}")

            logger.info(f"Using local Excel file: {file_path}")
            return file_path, None

    async def _gather_comprehensive_context(
        self,
        file_path: Path,
        query: str,
        context_sources: List[str],
        web_source_url: Optional[str],
        web_search_enabled: bool,
        rag_enabled: bool,
    ) -> Dict[str, Any]:
        """Gather context from multiple sources in parallel."""

        context_tasks = []

        # Task 1: RAG + Knowledge Graph search
        if rag_enabled:
            context_tasks.append(self._get_rag_context(query))

        # Task 2: Web search for additional context
        if web_search_enabled:
            context_tasks.append(
                self._get_web_context(query, file_path, web_source_url)
            )

        # Task 3: Local context (PDFs, related files)
        context_tasks.append(self._get_local_context(file_path, context_sources))

        # Execute all context gathering in parallel
        context_results = await asyncio.gather(*context_tasks, return_exceptions=True)

        # Compile context information
        compiled_context = {
            "rag_context": {},
            "web_context": {},
            "local_context": {},
            "pdf_files": [],
            "context_sources": context_sources or [],
        }

        # Process results
        result_idx = 0

        if rag_enabled and result_idx < len(context_results):
            if not isinstance(context_results[result_idx], Exception):
                compiled_context["rag_context"] = context_results[result_idx]
            result_idx += 1

        if web_search_enabled and result_idx < len(context_results):
            if not isinstance(context_results[result_idx], Exception):
                compiled_context["web_context"] = context_results[result_idx]
            result_idx += 1

        if result_idx < len(context_results):
            if not isinstance(context_results[result_idx], Exception):
                compiled_context["local_context"] = context_results[result_idx]
                compiled_context["pdf_files"] = context_results[result_idx].get(
                    "pdf_files", []
                )

        return compiled_context

    async def _get_rag_context(self, query: str) -> Dict[str, Any]:
        """Get context from RAG + Knowledge Graph."""

        try:
            # Search using hybrid retriever
            rag_results = await self.hybrid_retriever.search(
                query=query, n_results=10, include_graph=True, include_rag=True
            )

            # Format results
            formatted_results = []
            for result in rag_results:
                formatted_results.append(
                    {
                        "content": result.content,
                        "source": result.metadata.get("file_path", "Unknown"),
                        "source_type": result.source_type,
                        "relevance_score": result.relevance_score,
                        "entities": getattr(result, "entities", []),
                        "relationships": getattr(result, "relationships", []),
                    }
                )

            return {
                "results": formatted_results,
                "total_found": len(formatted_results),
                "search_query": query,
            }

        except Exception as e:
            logger.error(f"RAG context retrieval failed: {e}")
            return {"error": str(e)}

    async def _get_web_context(
        self, query: str, file_path: Path, web_source_url: Optional[str]
    ) -> Dict[str, Any]:
        """Get context from web search."""

        try:
            web_queries = [
                f"{query} excel spreadsheet data analysis",
                f"{file_path.stem} data documentation",
            ]

            if web_source_url:
                domain = urlparse(web_source_url).netloc
                web_queries.append(f"site:{domain} {file_path.stem}")

            # Execute web searches in parallel
            search_tasks = [
                self.web_search.search(q, max_results=5) for q in web_queries
            ]

            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

            # Compile web results
            all_web_results = []
            for i, results in enumerate(search_results):
                if not isinstance(results, Exception):
                    for result in results:
                        result["search_query"] = web_queries[i]
                        all_web_results.append(result)

            return {
                "results": all_web_results,
                "search_queries": web_queries,
                "source_url": web_source_url,
                "total_found": len(all_web_results),
            }

        except Exception as e:
            logger.error(f"Web context retrieval failed: {e}")
            return {"error": str(e)}

    async def _get_local_context(
        self, file_path: Path, context_sources: List[str]
    ) -> Dict[str, Any]:
        """Get context from local files and directories."""

        try:
            local_context = {
                "directory": str(file_path.parent),
                "related_files": [],
                "pdf_files": context_sources or [],
            }

            # Find related files in the same directory
            parent_dir = file_path.parent

            for ext in [".pdf", ".docx", ".txt", ".xlsx", ".csv"]:
                related_files = list(parent_dir.glob(f"*{ext}"))
                for f in related_files[:10]:  # Limit to avoid too many files
                    if f != file_path:  # Exclude the current file
                        local_context["related_files"].append(
                            {
                                "name": f.name,
                                "path": str(f),
                                "type": ext[1:],  # Remove the dot
                            }
                        )

            # If context_sources not provided, look for PDFs in the directory
            if not context_sources:
                pdf_files = [str(f) for f in parent_dir.glob("*.pdf")]
                local_context["pdf_files"] = pdf_files[:5]  # Limit PDFs

            return local_context

        except Exception as e:
            logger.error(f"Local context retrieval failed: {e}")
            return {"error": str(e)}

    def _format_context_for_analysis(self, context_info: Dict[str, Any]) -> str:
        """Format gathered context for use in code generation."""

        context_text = []

        # Add RAG context
        rag_context = context_info.get("rag_context", {})
        if rag_context.get("results"):
            context_text.append("RELATED DOCUMENTS:")
            for result in rag_context["results"][:3]:  # Top 3 most relevant
                context_text.append(f"- {result['content'][:200]}...")

        # Add web context
        web_context = context_info.get("web_context", {})
        if web_context.get("results"):
            context_text.append("\nWEB CONTEXT:")
            for result in web_context["results"][:3]:
                context_text.append(
                    f"- {result['title']}: {result['snippet'][:150]}..."
                )

        # Add local file context
        local_context = context_info.get("local_context", {})
        if local_context.get("related_files"):
            context_text.append("\nRELATED FILES IN DIRECTORY:")
            for file_info in local_context["related_files"][:5]:
                context_text.append(f"- {file_info['name']} ({file_info['type']})")

        return "\n".join(context_text)

    async def _synthesize_response(
        self,
        query: str,
        excel_analysis: Dict[str, Any],
        context_info: Dict[str, Any],
        file_source: Union[Path, str],
    ) -> Dict[str, Any]:
        """Synthesize comprehensive response combining all sources."""

        # Extract key information
        schema_info = excel_analysis.get("schema", {})
        analysis_result = excel_analysis.get("analysis", {})
        suggested_queries = excel_analysis.get("suggested_queries", [])

        # Build comprehensive response
        response = {
            "success": True,
            "query": query,
            "file_source": str(file_source),
            "excel_analysis": {
                "direct_answer": self._extract_direct_answer(analysis_result),
                "schema_summary": self._summarize_schema(schema_info),
                "code_executed": analysis_result.get("generated_code", ""),
                "execution_successful": analysis_result.get("success", False),
                "suggested_queries": suggested_queries,
            },
            "contextual_information": {
                "rag_findings": self._summarize_rag_context(
                    context_info.get("rag_context", {})
                ),
                "web_findings": self._summarize_web_context(
                    context_info.get("web_context", {})
                ),
                "local_files": context_info.get("local_context", {}).get(
                    "related_files", []
                ),
            },
            "confidence_score": self._calculate_confidence_score(
                excel_analysis, context_info
            ),
            "references": self._compile_references(context_info, file_source),
        }

        return response

    def _extract_direct_answer(self, analysis_result: Dict[str, Any]) -> str:
        """Extract direct answer from code execution result."""

        if not analysis_result.get("success", False):
            return f"Analysis failed: {analysis_result.get('error', 'Unknown error')}"

        exec_result = analysis_result.get("execution_result", {})

        if not exec_result.get("execution_success", False):
            return f"Code execution failed: {exec_result.get('error_message', 'Unknown error')}"

        result_data = exec_result.get("result", {})

        if isinstance(result_data, dict):
            return result_data.get(
                "answer",
                str(result_data.get("summary", "Analysis completed successfully")),
            )
        else:
            return str(result_data)

    def _summarize_schema(self, schema_info: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize schema information."""

        if "error" in schema_info:
            return {"error": schema_info["error"]}

        schemas = schema_info.get("schemas", {})
        summary = {"total_sheets": len(schemas), "sheet_types": [], "main_tables": []}

        for sheet_name, schema in schemas.items():
            sheet_type = schema.get("sheet_type", "unknown")
            summary["sheet_types"].append(f"{sheet_name}: {sheet_type}")

            tables = schema.get("tables", [])
            for table in tables:
                summary["main_tables"].append(
                    {
                        "sheet": sheet_name,
                        "name": table.get("table_name", "Unknown"),
                        "type": table.get("table_type", "unknown"),
                        "columns": len(table.get("columns", [])),
                    }
                )

        return summary

    def _summarize_rag_context(self, rag_context: Dict[str, Any]) -> List[str]:
        """Summarize RAG context findings."""

        if "error" in rag_context:
            return [f"RAG search error: {rag_context['error']}"]

        results = rag_context.get("results", [])
        summaries = []

        for result in results[:5]:  # Top 5 results
            source = result.get("source", "Unknown")
            content_preview = result.get("content", "")[:150] + "..."
            summaries.append(f"From {source}: {content_preview}")

        return summaries

    def _summarize_web_context(self, web_context: Dict[str, Any]) -> List[str]:
        """Summarize web search findings."""

        if "error" in web_context:
            return [f"Web search error: {web_context['error']}"]

        results = web_context.get("results", [])
        summaries = []

        for result in results[:5]:  # Top 5 results
            title = result.get("title", "Unknown")
            url = result.get("url", "")
            snippet = result.get("snippet", "")[:100] + "..."
            summaries.append(f"{title} ({url}): {snippet}")

        return summaries

    def _calculate_confidence_score(
        self, excel_analysis: Dict[str, Any], context_info: Dict[str, Any]
    ) -> float:
        """Calculate confidence score based on analysis quality."""

        score = 0.0

        # Excel analysis quality (40% weight)
        analysis = excel_analysis.get("analysis", {})
        if analysis.get("success", False):
            exec_result = analysis.get("execution_result", {})
            if exec_result.get("execution_success", False):
                score += 0.4
            else:
                score += 0.1  # Partial credit for attempt

        # Schema detection quality (30% weight)
        schema = excel_analysis.get("schema", {})
        if "error" not in schema:
            score += 0.3

        # Context availability (30% weight)
        context_score = 0.0
        if context_info.get("rag_context", {}).get("results"):
            context_score += 0.15
        if context_info.get("web_context", {}).get("results"):
            context_score += 0.15
        score += context_score

        return min(score, 1.0)  # Cap at 1.0

    def _compile_references(
        self, context_info: Dict[str, Any], file_source: Union[Path, str]
    ) -> List[Dict[str, str]]:
        """Compile all references used in the analysis."""

        references = []

        # Primary source
        references.append(
            {
                "type": "primary",
                "source": str(file_source),
                "description": "Excel file analyzed",
            }
        )

        # RAG sources
        rag_results = context_info.get("rag_context", {}).get("results", [])
        for result in rag_results[:3]:
            references.append(
                {
                    "type": "document",
                    "source": result.get("source", "Unknown"),
                    "description": f"Related document (relevance: {result.get('relevance_score', 0):.2f})",
                }
            )

        # Web sources
        web_results = context_info.get("web_context", {}).get("results", [])
        for result in web_results[:3]:
            references.append(
                {
                    "type": "web",
                    "source": result.get("url", "Unknown"),
                    "description": result.get("title", "Web resource"),
                }
            )

        return references


# Convenience functions for easy integration


async def analyze_excel_query(
    query: str,
    file_source: Union[Path, str],
    context_sources: List[str] = None,
    enable_web_search: bool = True,
    enable_rag: bool = True,
) -> Dict[str, Any]:
    """Convenience function for analyzing Excel with query."""

    processor = HybridExcelProcessor()
    return await processor.process_excel_query(
        query=query,
        file_source=file_source,
        context_sources=context_sources,
        web_search_enabled=enable_web_search,
        rag_enabled=enable_rag,
    )


async def batch_analyze_excel(
    queries: List[str], file_source: Union[Path, str], context_sources: List[str] = None
) -> List[Dict[str, Any]]:
    """Analyze multiple queries against the same Excel file."""

    processor = HybridExcelProcessor()
    results = []

    for query in queries:
        result = await processor.process_excel_query(
            query=query, file_source=file_source, context_sources=context_sources
        )
        results.append(result)

    return results
