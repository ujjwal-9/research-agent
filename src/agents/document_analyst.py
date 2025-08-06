"""
Document analyst agent for internal document research and analysis.
"""

import asyncio
from typing import Dict, Any, List
from dataclasses import dataclass

from .base_agent import BaseAgent, AgentContext, AgentResult, AgentState
from src.tools.document_retriever import DocumentRetriever, SearchResult
from src.tools.link_extractor import LinkExtractor, ExtractedLink
from src.tools.excel_code_analyzer import ExcelCodeAnalyzer, ExcelAnalysisResult


@dataclass
class DocumentAnalysis:
    """Results from document analysis."""

    research_question: str
    search_results: List[SearchResult]
    extracted_links: List[ExtractedLink]
    key_findings: List[str]
    coverage_analysis: Dict[str, Any]
    source_summary: Dict[str, Any]
    excel_analysis: ExcelAnalysisResult = None


class DocumentAnalystAgent(BaseAgent):
    """Agent responsible for analyzing internal documents."""

    def __init__(self, collection_name: str = None):
        """Initialize the document analyst agent.

        Args:
            collection_name: Qdrant collection name to search
        """
        super().__init__("document_analyst")
        self.document_retriever = DocumentRetriever(collection_name)
        self.link_extractor = LinkExtractor(self.document_retriever)
        self.excel_analyzer = ExcelCodeAnalyzer(self.document_retriever)

    async def execute(self, context: AgentContext) -> AgentResult:
        """Analyze internal documents based on research plan.

        Args:
            context: Shared context containing research plan

        Returns:
            AgentResult containing document analysis
        """
        try:
            # Get research plan from context
            research_plan = context.agent_results.get("research_planner")
            if not research_plan or not research_plan.data:
                raise ValueError("Research plan not found in context")

            plan = research_plan.data
            self.logger.info(
                f"📚 Analyzing internal documents for: {plan.research_question}"
            )

            # Execute internal searches based on plan
            all_search_results = await self._execute_internal_searches(
                plan.internal_search_queries
            )

            # Extract links from search results
            extracted_links = await self._extract_relevant_links(
                plan.research_question, all_search_results
            )

            # Analyze search coverage and quality
            coverage_analysis = await self._analyze_search_coverage(
                plan.internal_search_queries, all_search_results
            )

            # Generate key findings
            key_findings = await self._extract_key_findings(
                plan.research_question, all_search_results
            )

            # Create source summary
            source_summary = await self._create_source_summary(all_search_results)

            # Perform Excel analysis if Excel files are found in search results
            excel_analysis = await self._perform_excel_analysis(
                plan.research_question, all_search_results
            )

            # Create document analysis
            analysis = DocumentAnalysis(
                research_question=plan.research_question,
                search_results=all_search_results,
                extracted_links=extracted_links,
                key_findings=key_findings,
                coverage_analysis=coverage_analysis,
                source_summary=source_summary,
                excel_analysis=excel_analysis,
            )

            excel_info = ""
            if excel_analysis and excel_analysis.success:
                excel_info = (
                    f", analyzed {len(excel_analysis.excel_files_analyzed)} Excel files"
                )
            elif excel_analysis and not excel_analysis.success:
                excel_info = f", Excel analysis failed: {excel_analysis.error_message}"

            self.logger.info(
                f"✅ Analyzed {len(all_search_results)} documents, "
                f"found {len(key_findings)} key findings, "
                f"extracted {len(extracted_links)} links{excel_info}"
            )

            return AgentResult(
                agent_name=self.name,
                status=AgentState.COMPLETED,
                data=analysis,
                metadata={
                    "total_search_results": len(all_search_results),
                    "unique_sources": len(set(r.source for r in all_search_results)),
                    "extracted_links_count": len(extracted_links),
                    "key_findings_count": len(key_findings),
                    "queries_executed": len(plan.internal_search_queries),
                },
            )

        except Exception as e:
            self.logger.error(f"❌ Error analyzing internal documents: {e}")
            return AgentResult(
                agent_name=self.name, status=AgentState.ERROR, data=None, error=str(e)
            )

    async def _execute_internal_searches(
        self, search_queries: List[str]
    ) -> List[SearchResult]:
        """Execute all internal search queries.

        Args:
            search_queries: List of search queries to execute

        Returns:
            Combined list of search results
        """
        all_results = []

        # Execute searches with some parallelization
        batch_size = 3  # Process 3 queries at a time
        for i in range(0, len(search_queries), batch_size):
            batch_queries = search_queries[i : i + batch_size]

            # Create tasks for this batch
            tasks = [self._execute_single_search(query) for query in batch_queries]

            # Execute batch
            batch_results = await asyncio.gather(*tasks)

            # Flatten results
            for results in batch_results:
                all_results.extend(results)

        # Deduplicate results by content similarity
        unique_results = self._deduplicate_search_results(all_results)

        self.logger.info(
            f"Executed {len(search_queries)} queries, "
            f"got {len(all_results)} total results, "
            f"kept {len(unique_results)} unique results"
        )

        return unique_results

    async def _execute_single_search(self, query: str) -> List[SearchResult]:
        """Execute a single search query.

        Args:
            query: Search query to execute

        Returns:
            List of search results
        """
        try:
            self.logger.debug(f"🔍 Searching: {query}")

            # Search with reasonable limits
            results = self.document_retriever.search_documents(
                query=query,
                limit=15,  # Up to 15 results per query
                score_threshold=0.3,  # Lower threshold for broader coverage
            )

            self.logger.debug(f"Found {len(results)} results for: {query}")
            return results

        except Exception as e:
            self.logger.error(f"Error searching for '{query}': {e}")
            return []

    def _deduplicate_search_results(
        self, results: List[SearchResult]
    ) -> List[SearchResult]:
        """Remove duplicate search results based on content similarity.

        Args:
            results: List of search results to deduplicate

        Returns:
            List of unique search results
        """
        if not results:
            return []

        unique_results = []
        seen_content_hashes = set()

        for result in results:
            # Create a simple hash of the content for deduplication
            content_preview = result.content[:200].strip().lower()
            content_hash = hash(content_preview)

            if content_hash not in seen_content_hashes:
                unique_results.append(result)
                seen_content_hashes.add(content_hash)

        # Sort by relevance score
        unique_results.sort(key=lambda x: x.score, reverse=True)

        return unique_results

    async def _extract_relevant_links(
        self, research_question: str, search_results: List[SearchResult]
    ) -> List[ExtractedLink]:
        """Extract relevant links from search results.

        Args:
            research_question: Original research question
            search_results: Search results to extract links from

        Returns:
            List of extracted links
        """
        try:
            # Extract links that are good candidates for external research
            research_links = self.link_extractor.get_external_research_candidates(
                query=research_question, limit=50
            )

            self.logger.debug(
                f"Extracted {len(research_links)} research candidate links"
            )
            return research_links

        except Exception as e:
            self.logger.error(f"Error extracting links: {e}")
            return []

    async def _analyze_search_coverage(
        self, queries: List[str], results: List[SearchResult]
    ) -> Dict[str, Any]:
        """Analyze the coverage and quality of search results.

        Args:
            queries: Original search queries
            results: Search results obtained

        Returns:
            Coverage analysis results
        """
        # Analyze source distribution
        sources = [r.source for r in results]
        source_counts = {}
        for source in sources:
            source_counts[source] = source_counts.get(source, 0) + 1

        # Calculate coverage metrics
        unique_sources = len(set(sources))
        avg_score = sum(r.score for r in results) / len(results) if results else 0

        # Analyze query effectiveness
        queries_with_results = 0
        for query in queries:
            query_results = [r for r in results if query.lower() in r.content.lower()]
            if query_results:
                queries_with_results += 1

        coverage_analysis = {
            "total_results": len(results),
            "unique_sources": unique_sources,
            "source_distribution": source_counts,
            "average_relevance_score": avg_score,
            "queries_total": len(queries),
            "queries_with_results": queries_with_results,
            "query_effectiveness": (
                queries_with_results / len(queries) if queries else 0
            ),
            "results_per_query": len(results) / len(queries) if queries else 0,
        }

        return coverage_analysis

    async def _extract_key_findings(
        self, research_question: str, results: List[SearchResult]
    ) -> List[str]:
        """Extract key findings from search results.

        Args:
            research_question: Original research question
            results: Search results to analyze

        Returns:
            List of key findings
        """
        key_findings = []

        if not results:
            return ["No relevant internal documents found for the research question."]

        # Group results by source for analysis
        results_by_source = {}
        for result in results:
            if result.source not in results_by_source:
                results_by_source[result.source] = []
            results_by_source[result.source].append(result)

        # Extract findings from high-scoring results
        high_score_results = [r for r in results if r.score > 0.7]
        if high_score_results:
            key_findings.append(
                f"Found {len(high_score_results)} highly relevant sections "
                f"across {len(set(r.source for r in high_score_results))} documents"
            )

        # Identify most relevant sources
        if results_by_source:
            best_source = max(
                results_by_source.keys(),
                key=lambda s: max(r.score for r in results_by_source[s]),
            )
            best_score = max(r.score for r in results_by_source[best_source])
            key_findings.append(
                f"Most relevant information found in '{best_source}' "
                f"with relevance score of {best_score:.2f}"
            )

        # Analyze content themes (simple keyword analysis)
        all_content = " ".join(r.content for r in results[:10])  # Top 10 results
        common_terms = self._extract_common_terms(all_content)
        if common_terms:
            key_findings.append(
                f"Common themes identified: {', '.join(common_terms[:5])}"
            )

        # Coverage assessment
        if len(results_by_source) >= 3:
            key_findings.append(
                f"Comprehensive coverage with information from {len(results_by_source)} different sources"
            )
        elif len(results_by_source) == 1:
            key_findings.append(
                f"Limited to single source: {list(results_by_source.keys())[0]}"
            )

        return key_findings

    def _extract_common_terms(self, text: str) -> List[str]:
        """Extract common terms from text content.

        Args:
            text: Text to analyze

        Returns:
            List of common terms
        """
        import re
        from collections import Counter

        # Simple term extraction
        words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())

        # Remove common words
        stop_words = {
            "this",
            "that",
            "with",
            "have",
            "will",
            "from",
            "they",
            "been",
            "were",
            "their",
            "said",
            "each",
            "which",
            "them",
            "than",
            "many",
            "some",
            "what",
            "would",
            "make",
            "like",
            "time",
            "very",
            "when",
            "come",
            "here",
            "more",
            "also",
            "back",
            "after",
            "first",
            "well",
            "work",
            "such",
            "good",
            "only",
            "other",
            "then",
            "being",
            "over",
            "think",
            "where",
            "much",
            "take",
            "most",
            "know",
            "just",
            "into",
        }

        filtered_words = [w for w in words if w not in stop_words]

        # Count and return most common
        word_counts = Counter(filtered_words)
        return [word for word, count in word_counts.most_common(10) if count > 1]

    async def _create_source_summary(
        self, results: List[SearchResult]
    ) -> Dict[str, Any]:
        """Create a summary of sources analyzed.

        Args:
            results: Search results to summarize

        Returns:
            Source summary information
        """
        if not results:
            return {"message": "No sources analyzed"}

        # Group by source
        by_source = {}
        for result in results:
            if result.source not in by_source:
                by_source[result.source] = {
                    "count": 0,
                    "max_score": 0,
                    "avg_score": 0,
                    "total_content_length": 0,
                }

            source_data = by_source[result.source]
            source_data["count"] += 1
            source_data["max_score"] = max(source_data["max_score"], result.score)
            source_data["total_content_length"] += len(result.content)

        # Calculate averages
        for source_data in by_source.values():
            if source_data["count"] > 0:
                source_data["avg_score"] = (
                    sum(r.score for r in results if r.source == source_data)
                    / source_data["count"]
                )

        # Create summary
        total_sources = len(by_source)
        total_results = len(results)
        best_source = max(by_source.keys(), key=lambda s: by_source[s]["max_score"])

        summary = {
            "total_sources": total_sources,
            "total_results": total_results,
            "sources_analyzed": list(by_source.keys()),
            "best_performing_source": best_source,
            "source_details": by_source,
            "overall_avg_score": sum(r.score for r in results) / len(results),
        }

        return summary

    async def _perform_excel_analysis(
        self, research_question: str, search_results: List[SearchResult]
    ) -> ExcelAnalysisResult:
        """Perform Excel code analysis on relevant Excel files found in search results.

        Args:
            research_question: The research question being investigated
            search_results: List of search results to analyze for Excel files

        Returns:
            ExcelAnalysisResult containing the analysis or None if no Excel files found
        """
        try:
            # Get Excel files summary first to check if any exist
            excel_summary = self.excel_analyzer.get_excel_files_summary(search_results)

            if excel_summary["excel_files_found"] == 0:
                self.logger.info("📊 No Excel files found in search results")
                return None

            self.logger.info(
                f"📊 Found {excel_summary['excel_files_found']} Excel files, performing code analysis"
            )

            # Build context for Excel analysis
            excel_context = f"""
            This analysis is part of a larger research investigation into: {research_question}
            
            The Excel files were identified from document chunks that were relevant to this research question.
            Please focus your analysis on aspects that would help answer or provide insights into the research question.
            
            Consider:
            - Data patterns that relate to the research question
            - Key metrics, trends, or statistics
            - Any correlations or insights that address the research objectives
            - Data quality and completeness relevant to the research
            """

            # Perform the Excel analysis
            analysis_result = self.excel_analyzer.analyze_excel_from_search_results(
                search_results=search_results,
                user_query=research_question,
                additional_context=excel_context,
            )

            if analysis_result.success:
                self.logger.info(
                    f"✅ Excel analysis completed successfully for {len(analysis_result.excel_files_analyzed)} files"
                )
            else:
                self.logger.warning(
                    f"⚠️ Excel analysis failed: {analysis_result.error_message}"
                )

            return analysis_result

        except Exception as e:
            self.logger.error(f"❌ Error during Excel analysis: {e}")
            # Return a failed analysis result rather than None to preserve error information
            return ExcelAnalysisResult(
                query=research_question,
                excel_files_analyzed=[],
                analysis_result={"success": False, "error": str(e)},
                source_chunks=search_results,
                success=False,
                error_message=str(e),
            )
