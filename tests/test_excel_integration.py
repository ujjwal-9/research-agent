"""
Test Excel Code Analyzer integration with research agents.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import asyncio
from pathlib import Path

from src.tools.excel_code_analyzer import ExcelCodeAnalyzer, ExcelAnalysisResult
from src.tools.document_retriever import SearchResult
from src.agents.document_analyst import DocumentAnalystAgent, DocumentAnalysis


class TestExcelCodeAnalyzer(unittest.TestCase):
    """Test the Excel Code Analyzer tool."""

    def setUp(self):
        """Set up test fixtures."""
        self.excel_analyzer = ExcelCodeAnalyzer()

        # Mock search results with Excel file references
        self.mock_search_results = [
            SearchResult(
                content="This is financial data from Q1 2024",
                score=0.85,
                metadata={
                    "document_name": "financial_report_q1.xlsx",
                    "file_path": "/path/to/financial_report_q1.xlsx",
                    "chunk_index": 1,
                },
                source="financial_report_q1.xlsx",
            ),
            SearchResult(
                content="Budget analysis for the current year",
                score=0.78,
                metadata={
                    "document_name": "budget_2024.xlsx",
                    "file_path": "/path/to/budget_2024.xlsx",
                    "chunk_index": 2,
                },
                source="budget_2024.xlsx",
            ),
            SearchResult(
                content="Regular text document without Excel data",
                score=0.65,
                metadata={"document_name": "text_report.pdf", "chunk_index": 1},
                source="text_report.pdf",
            ),
        ]

    def test_identify_excel_files_from_results(self):
        """Test identification of Excel files from search results."""
        excel_files = self.excel_analyzer._identify_excel_files_from_results(
            self.mock_search_results
        )

        # Should identify Excel files but not PDF files
        self.assertGreater(len(excel_files), 0)

        # Check that Excel extensions are properly identified
        for file_path in excel_files:
            self.assertTrue(
                any(file_path.endswith(ext) for ext in [".xlsx", ".xls", ".csv"]),
                f"File {file_path} should have Excel extension",
            )

    def test_is_excel_file(self):
        """Test Excel file identification."""
        # Test various Excel file formats
        self.assertTrue(self.excel_analyzer._is_excel_file("test.xlsx"))
        self.assertTrue(self.excel_analyzer._is_excel_file("test.xls"))
        self.assertTrue(self.excel_analyzer._is_excel_file("test.csv"))
        self.assertTrue(self.excel_analyzer._is_excel_file("test.xlsm"))

        # Test non-Excel files
        self.assertFalse(self.excel_analyzer._is_excel_file("test.pdf"))
        self.assertFalse(self.excel_analyzer._is_excel_file("test.docx"))
        self.assertFalse(self.excel_analyzer._is_excel_file("test.txt"))
        self.assertFalse(self.excel_analyzer._is_excel_file(""))
        self.assertFalse(self.excel_analyzer._is_excel_file(None))

    def test_build_enhanced_context(self):
        """Test building enhanced context from search results."""
        additional_context = "Focus on financial metrics"

        enhanced_context = self.excel_analyzer._build_enhanced_context(
            self.mock_search_results, additional_context
        )

        self.assertIn(additional_context, enhanced_context)
        self.assertIn("financial_report_q1.xlsx", enhanced_context)
        self.assertIn("budget_2024.xlsx", enhanced_context)

    @patch("src.tools.excel_code_analyzer.ExcelAnalyzer")
    def test_analyze_excel_from_search_results(self, mock_excel_analyzer_class):
        """Test Excel analysis from search results."""
        # Mock the ExcelAnalyzer
        mock_analyzer = MagicMock()
        mock_excel_analyzer_class.return_value = mock_analyzer

        # Mock successful analysis result
        mock_analyzer.analyze_excel_files.return_value = {
            "success": True,
            "result": "Analysis completed successfully",
            "files_analyzed": ["/path/to/financial_report_q1.xlsx"],
            "output_directory": "outputs",
        }

        # Create new analyzer instance to use the mock
        analyzer = ExcelCodeAnalyzer()

        result = analyzer.analyze_excel_from_search_results(
            search_results=self.mock_search_results,
            user_query="What are the financial trends?",
            additional_context="Focus on revenue",
        )

        self.assertIsInstance(result, ExcelAnalysisResult)
        self.assertEqual(result.query, "What are the financial trends?")

    def test_get_excel_files_summary(self):
        """Test getting Excel files summary."""
        with patch.object(
            self.excel_analyzer, "_identify_excel_files_from_results"
        ) as mock_identify:
            # Mock finding Excel files
            mock_identify.return_value = {"/path/to/test.xlsx", "/path/to/data.csv"}

            with patch.object(
                self.excel_analyzer.excel_analyzer, "get_file_summary"
            ) as mock_summary:
                mock_summary.return_value = {
                    "success": True,
                    "summary": "2 files analyzed",
                    "files_examined": ["/path/to/test.xlsx", "/path/to/data.csv"],
                }

                summary = self.excel_analyzer.get_excel_files_summary(
                    self.mock_search_results
                )

                self.assertEqual(summary["excel_files_found"], 2)
                self.assertIn("2 files analyzed", summary["summary"]["summary"])


class TestDocumentAnalystIntegration(unittest.TestCase):
    """Test Excel analysis integration with DocumentAnalyst."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyst = DocumentAnalystAgent()

    @patch(
        "src.agents.document_analyst.DocumentAnalystAgent._execute_internal_searches"
    )
    @patch("src.agents.document_analyst.DocumentAnalystAgent._extract_relevant_links")
    @patch("src.agents.document_analyst.DocumentAnalystAgent._analyze_search_coverage")
    @patch("src.agents.document_analyst.DocumentAnalystAgent._extract_key_findings")
    @patch("src.agents.document_analyst.DocumentAnalystAgent._create_source_summary")
    @patch("src.agents.document_analyst.DocumentAnalystAgent._perform_excel_analysis")
    async def test_excel_analysis_integration(
        self,
        mock_excel_analysis,
        mock_source_summary,
        mock_key_findings,
        mock_coverage,
        mock_links,
        mock_searches,
    ):
        """Test that Excel analysis is properly integrated into document analysis."""
        from src.agents.base_agent import AgentContext, AgentResult, AgentState
        from dataclasses import dataclass

        # Mock research plan
        @dataclass
        class MockResearchPlan:
            research_question: str = "Test question"
            internal_search_queries: list = None

            def __post_init__(self):
                if self.internal_search_queries is None:
                    self.internal_search_queries = ["test query"]

        mock_plan = MockResearchPlan()
        mock_planner_result = AgentResult(
            agent_name="research_planner", status=AgentState.COMPLETED, data=mock_plan
        )

        # Create context
        context = AgentContext(research_question="Test question", user_requirements={})
        context.agent_results["research_planner"] = mock_planner_result

        # Setup mocks
        mock_searches.return_value = []
        mock_links.return_value = []
        mock_coverage.return_value = {}
        mock_key_findings.return_value = []
        mock_source_summary.return_value = {}

        # Mock Excel analysis result
        mock_excel_result = ExcelAnalysisResult(
            query="Test question",
            excel_files_analyzed=["/path/to/test.xlsx"],
            analysis_result={"success": True, "result": "Test analysis"},
            source_chunks=[],
            success=True,
        )
        mock_excel_analysis.return_value = mock_excel_result

        # Execute the analyst
        result = await self.analyst.execute(context)

        # Verify Excel analysis was called
        mock_excel_analysis.assert_called_once()

        # Verify result includes Excel analysis
        self.assertEqual(result.status, AgentState.COMPLETED)
        self.assertIsInstance(result.data, DocumentAnalysis)
        self.assertEqual(result.data.excel_analysis, mock_excel_result)


if __name__ == "__main__":
    # Run async tests
    def run_async_test(test_func):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(test_func())
        finally:
            loop.close()

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestExcelCodeAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestDocumentAnalystIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
