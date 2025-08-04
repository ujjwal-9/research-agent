"""
Main research workflow interface for the multi-agent research system.

This module provides a high-level interface for executing research workflows
and includes example usage patterns.
"""

import logging
import asyncio
import os
from typing import Dict, Any, Optional
from datetime import datetime

from src.agents.orchestrator import ResearchOrchestrator, execute_research_workflow
from src.tools.report_generator import ReportGenerator


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration for the research workflow.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/research_workflow_{timestamp}.log"

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"🔧 Research workflow logging initialized - Log file: {log_file}")


class ResearchWorkflowManager:
    """High-level manager for research workflows."""

    def __init__(self, collection_name: str = None):
        """Initialize the research workflow manager.

        Args:
            collection_name: Qdrant collection name for document retrieval
        """
        self.logger = logging.getLogger(__name__)
        self.collection_name = collection_name
        self.orchestrator = ResearchOrchestrator(collection_name)
        self.report_generator = ReportGenerator()

    async def conduct_research(
        self,
        research_question: str,
        user_requirements: Dict[str, Any] = None,
        export_format: str = "markdown",
    ) -> Dict[str, Any]:
        """Conduct a complete research workflow.

        Args:
            research_question: Research question to investigate
            user_requirements: Optional user requirements and preferences
            export_format: Export format for the report ("markdown", "json", or "both")

        Returns:
            Dictionary containing workflow results and exported reports
        """
        try:
            self.logger.info(f"🚀 Starting research workflow")
            self.logger.info(f"📝 Research Question: {research_question}")

            # Execute workflow
            workflow_result = await self.orchestrator.execute_research_workflow(
                research_question=research_question,
                user_requirements=user_requirements or {},
            )

            # Export reports in requested format(s)
            exported_reports = {}

            if export_format in ["markdown", "both"]:
                markdown_report = self.report_generator.export_to_markdown(
                    workflow_result.research_report
                )
                exported_reports["markdown"] = markdown_report

                # Save to file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"docs/research_report_{timestamp}.md"
                os.makedirs("docs", exist_ok=True)

                with open(filename, "w", encoding="utf-8") as f:
                    f.write(markdown_report)

                exported_reports["markdown_file"] = filename
                self.logger.info(f"📄 Markdown report saved to: {filename}")

            if export_format in ["json", "both"]:
                json_report = self.report_generator.export_to_json(
                    workflow_result.research_report
                )
                exported_reports["json"] = json_report

                # Save to file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"docs/research_report_{timestamp}.json"

                with open(filename, "w", encoding="utf-8") as f:
                    f.write(json_report)

                exported_reports["json_file"] = filename
                self.logger.info(f"📊 JSON report saved to: {filename}")

            # Compile final results
            final_results = {
                "research_question": research_question,
                "workflow_result": workflow_result,
                "exported_reports": exported_reports,
                "execution_summary": {
                    "total_execution_time": workflow_result.workflow_metadata[
                        "total_execution_time"
                    ],
                    "agents_executed": workflow_result.workflow_metadata[
                        "agents_executed"
                    ],
                    "user_interactions": workflow_result.workflow_metadata[
                        "user_interactions"
                    ],
                    "report_sections": len(workflow_result.research_report.sections),
                    "total_sources": len(workflow_result.research_report.sources),
                },
            }

            self.logger.info(
                f"✅ Research workflow completed successfully in "
                f"{workflow_result.workflow_metadata['total_execution_time']:.1f}s"
            )

            return final_results

        except Exception as e:
            self.logger.error(f"❌ Research workflow failed: {e}")
            raise

    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status.

        Returns:
            Current workflow status
        """
        return self.orchestrator.get_workflow_status()

    async def validate_environment(self) -> Dict[str, bool]:
        """Validate that the environment is properly configured.

        Returns:
            Dictionary showing validation results
        """
        validation_results = {}

        # Check required environment variables
        required_env_vars = [
            "OPENAI_API_KEY",
            "QDRANT_URL",
            "QDRANT_SEMANTIC_COLLECTION_NAME",
        ]

        for env_var in required_env_vars:
            validation_results[f"env_{env_var}"] = bool(os.getenv(env_var))

        # Test Qdrant connection
        try:
            from .tools.document_retriever import DocumentRetriever

            retriever = DocumentRetriever(self.collection_name)
            sources = retriever.get_all_sources()
            validation_results["qdrant_connection"] = True
            validation_results["qdrant_sources_available"] = len(sources) > 0
            validation_results["qdrant_source_count"] = len(sources)
            self.logger.info(f"✅ Qdrant connection successful: {self.collection_name}")
        except Exception as e:
            validation_results["qdrant_connection"] = False
            validation_results["qdrant_error"] = str(e)

        # Test OpenAI connection
        try:
            import openai

            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            # Simple test - just check if we can create a client
            validation_results["openai_connection"] = True
        except Exception as e:
            validation_results["openai_connection"] = False
            validation_results["openai_error"] = str(e)

        return validation_results


# Convenience functions for common use cases
async def quick_research(research_question: str, collection_name: str = None) -> str:
    """Conduct quick research and return markdown report.

    Args:
        research_question: Research question to investigate
        collection_name: Optional Qdrant collection name

    Returns:
        Markdown formatted research report
    """
    setup_logging()

    manager = ResearchWorkflowManager(collection_name)
    results = await manager.conduct_research(
        research_question=research_question, export_format="markdown"
    )

    return results["exported_reports"]["markdown"]


async def comprehensive_research(
    research_question: str,
    user_requirements: Dict[str, Any] = None,
    collection_name: str = None,
) -> Dict[str, Any]:
    """Conduct comprehensive research with full workflow tracking.

    Args:
        research_question: Research question to investigate
        user_requirements: User requirements and preferences
        collection_name: Optional Qdrant collection name

    Returns:
        Complete workflow results with all metadata
    """
    setup_logging()

    manager = ResearchWorkflowManager(collection_name)

    # Validate environment first
    validation = await manager.validate_environment()
    logger = logging.getLogger(__name__)

    for check, result in validation.items():
        if isinstance(result, bool):
            status = "✅" if result else "❌"
            logger.info(f"{status} {check}: {result}")

    # Proceed with research
    return await manager.conduct_research(
        research_question=research_question,
        user_requirements=user_requirements,
        export_format="both",
    )


# Example usage patterns
if __name__ == "__main__":
    # Example 1: Quick research
    async def example_quick_research():
        question = "How does semantic chunking work in document processing?"
        report = await quick_research(question)
        print("📄 Quick Research Report:")
        print(report[:500] + "..." if len(report) > 500 else report)

    # Example 2: Comprehensive research
    async def example_comprehensive_research():
        question = "What are the best practices for implementing vector databases in production?"

        user_requirements = {
            "focus_areas": ["performance", "scalability", "security"],
            "depth_preference": "detailed",
            "include_code_examples": True,
        }

        results = await comprehensive_research(
            research_question=question, user_requirements=user_requirements
        )

        print("📊 Comprehensive Research Summary:")
        print(f"- Research Question: {results['research_question']}")
        print(
            f"- Execution Time: {results['execution_summary']['total_execution_time']:.1f}s"
        )
        print(f"- Agents Executed: {results['execution_summary']['agents_executed']}")
        print(f"- Report Sections: {results['execution_summary']['report_sections']}")
        print(f"- Total Sources: {results['execution_summary']['total_sources']}")

    # Run examples
    async def main():
        print("🔬 Research Workflow Examples")
        print("=" * 50)

        print("\n1. Quick Research Example:")
        await example_quick_research()

        print("\n2. Comprehensive Research Example:")
        await example_comprehensive_research()

    # Execute examples
    asyncio.run(main())
