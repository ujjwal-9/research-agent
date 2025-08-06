#!/usr/bin/env python3
"""
Research workflow execution script.

This script provides a command-line interface for running research workflows
with the multi-agent system.

Usage:
    python scripts/run_research_workflow.py "How does vector database indexing work?"
    python scripts/run_research_workflow.py "Compare different embedding models" --detailed
    python scripts/run_research_workflow.py "Best practices for ML pipelines" --collection semantic_redesign
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from research_workflow import ResearchWorkflowManager, setup_logging


async def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Execute a research workflow using the multi-agent system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_research_workflow.py "How does semantic chunking work?"
    python scripts/run_research_workflow.py "Compare vector databases" --detailed --format both
    python scripts/run_research_workflow.py "ML pipeline best practices" --collection custom_collection
    python scripts/run_research_workflow.py "AI in healthcare" --interactive --validate
    python scripts/run_research_workflow.py "Vector database indexing" --non-interactive --detailed
    python scripts/run_research_workflow.py "Research trends" --no-code-interpreter
        """,
    )

    parser.add_argument("research_question", help="Research question to investigate")

    parser.add_argument(
        "--collection",
        default=None,
        help="Qdrant collection name to search (default: from environment)",
    )

    parser.add_argument(
        "--format",
        choices=["markdown", "json", "both"],
        default="markdown",
        help="Output format for the research report (default: markdown)",
    )

    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Generate detailed analysis with comprehensive coverage",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate environment before running research",
    )

    parser.add_argument(
        "--output-dir",
        default="docs",
        help="Output directory for reports (default: docs)",
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        default=True,
        help="Enable interactive mode with clarification questions and plan approval (default: enabled)",
    )

    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Disable interactive mode for automated execution",
    )

    parser.add_argument(
        "--no-code-interpreter",
        action="store_true",
        help="Disable Excel Code Interpreter analysis (default: enabled)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("🔬 Research Workflow Script Starting")
    logger.info(f"📝 Research Question: {args.research_question}")

    try:
        # Determine interactive mode
        interactive_mode = not args.non_interactive

        # Initialize workflow manager
        manager = ResearchWorkflowManager(args.collection, interactive_mode)

        if interactive_mode:
            print("🔬 Interactive Research Workflow Mode")
            print(
                "   You'll be asked clarifying questions and to approve the research plan."
            )
            print("   Use --non-interactive flag to disable this.\n")

        # Validate environment if requested
        if args.validate:
            logger.info("🔍 Validating environment...")
            validation = await manager.validate_environment()

            print("\n🔧 Environment Validation Results:")
            print("=" * 50)

            for check, result in validation.items():
                if isinstance(result, bool):
                    status = "✅" if result else "❌"
                    print(f"{status} {check}: {result}")
                elif isinstance(result, (int, str)):
                    print(f"ℹ️  {check}: {result}")

            # Check if critical components are working
            critical_checks = ["qdrant_connection", "openai_connection"]
            if not all(validation.get(check, False) for check in critical_checks):
                logger.error("❌ Critical environment validation failed!")
                print(
                    "\n❌ Environment validation failed. Please check your configuration."
                )
                return 1

            print("\n✅ Environment validation passed!")

        # Prepare user requirements for detailed analysis
        user_requirements = {}
        if args.detailed:
            user_requirements = {
                "depth_preference": "detailed",
                "comprehensive_analysis": True,
                "include_methodology": True,
                "extended_sources": True,
            }

        # Execute research workflow
        logger.info("🚀 Starting research workflow execution...")
        print(f"\n🔬 Conducting research on: {args.research_question}")
        print("⏳ This may take a few minutes...")

        # Show code interpreter status
        code_status = "enabled" if not args.no_code_interpreter else "disabled"
        print(f"🔧 Code Interpreter: {code_status}")
        logger.info(f"🔧 Code Interpreter: {code_status}")

        results = await manager.conduct_research(
            research_question=args.research_question,
            user_requirements=user_requirements,
            export_format=args.format,
            use_code_interpreter=not args.no_code_interpreter,
        )

        # Display results summary
        print("\n📊 Research Workflow Completed!")
        print("=" * 50)

        summary = results["execution_summary"]
        print(f"⏱️  Total Execution Time: {summary['total_execution_time']:.1f} seconds")
        print(f"🤖 Agents Executed: {summary['agents_executed']}")
        print(f"💬 User Interactions: {summary['user_interactions']}")
        print(f"📄 Report Sections: {summary['report_sections']}")
        print(f"📚 Total Sources: {summary['total_sources']}")

        # Show export information
        print("\n📁 Generated Reports:")
        for format_type, content in results["exported_reports"].items():
            if format_type.endswith("_file"):
                file_path = content
                file_size = (
                    os.path.getsize(file_path) if os.path.exists(file_path) else 0
                )
                print(
                    f"📄 {format_type.replace('_file', '').upper()}: {file_path} ({file_size:,} bytes)"
                )

        # Display comprehensive answer if available
        workflow_result = results.get("workflow_result")
        if (
            workflow_result
            and hasattr(workflow_result, "comprehensive_answer")
            and workflow_result.comprehensive_answer
        ):
            print("\n🎯 Comprehensive Research Answer:")
            print("=" * 60)
            print(workflow_result.comprehensive_answer)
            print("=" * 60)

        # Display brief preview of markdown report if available
        if "markdown" in results["exported_reports"]:
            markdown_content = results["exported_reports"]["markdown"]
            preview_length = 500

            print(f"\n📖 Full Report Preview (first {preview_length} characters):")
            print("-" * 50)
            print(markdown_content[:preview_length])
            if len(markdown_content) > preview_length:
                print("...")
                print(
                    f"\n📄 Full report available in: {results['exported_reports'].get('markdown_file', 'generated file')}"
                )

        logger.info("✅ Research workflow completed successfully")
        print("\n✅ Research workflow completed successfully!")

        return 0

    except KeyboardInterrupt:
        logger.info("⚠️ Research workflow interrupted by user")
        print("\n⚠️ Research workflow interrupted by user")
        return 1

    except Exception as e:
        logger.error(f"❌ Research workflow failed: {e}")
        print(f"\n❌ Research workflow failed: {e}")

        if args.log_level == "DEBUG":
            import traceback

            print("\n🐛 Debug traceback:")
            traceback.print_exc()

        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
