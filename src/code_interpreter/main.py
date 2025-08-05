"""Main script for running Excel analysis with CrewAI CodeInterpreter."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from .excel_analyzer import ExcelAnalyzer
from .config import CodeInterpreterConfig


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("excel_analysis.log"),
        ],
    )


def main():
    """Main function for Excel analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze Excel files using CrewAI CodeInterpreter"
    )

    # Required arguments
    parser.add_argument("query", help="User query or question about the data")

    # Optional arguments
    parser.add_argument(
        "--directory",
        "-d",
        help="Directory containing Excel files (default: current directory)",
        default=".",
    )
    parser.add_argument(
        "--files", "-f", nargs="+", help="Specific Excel files to analyze"
    )
    parser.add_argument(
        "--context", "-c", help="Additional context or constraints for the analysis"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        help="Output directory for results and plots",
        default="outputs",
    )
    parser.add_argument(
        "--unsafe-mode",
        action="store_true",
        help="Enable unsafe mode for code execution (not recommended)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only provide file summaries without full analysis",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Create configuration
    config = CodeInterpreterConfig(
        unsafe_mode=args.unsafe_mode, verbose=args.verbose, output_dir=args.output_dir
    )

    # Initialize analyzer
    analyzer = ExcelAnalyzer(config)

    try:
        if args.files:
            # Analyze specific files
            logger.info(f"Analyzing specific files: {args.files}")

            if args.summary_only:
                result = analyzer.get_file_summary(args.files)
            else:
                result = analyzer.analyze_excel_files(
                    args.files, args.query, args.context
                )
        else:
            # Analyze directory
            logger.info(f"Analyzing Excel files in directory: {args.directory}")

            if args.summary_only:
                excel_files = analyzer.find_excel_files(args.directory)
                result = analyzer.get_file_summary(excel_files)
            else:
                result = analyzer.analyze_directory(
                    args.directory, args.query, args.context
                )

        # Display results
        if result.get("success", False):
            print("\n" + "=" * 80)
            print("ANALYSIS RESULTS")
            print("=" * 80)

            if args.summary_only:
                print(result.get("summary", "No summary available"))
            else:
                print(result.get("result", "No result available"))

                if "files_analyzed" in result:
                    print(f"\nFiles analyzed: {len(result['files_analyzed'])}")
                    for file_path in result["files_analyzed"]:
                        print(f"  - {file_path}")

                if "output_directory" in result:
                    print(f"\nOutputs saved to: {result['output_directory']}")

            print("=" * 80)

        else:
            print(f"\nError: {result.get('error', 'Unknown error occurred')}")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
