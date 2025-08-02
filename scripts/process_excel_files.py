"""
Script to process Excel files from data/documents folder.

This script provides a simple interface to run the Excel processing pipeline.
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.preprocessing.excel_processor import ExcelProcessor


def main():
    """Main function to process Excel files."""

    print("=" * 80)
    print("EXCEL FILE PROCESSOR")
    print("=" * 80)
    print()

    # Get OpenAI API key from environment or prompt
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠ OpenAI API key not found in environment variable 'OPENAI_API_KEY'")
        print("  Knowledge extraction will use fallback method.")
        print("  To use OpenAI GPT-4.1, set the OPENAI_API_KEY environment variable.")
        print()
    else:
        print("✓ OpenAI API key found. Will use GPT-4.1 for knowledge extraction.")
        print()

    # Define directories
    input_dir = Path("data/documents")
    output_dir = Path("data/processed_sheet")
    log_dir = Path("logs")

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Log directory: {log_dir}")
    print()

    if not input_dir.exists():
        print(f"❌ Error: Input directory '{input_dir}' does not exist!")
        print("Please ensure you have Excel files in the data/documents folder.")
        return 1

    try:
        # Initialize processor
        print("Initializing Excel processor...")
        processor = ExcelProcessor(openai_api_key=api_key, log_dir=log_dir)

        # Process files
        print("Starting processing...")
        print("-" * 40)

        results = processor.process_directory(input_dir, output_dir)

        # Generate report
        report_path = log_dir / "processing_report.txt"
        report = processor.generate_processing_report(results, report_path)

        print()
        print("=" * 80)
        print("PROCESSING COMPLETED")
        print("=" * 80)
        print(report)

        # Summary
        if results["successful"] > 0:
            print("✅ Processing completed successfully!")
            print(f"📁 Check output files in: {output_dir}")
            print(f"📋 Full report saved to: {report_path}")
        else:
            print("⚠ No files were processed successfully.")
            print("Check the logs for error details.")

        return 0

    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
