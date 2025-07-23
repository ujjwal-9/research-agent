#!/usr/bin/env python3
"""Setup script for the research system."""

import os
import sys
from pathlib import Path


def setup_environment():
    """Set up the development environment."""
    print("Setting up research system environment...")

    # Create necessary directories
    directories = ["qdrant_storage", "logs", "data/processed", "tests/fixtures"]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

    # Check if .env exists
    if not Path(".env").exists():
        print(
            "⚠️  .env file not found. Please copy .env.example to .env and configure your API keys."
        )
        return False

    print("✓ Environment setup complete!")
    return True


def check_dependencies():
    """Check if all required dependencies are installed."""
    print("Checking dependencies...")

    required_packages = [
        ("openai", "openai"),
        ("langchain", "langchain"),
        ("qdrant-client", "qdrant_client"),
        ("fastapi", "fastapi"),
        ("python-docx", "docx"),
        ("openpyxl", "openpyxl"),
        ("python-pptx", "pptx"),
        ("PyPDF2", "PyPDF2"),
        ("pillow", "PIL"),
        ("pytesseract", "pytesseract"),
        ("duckduckgo-search", "duckduckgo_search"),
    ]

    missing_packages = []

    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✓ {package_name}")
        except ImportError:
            missing_packages.append(package_name)
            print(f"✗ {package_name}")

    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: uv pip install -r requirements.txt")
        return False

    print("✓ All dependencies installed!")
    return True


if __name__ == "__main__":
    success = setup_environment() and check_dependencies()

    if success:
        print("\n🎉 Setup complete! You can now:")
        print("1. Configure your .env file with API keys")
        print("2. Run document ingestion: python -m src.ingestion.ingest_documents")
        print(
            "3. Start the research system: python -m src.main research -q 'your query'"
        )
        print("4. Start the API server: python -m src.main serve")
    else:
        print("\n❌ Setup failed. Please fix the issues above.")
        sys.exit(1)
