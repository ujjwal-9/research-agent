#!/usr/bin/env python3
"""
Script to run the FastAPI application.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import uvicorn


def main():
    """Run the FastAPI application."""

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return

    print("🚀 Starting Research Chat API...")
    print("📡 WebSocket endpoint: ws://localhost:8000/ws/{connection_id}")
    print("🌐 REST API docs: http://localhost:8000/docs")
    print("💡 Health check: http://localhost:8000/health")

    # Run the application
    uvicorn.run(
        "src.app.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )


if __name__ == "__main__":
    main()
