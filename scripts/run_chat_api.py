#!/usr/bin/env python3
"""
Startup script for the chat API server.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def check_environment():
    """Check if required environment variables are set."""
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = []

    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        logger.error(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )
        logger.error("Please set these in your .env file or environment")
        return False

    logger.info("Environment variables check passed")
    return True


def main():
    """Start the chat API server."""
    logger.info("Starting Chat API Server...")

    # Check environment
    if not check_environment():
        sys.exit(1)

    # Load environment variables from .env file if it exists
    try:
        from dotenv import load_dotenv

        load_dotenv()
        logger.info("Loaded environment variables from .env file")
    except ImportError:
        logger.warning(
            "python-dotenv not available, using system environment variables"
        )

    # Import and run the app
    try:
        import uvicorn
        from src.app.main import app

        logger.info("Starting server on http://0.0.0.0:8000")
        logger.info("API Documentation available at: http://localhost:8000/docs")
        logger.info("WebSocket endpoint: ws://localhost:8000/ws")

        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            reload=True,  # Enable auto-reload for development
        )

    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error(
            "Make sure all dependencies are installed: uv add fastapi uvicorn websockets"
        )
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
