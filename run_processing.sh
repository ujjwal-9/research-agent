#!/bin/bash

# Excel Processing Script with Environment Setup
# This script sets up the environment and runs the Excel processing

# Set PYTHONPATH to include current directory
export PYTHONPATH=".:$PYTHONPATH"

# Load .env file if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    set -o allexport
    source .env
    set +o allexport
else
    echo "Warning: .env file not found. Please create one with your OpenAI API key."
fi

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY not set. AI features will use fallback methods."
else
    echo "✓ OpenAI API key found"
fi

# Display configuration
echo "Configuration:"
echo "  Model: ${OPENAI_EXCEL_PROCESSING_MODEL:-gpt-4-turbo-preview (default)}"
echo "  Header check rows: ${HEADER_CHECK_ROWS:-10 (default)}"
echo ""

# Run the processing script
echo "Starting Excel processing..."
uv run python scripts/process_excel_files.py

echo "Processing completed!"