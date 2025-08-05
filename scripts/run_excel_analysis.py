#!/usr/bin/env python3
"""Convenience script to run Excel analysis using CrewAI CodeInterpreter."""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from code_interpreter.main import main

if __name__ == "__main__":
    main()
