"""Configuration for the Code Interpreter module."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CodeInterpreterConfig:
    """Configuration for the CodeInterpreter tool."""

    # Docker configuration
    user_dockerfile_path: Optional[str] = None
    user_docker_base_url: Optional[str] = None
    default_image_tag: str = "code-interpreter:latest"

    # Safety configuration
    unsafe_mode: bool = False

    # Agent configuration
    verbose: bool = True
    max_execution_time: int = 300  # seconds

    # Data processing configuration
    max_file_size_mb: int = 100
    supported_formats: tuple = (".xlsx", ".xls", ".csv")

    # Output configuration
    output_dir: str = "outputs"
    save_plots: bool = True
    save_results: bool = True
