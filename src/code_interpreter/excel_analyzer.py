"""Excel Analyzer using CrewAI CodeInterpreterTool."""

import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from crewai import Agent, Task, Crew, Process
from crewai_tools import CodeInterpreterTool

from .config import CodeInterpreterConfig


class ExcelAnalyzer:
    """Excel file analyzer using CrewAI's CodeInterpreterTool."""

    def __init__(self, config: Optional[CodeInterpreterConfig] = None):
        """Initialize the Excel Analyzer.

        Args:
            config: Configuration object for the code interpreter
        """
        self.config = config or CodeInterpreterConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize the code interpreter tool
        self.code_interpreter = CodeInterpreterTool(
            user_dockerfile_path=self.config.user_dockerfile_path,
            user_docker_base_url=self.config.user_docker_base_url,
            unsafe_mode=self.config.unsafe_mode,
            default_image_tag=self.config.default_image_tag,
        )

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Initialize the data analyst agent
        self.data_analyst = Agent(
            role="Excel Data Analyst",
            goal="Analyze Excel files and provide insights based on user queries",
            backstory="""You are an expert data analyst specializing in Excel file analysis. 
            You can read, process, and analyze data from Excel files to answer complex questions. 
            You excel at data visualization, statistical analysis, and extracting meaningful 
            insights from spreadsheet data. You always provide clear, actionable results 
            with proper visualizations when needed.""",
            tools=[self.code_interpreter],
            verbose=self.config.verbose,
            allow_code_execution=True,
        )

    def find_excel_files(self, directory: str) -> List[str]:
        """Find all Excel files in the given directory.

        Args:
            directory: Directory path to search for Excel files

        Returns:
            List of Excel file paths
        """
        excel_files = []
        for ext in self.config.supported_formats:
            pattern = os.path.join(directory, f"**/*{ext}")
            excel_files.extend(glob.glob(pattern, recursive=True))

        # Filter by file size
        filtered_files = []
        for file_path in excel_files:
            try:
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                if file_size_mb <= self.config.max_file_size_mb:
                    filtered_files.append(file_path)
                else:
                    self.logger.warning(
                        f"Skipping {file_path}: file size {file_size_mb:.1f}MB exceeds limit"
                    )
            except OSError:
                self.logger.warning(f"Could not access file: {file_path}")

        return filtered_files

    def analyze_excel_files(
        self,
        excel_files: List[str],
        user_query: str,
        additional_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyze Excel files based on user query.

        Args:
            excel_files: List of Excel file paths to analyze
            user_query: User's question or analysis request
            additional_context: Additional context or constraints

        Returns:
            Dictionary containing analysis results
        """
        if not excel_files:
            return {"error": "No Excel files provided for analysis", "result": None}

        # Prepare file list for the task description
        file_list = "\n".join([f"- {os.path.basename(f)}: {f}" for f in excel_files])

        # Build the task description
        task_description = f"""
        Analyze the following Excel files to answer the user's query:
        
        Files to analyze:
        {file_list}
        
        User Query: {user_query}
        
        {f"Additional Context: {additional_context}" if additional_context else ""}
        
        Requirements:
        1. Load and examine all the Excel files using pandas
        2. Understand the structure and content of each file
        3. Analyze the data to answer the user's specific query
        4. Provide clear, actionable insights
        5. Include data statistics, trends, or patterns that are relevant
        6. Handle missing data or inconsistencies gracefully
        
        Libraries you might need: pandas, matplotlib, seaborn, numpy, openpyxl
        
        Make sure to:
        - Print step-by-step analysis progress
        - Show sample data from each file
        - Provide summary statistics where relevant
        - Create visualizations that support your findings
        - Give a clear final answer to the user's query
        """

        # Create the analysis task
        analysis_task = Task(
            description=task_description,
            expected_output=f"""
            A comprehensive analysis report including:
            1. Overview of the Excel files and their contents
            2. Direct answer to the user query: "{user_query}"
            3. Supporting data analysis and statistics
            4. Key insights and recommendations
            5. Any limitations or assumptions made during analysis
            """,
            agent=self.data_analyst,
        )

        # Create and run the crew
        crew = Crew(
            agents=[self.data_analyst],
            tasks=[analysis_task],
            verbose=self.config.verbose,
            process=Process.sequential,
        )

        try:
            self.logger.info(f"Starting analysis of {len(excel_files)} Excel files")
            result = crew.kickoff()

            return {
                "success": True,
                "result": str(result),
                "files_analyzed": excel_files,
                "output_directory": self.config.output_dir,
            }

        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            return {"success": False, "error": str(e), "result": None}

    def analyze_directory(
        self, directory: str, user_query: str, additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze all Excel files in a directory based on user query.

        Args:
            directory: Directory containing Excel files
            user_query: User's question or analysis request
            additional_context: Additional context or constraints

        Returns:
            Dictionary containing analysis results
        """
        if not os.path.exists(directory):
            return {
                "success": False,
                "error": f"Directory does not exist: {directory}",
                "result": None,
            }

        excel_files = self.find_excel_files(directory)

        if not excel_files:
            return {
                "success": False,
                "error": f"No Excel files found in directory: {directory}",
                "result": None,
            }

        self.logger.info(f"Found {len(excel_files)} Excel files in {directory}")

        return self.analyze_excel_files(excel_files, user_query, additional_context)

    def get_file_summary(self, excel_files: List[str]) -> Dict[str, Any]:
        """Get a summary of Excel files without full analysis.

        Args:
            excel_files: List of Excel file paths

        Returns:
            Dictionary containing file summaries
        """
        summary_task_description = f"""
        Provide a quick summary of the following Excel files:
        
        Files to examine:
        {chr(10).join([f"- {os.path.basename(f)}: {f}" for f in excel_files])}
        
        For each file, provide:
        1. File name and size
        2. Number of sheets
        3. Column names and data types for each sheet
        4. Number of rows in each sheet
        5. Brief description of the data content
        
        Use pandas to read the files. Handle any errors gracefully.
        Print the summary in a clear, organized format.
        """

        summary_task = Task(
            description=summary_task_description,
            expected_output="A structured summary of all Excel files including their contents and metadata",
            agent=self.data_analyst,
        )

        crew = Crew(
            agents=[self.data_analyst],
            tasks=[summary_task],
            verbose=self.config.verbose,
            process=Process.sequential,
        )

        try:
            result = crew.kickoff()
            return {
                "success": True,
                "summary": str(result),
                "files_examined": excel_files,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "summary": None}
