"""Code execution engine for structured data analysis using CrewAI's CodeInterpreterTool with Docker."""

import pandas as pd
import numpy as np
import openpyxl
import json
import traceback
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from openai import OpenAI
from loguru import logger
import tempfile
import warnings

warnings.filterwarnings("ignore")

from src.config import settings

# CrewAI imports
try:
    from crewai_tools import CodeInterpreterTool
    from crewai import Agent, Task, Crew, Process

    CREWAI_AVAILABLE = True
except ImportError:
    logger.warning("CrewAI not available. Install with: pip install 'crewai[tools]'")
    CREWAI_AVAILABLE = False


class CrewAICodeExecutor:
    """Executes LLM-generated Python code using CrewAI's secure Docker environment."""

    def __init__(self):
        self.openai_client = OpenAI(api_key=settings.openai_api_key)

        if CREWAI_AVAILABLE:
            # Initialize CrewAI code interpreter with Docker (preferred)
            self.code_interpreter = CodeInterpreterTool()

            # Create a specialized agent for data analysis
            self.data_analyst = Agent(
                role="Data Analysis Specialist",
                goal="Execute Python code for precise data analysis and return structured results",
                backstory="""You are an expert data analyst who specializes in executing 
                Python code for Excel data analysis. You write clean, efficient code that 
                handles edge cases and returns well-structured results. You always ensure 
                the final result is stored in a 'result' variable as a dictionary.""",
                tools=[self.code_interpreter],
                verbose=False,
                allow_delegation=False,
            )
        else:
            self.code_interpreter = None
            self.data_analyst = None
            logger.error("CrewAI not available - code execution will be limited")

    async def analyze_with_code(
        self,
        query: str,
        file_path: Path,
        schema: Dict[str, Any],
        additional_context: str = "",
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """Generate and execute code to analyze data based on query with retry logic."""

        if not CREWAI_AVAILABLE:
            return {
                "success": False,
                "error": "CrewAI not available. Please install with: pip install 'crewai[tools]'",
                "query": query,
            }

        try:
            # Generate Python code for analysis
            code = await self._generate_analysis_code(
                query, file_path, schema, additional_context
            )

            # Execute with retry logic
            result = await self._execute_code_with_retries(code, file_path, max_retries)

            return {
                "success": True,
                "query": query,
                "generated_code": code,
                "execution_result": result,
                "file_analyzed": str(file_path),
            }

        except Exception as e:
            logger.error(f"Code analysis failed: {e}")
            return {"success": False, "error": str(e), "query": query}

    async def _execute_code_with_retries(
        self, code: str, file_path: Path, max_retries: int
    ) -> Dict[str, Any]:
        """Execute code with retry logic using different strategies."""

        last_error = None
        execution_strategies = [
            ("docker", False),  # Docker container (recommended)
            ("sandbox", False),  # Sandbox environment
            (
                ("unsafe", True) if max_retries >= 3 else None
            ),  # Unsafe mode as last resort
        ]

        # Filter out None strategies
        execution_strategies = [s for s in execution_strategies if s is not None]

        for attempt in range(max_retries):
            try:
                strategy_name, unsafe_mode = execution_strategies[
                    min(attempt, len(execution_strategies) - 1)
                ]

                logger.info(
                    f"Execution attempt {attempt + 1}/{max_retries} using {strategy_name} mode"
                )

                # Prepare the execution task
                task_description = self._create_execution_task(
                    code, file_path, strategy_name
                )

                # Create task for code execution
                execution_task = Task(
                    description=task_description,
                    expected_output="""A JSON object containing the analysis results with the following structure:
                    {
                        "result": {...},  // The main analysis result
                        "summary": "...", // Summary of findings
                        "method": "...",  // Method used for analysis
                        "data_points": [...] // Relevant data points
                    }""",
                    agent=self.data_analyst,
                )

                # Execute the task
                crew = Crew(
                    agents=[self.data_analyst],
                    tasks=[execution_task],
                    verbose=False,
                    process=Process.sequential,
                )

                # Run with timeout
                result = await asyncio.wait_for(
                    asyncio.to_thread(crew.kickoff), timeout=120  # 2 minute timeout
                )

                # Parse and validate result
                parsed_result = self._parse_execution_result(result, strategy_name)

                if parsed_result["execution_success"]:
                    logger.info(f"Code execution successful using {strategy_name} mode")
                    return parsed_result
                else:
                    last_error = parsed_result.get(
                        "error_message", "Unknown execution error"
                    )
                    logger.warning(
                        f"Execution failed with {strategy_name} mode: {last_error}"
                    )

            except asyncio.TimeoutError:
                last_error = f"Code execution timeout after 120 seconds"
                logger.error(last_error)

            except Exception as e:
                last_error = str(e)
                logger.error(f"Execution attempt {attempt + 1} failed: {last_error}")

                # If it's a Docker-related error, try next strategy immediately
                if "docker" in str(e).lower() and attempt < max_retries - 1:
                    logger.info("Docker error detected, trying next execution strategy")
                    continue

        # All retries failed
        return {
            "result": None,
            "output": "",
            "errors": f"All {max_retries} execution attempts failed. Last error: {last_error}",
            "execution_success": False,
            "error_message": last_error,
            "execution_strategy": "failed",
        }

    def _create_execution_task(self, code: str, file_path: Path, strategy: str) -> str:
        """Create task description for code execution."""

        # Make file path relative to working directory for security
        relative_path = (
            f"'{file_path.name}'"
            if file_path.name in str(file_path)
            else f"'{file_path}'"
        )

        task_description = f"""
Execute the following Python code to analyze the Excel file and return structured results.

IMPORTANT REQUIREMENTS:
1. The code must work with the file path: {relative_path}
2. Handle any missing libraries by installing them if needed
3. Store the final result in a variable called 'result' as a dictionary
4. Include error handling for common issues (file not found, data format problems)
5. Return meaningful results even if some analysis steps fail

EXECUTION MODE: {strategy}

PYTHON CODE TO EXECUTE:
```python
{code}

# Ensure we have a result variable with structured output
if 'result' not in locals():
    result = {{
        "answer": "Analysis completed",
        "summary": "Code executed successfully",
        "method": "Data analysis",
        "execution_mode": "{strategy}"
    }}

# Print the result for output capture
print("EXECUTION RESULT:")
print(json.dumps(result, indent=2, default=str))
```

Execute this code and return the analysis results. If the execution fails, provide error details and attempt to return partial results if possible.
"""
        return task_description

    def _parse_execution_result(
        self, crew_result: Any, strategy: str
    ) -> Dict[str, Any]:
        """Parse and validate the execution result from CrewAI."""

        try:
            # Convert crew result to string if needed
            if hasattr(crew_result, "raw"):
                output_text = str(crew_result.raw)
            else:
                output_text = str(crew_result)

            # Look for JSON result in the output
            result_data = None

            # Try to extract JSON from the output
            if "EXECUTION RESULT:" in output_text:
                json_start = output_text.find("EXECUTION RESULT:") + len(
                    "EXECUTION RESULT:"
                )
                json_text = output_text[json_start:].strip()

                # Try to parse JSON
                try:
                    # Look for JSON object
                    import re

                    json_match = re.search(r"\{.*\}", json_text, re.DOTALL)
                    if json_match:
                        result_data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON from execution result")

            # If no structured result found, create one from output
            if not result_data:
                result_data = {
                    "answer": "Code executed, see output for details",
                    "summary": (
                        output_text[:200] + "..."
                        if len(output_text) > 200
                        else output_text
                    ),
                    "method": f"Code execution via {strategy}",
                    "raw_output": output_text,
                }

            return {
                "result": result_data,
                "output": output_text,
                "errors": None,
                "execution_success": True,
                "execution_strategy": strategy,
            }

        except Exception as e:
            error_msg = f"Failed to parse execution result: {e}"
            logger.error(error_msg)

            return {
                "result": None,
                "output": str(crew_result) if crew_result else "",
                "errors": error_msg,
                "execution_success": False,
                "error_message": error_msg,
                "execution_strategy": strategy,
            }

    async def _generate_analysis_code(
        self,
        query: str,
        file_path: Path,
        schema: Dict[str, Any],
        additional_context: str = "",
    ) -> str:
        """Generate Python code for data analysis using LLM."""

        prompt = self._build_code_generation_prompt(
            query, file_path, schema, additional_context
        )

        try:
            response = self.openai_client.chat.completions.create(
                model=settings.llm_extraction_model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert Python data analyst. Generate clean, executable Python code for Excel data analysis.

REQUIREMENTS:
1. Always use pandas for data manipulation
2. Handle missing values and edge cases
3. Include proper error handling with try-except blocks
4. Return results as structured data (dict/list)
5. Use descriptive variable names
6. Add comments explaining key steps
7. NEVER use matplotlib or plotting libraries (CrewAI environment limitation)
8. Focus on data extraction and analysis, not visualization
9. Always return a final 'result' variable containing the answer as a dictionary

SAFETY:
- Only use safe libraries: pandas, numpy, openpyxl, datetime, json
- Include error handling for file not found and data format issues
- Make code robust to handle different Excel structures
- No subprocess calls or system commands

CODE STRUCTURE:
```python
import pandas as pd
import numpy as np
import json
from pathlib import Path

try:
    # Load and analyze data
    # ... your analysis code ...
    
    # Final result as dictionary
    result = {
        "answer": "direct answer to query",
        "data": [...],  # relevant data
        "summary": "brief summary",
        "method": "analysis method used"
    }
    
except Exception as e:
    result = {
        "error": str(e),
        "answer": "Analysis failed",
        "summary": f"Error occurred: {e}"
    }
```""",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )

            code = response.choices[0].message.content

            # Extract code from markdown if present
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0].strip()
            elif "```" in code:
                code = code.split("```")[1].split("```")[0].strip()

            return code

        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            raise

    def _build_code_generation_prompt(
        self,
        query: str,
        file_path: Path,
        schema: Dict[str, Any],
        additional_context: str = "",
    ) -> str:
        """Build prompt for code generation."""

        schema_info = json.dumps(schema, indent=2)

        prompt = f"""
Generate Python code to analyze Excel data and answer this query:

QUERY: {query}

FILE INFORMATION:
- File path: {file_path.name} (use this exact filename)
- Full path for reference: {file_path}

SCHEMA INFORMATION:
{schema_info}

ADDITIONAL CONTEXT:
{additional_context}

INSTRUCTIONS:
1. Load the Excel file using pandas (try multiple methods if needed)
2. Handle multiple sheets if present based on schema information
3. Apply appropriate filtering, aggregation, or analysis based on the query
4. Use the schema information to understand column meanings and data types
5. Handle edge cases and missing data gracefully
6. Return structured results that directly answer the query
7. Include error handling for common issues

IMPORTANT: 
- Use filename '{file_path.name}' in your code
- Make the code robust to handle different Excel file structures
- Always return a 'result' dictionary with the analysis findings
- Include try-except blocks to handle errors gracefully

Generate ONLY the Python code, no explanations outside the code comments.
Ensure the code is complete and executable in a Docker environment.
"""
        return prompt

    async def analyze_multiple_queries(
        self,
        queries: List[str],
        file_path: Path,
        schema: Dict[str, Any],
        additional_context: str = "",
    ) -> List[Dict[str, Any]]:
        """Analyze multiple queries efficiently."""

        results = []

        for query in queries:
            logger.info(f"Analyzing query: {query}")
            result = await self.analyze_with_code(
                query, file_path, schema, additional_context
            )
            results.append(result)

        return results

    def get_suggested_queries(self, schema: Dict[str, Any]) -> List[str]:
        """Generate suggested queries based on schema analysis."""

        suggested_queries = []

        # Extract suggested queries from schema
        for sheet_name, sheet_schema in schema.get("schemas", {}).items():
            analysis_queries = sheet_schema.get("analysis_queries", [])
            suggested_queries.extend(analysis_queries)

        # Add generic queries based on data types
        for sheet_name, sheet_schema in schema.get("schemas", {}).items():
            tables = sheet_schema.get("tables", [])

            for table in tables:
                columns = table.get("columns", [])

                # Find numeric columns for aggregation
                numeric_cols = [
                    col["name"] for col in columns if col.get("data_type") == "number"
                ]
                if numeric_cols:
                    for col in numeric_cols[:2]:  # Limit to avoid too many suggestions
                        suggested_queries.append(f"What is the average {col}?")
                        suggested_queries.append(f"What is the total {col}?")

                # Find categorical columns for grouping
                categorical_cols = [
                    col["name"] for col in columns if col.get("data_type") == "text"
                ]
                if categorical_cols and numeric_cols:
                    cat_col = categorical_cols[0]
                    num_col = numeric_cols[0]
                    suggested_queries.append(f"Show {num_col} grouped by {cat_col}")

                # Date columns for time series
                date_cols = [
                    col["name"] for col in columns if col.get("data_type") == "date"
                ]
                if date_cols:
                    suggested_queries.append(
                        f"Show trends over time for {date_cols[0]}"
                    )

        return suggested_queries[:10]  # Limit suggestions


class ExcelDataAnalyzer:
    """High-level interface for Excel data analysis combining schema detection and CrewAI code execution."""

    def __init__(self):
        from src.tools.schema_detector import SchemaDetector

        self.schema_detector = SchemaDetector()
        self.code_executor = CrewAICodeExecutor()

    async def analyze_excel_with_query(
        self,
        file_path: Path,
        query: str,
        context_pdfs: List[str] = None,
        web_source_url: str = None,
        additional_context: str = "",
    ) -> Dict[str, Any]:
        """Complete Excel analysis pipeline: schema detection + CrewAI code execution."""

        try:
            # Step 1: Detect schema
            logger.info(f"Detecting schema for {file_path}")
            schema_result = await self.schema_detector.detect_sheet_schemas(
                file_path, context_pdfs, web_source_url
            )

            if "error" in schema_result:
                return {"error": f"Schema detection failed: {schema_result['error']}"}

            # Step 2: Execute code analysis with CrewAI
            logger.info(f"Executing CrewAI code analysis for query: {query}")
            analysis_result = await self.code_executor.analyze_with_code(
                query, file_path, schema_result, additional_context
            )

            # Combine results
            return {
                "query": query,
                "file_path": str(file_path),
                "schema": schema_result,
                "analysis": analysis_result,
                "suggested_queries": self.code_executor.get_suggested_queries(
                    schema_result
                ),
                "execution_method": "CrewAI Docker",
            }

        except Exception as e:
            logger.error(f"Excel analysis failed: {e}")
            return {"error": str(e)}

    async def batch_analyze(
        self,
        file_path: Path,
        queries: List[str],
        context_pdfs: List[str] = None,
        web_source_url: str = None,
    ) -> Dict[str, Any]:
        """Analyze multiple queries against the same Excel file using CrewAI."""

        # Detect schema once
        schema_result = await self.schema_detector.detect_sheet_schemas(
            file_path, context_pdfs, web_source_url
        )

        if "error" in schema_result:
            return {"error": f"Schema detection failed: {schema_result['error']}"}

        # Execute multiple analyses
        analysis_results = await self.code_executor.analyze_multiple_queries(
            queries, file_path, schema_result
        )

        return {
            "file_path": str(file_path),
            "schema": schema_result,
            "analyses": analysis_results,
            "total_queries": len(queries),
            "execution_method": "CrewAI Docker",
        }


# Backward compatibility - keep the same interface
CodeExecutor = CrewAICodeExecutor
