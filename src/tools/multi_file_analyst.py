"""
Multi-file data analyst that can work with multiple Excel files simultaneously.
Performs schema-based file selection and supports joins/merging operations.
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from loguru import logger
from dataclasses import dataclass
import pandas as pd
from openai import OpenAI

from src.config import settings
from src.tools.code_executor import CrewAICodeExecutor


@dataclass
class FileRelevanceScore:
    """Relevance score for a file to a given query."""

    file_path: str
    relevance_score: float
    matching_tables: List[str]
    relevant_columns: List[str]
    reasoning: str


@dataclass
class MultiFileAnalysisRequest:
    """Request for multi-file analysis."""

    query: str
    selected_files: List[str]
    analysis_type: str  # "join", "compare", "aggregate", "hybrid"
    join_strategy: Optional[str] = None  # "inner", "outer", "left", "right"
    primary_file: Optional[str] = None


class SchemaBasedFileSelector:
    """Select relevant files based on schema information and query analysis."""

    def __init__(self, schema_file: Path):
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
        self.schemas = self._load_schemas(schema_file)

    def _load_schemas(self, schema_file: Path) -> Dict[str, Any]:
        """Load preprocessed schema information."""
        try:
            with open(schema_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load schemas from {schema_file}: {e}")
            return {}

    async def select_relevant_files(
        self, query: str, max_files: int = 5, min_relevance: float = 0.3
    ) -> List[FileRelevanceScore]:
        """Select files most relevant to the query based on schema analysis."""

        if not self.schemas or "schemas" not in self.schemas:
            logger.warning("No schema information available")
            return []

        # Analyze query to understand what data is needed
        query_analysis = await self._analyze_query_requirements(query)

        # Score each file
        file_scores = []
        for file_path, file_schema in self.schemas["schemas"].items():
            if "error" in file_schema:
                continue

            score = await self._score_file_relevance(
                file_path, file_schema, query, query_analysis
            )

            if score.relevance_score >= min_relevance:
                file_scores.append(score)

        # Sort by relevance and return top files
        file_scores.sort(key=lambda x: x.relevance_score, reverse=True)
        return file_scores[:max_files]

    async def _analyze_query_requirements(self, query: str) -> Dict[str, Any]:
        """Analyze what the query is asking for."""

        prompt = f"""
Analyze this data query and identify what types of data/tables would be needed:

Query: "{query}"

Provide JSON response:
{{
    "data_types_needed": ["financial", "operational", "customer", "product", etc.],
    "key_metrics": ["specific metrics or KPIs mentioned"],
    "time_dimensions": ["any time periods or dates mentioned"],
    "entity_types": ["companies", "people", "products", etc.],
    "analysis_type": "aggregation|comparison|trend|correlation|forecasting",
    "join_candidates": ["fields that might connect different datasets"]
}}
"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data analyst. Analyze queries and identify data requirements.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=500,
                response_format={"type": "json_object"},
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            logger.warning(f"Query analysis failed: {e}")
            return {"data_types_needed": [], "analysis_type": "general"}

    async def _score_file_relevance(
        self,
        file_path: str,
        file_schema: Dict[str, Any],
        query: str,
        query_analysis: Dict[str, Any],
    ) -> FileRelevanceScore:
        """Score how relevant a file is to the query."""

        # Extract file info
        file_name = file_schema.get("file_name", "")
        schemas = file_schema.get("schemas", {})
        business_analysis = file_schema.get("business_analysis", {})

        # Build compact file summary for LLM
        file_summary = self._create_file_summary(file_name, schemas, business_analysis)

        prompt = f"""
Rate relevance of this file to the query (0.0-1.0):

Query: "{query}"

Query needs: {query_analysis.get('data_types_needed', [])}

File: {file_summary}

Provide JSON:
{{
    "relevance_score": 0.0-1.0,
    "matching_tables": ["table names that match query"],
    "relevant_columns": ["column names relevant to query"],
    "reasoning": "why this file is/isn't relevant"
}}
"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data analyst. Rate file relevance to queries.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=300,
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)

            return FileRelevanceScore(
                file_path=file_path,
                relevance_score=result.get("relevance_score", 0.0),
                matching_tables=result.get("matching_tables", []),
                relevant_columns=result.get("relevant_columns", []),
                reasoning=result.get("reasoning", ""),
            )

        except Exception as e:
            logger.warning(f"File scoring failed for {file_path}: {e}")
            return FileRelevanceScore(
                file_path=file_path,
                relevance_score=0.0,
                matching_tables=[],
                relevant_columns=[],
                reasoning=f"Scoring failed: {e}",
            )

    def _create_file_summary(
        self, file_name: str, schemas: Dict[str, Any], business_analysis: Dict[str, Any]
    ) -> str:
        """Create compact file summary for LLM analysis."""

        summary_parts = [f"File: {file_name}"]

        # Add table summaries
        for table_name, schema in schemas.items():
            if not schema.get("has_data", False):
                continue

            headers = schema.get("headers", [])[:8]  # Limit columns
            row_count = schema.get("total_rows", 0)
            summary_parts.append(
                f"Table '{table_name}': {len(headers)} cols, {row_count} rows"
            )
            summary_parts.append(f"Columns: {', '.join(headers)}")

        # Add business context if available
        if business_analysis and "file_purpose" in business_analysis:
            summary_parts.append(f"Purpose: {business_analysis['file_purpose']}")

        return "\n".join(summary_parts)


class MultiFileExcelAnalyst:
    """Main analyst that can work with multiple Excel files simultaneously."""

    def __init__(self, schema_file: Path):
        self.file_selector = SchemaBasedFileSelector(schema_file)
        self.code_executor = CrewAICodeExecutor()
        self.openai_client = OpenAI(api_key=settings.openai_api_key)

    async def analyze_multi_file_query(
        self,
        query: str,
        data_directory: Path,
        max_files: int = 5,
        analysis_type: str = "auto",  # "auto", "join", "compare", "aggregate"
    ) -> Dict[str, Any]:
        """Analyze query using multiple relevant files."""

        try:
            # Step 1: Select relevant files
            logger.info(f"Selecting relevant files for query: {query}")
            relevant_files = await self.file_selector.select_relevant_files(
                query, max_files=max_files
            )

            if not relevant_files:
                return {"error": "No relevant files found for the query"}

            logger.info(f"Selected {len(relevant_files)} relevant files")

            # Step 2: Determine analysis strategy
            analysis_request = await self._plan_multi_file_analysis(
                query, relevant_files, analysis_type
            )

            # Step 3: Execute multi-file analysis
            result = await self._execute_multi_file_analysis(
                analysis_request, data_directory
            )

            return {
                "query": query,
                "selected_files": [f.file_path for f in relevant_files],
                "file_relevance_scores": [
                    {
                        "file": f.file_path,
                        "score": f.relevance_score,
                        "reasoning": f.reasoning,
                    }
                    for f in relevant_files
                ],
                "analysis_type": analysis_request.analysis_type,
                "result": result,
            }

        except Exception as e:
            logger.error(f"Multi-file analysis failed: {e}")
            return {"error": str(e)}

    async def _plan_multi_file_analysis(
        self, query: str, relevant_files: List[FileRelevanceScore], analysis_type: str
    ) -> MultiFileAnalysisRequest:
        """Plan how to analyze multiple files together."""

        if analysis_type == "auto":
            # Determine best analysis type based on query and files
            analysis_type = await self._determine_analysis_type(query, relevant_files)

        # Select primary file (most relevant)
        primary_file = relevant_files[0].file_path if relevant_files else None

        return MultiFileAnalysisRequest(
            query=query,
            selected_files=[f.file_path for f in relevant_files],
            analysis_type=analysis_type,
            primary_file=primary_file,
        )

    async def _determine_analysis_type(
        self, query: str, relevant_files: List[FileRelevanceScore]
    ) -> str:
        """Determine the best analysis type for the query and files."""

        file_info = "\n".join(
            [f"- {f.file_path}: {f.reasoning}" for f in relevant_files[:3]]
        )

        prompt = f"""
Determine best analysis strategy for this multi-file query:

Query: "{query}"

Available files:
{file_info}

Choose analysis type:
- "join": Combine data from multiple files using common keys
- "compare": Compare metrics/values across different files
- "aggregate": Aggregate data from multiple files into summary
- "hybrid": Complex analysis combining multiple approaches

Respond with JSON:
{{
    "analysis_type": "join|compare|aggregate|hybrid",
    "reasoning": "why this approach is best"
}}
"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data analyst. Choose optimal multi-file analysis strategies.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=200,
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)
            return result.get("analysis_type", "aggregate")

        except Exception as e:
            logger.warning(f"Analysis type determination failed: {e}")
            return "aggregate"  # Safe fallback

    async def _execute_multi_file_analysis(
        self, request: MultiFileAnalysisRequest, data_directory: Path
    ) -> Dict[str, Any]:
        """Execute the multi-file analysis using CrewAI."""

        # Create enhanced schema info for selected files
        enhanced_schema = await self._build_multi_file_schema(
            request.selected_files, request.analysis_type
        )

        # Create multi-file analysis prompt
        analysis_prompt = self._create_multi_file_prompt(request, enhanced_schema)

        # Execute with CrewAI
        try:
            result = await self.code_executor.analyze_with_code(
                query=request.query,
                file_path=data_directory / request.primary_file,
                schema_info=enhanced_schema,
                additional_context=analysis_prompt,
            )

            return result

        except Exception as e:
            logger.error(f"Multi-file execution failed: {e}")
            return {"error": str(e)}

    async def _build_multi_file_schema(
        self, file_paths: List[str], analysis_type: str
    ) -> Dict[str, Any]:
        """Build combined schema information for multiple files."""

        schemas = self.file_selector.schemas.get("schemas", {})

        combined_schema = {
            "multi_file_analysis": True,
            "analysis_type": analysis_type,
            "files": {},
            "suggested_joins": [],
            "common_columns": [],
        }

        # Collect schemas for selected files
        all_columns = set()
        for file_path in file_paths:
            if file_path in schemas and "error" not in schemas[file_path]:
                file_schema = schemas[file_path]
                combined_schema["files"][file_path] = {
                    "schemas": file_schema.get("schemas", {}),
                    "business_analysis": file_schema.get("business_analysis", {}),
                }

                # Collect all column names
                for table_schema in file_schema.get("schemas", {}).values():
                    all_columns.update(table_schema.get("headers", []))

        # Find potential join columns (common column names)
        join_candidates = []
        for file_path in file_paths:
            if file_path in combined_schema["files"]:
                file_columns = set()
                for table_schema in combined_schema["files"][file_path][
                    "schemas"
                ].values():
                    file_columns.update(table_schema.get("headers", []))

                for other_file in file_paths:
                    if (
                        other_file != file_path
                        and other_file in combined_schema["files"]
                    ):
                        other_columns = set()
                        for table_schema in combined_schema["files"][other_file][
                            "schemas"
                        ].values():
                            other_columns.update(table_schema.get("headers", []))

                        common = file_columns.intersection(other_columns)
                        if common:
                            join_candidates.append(
                                {
                                    "file1": file_path,
                                    "file2": other_file,
                                    "common_columns": list(common),
                                }
                            )

        combined_schema["suggested_joins"] = join_candidates
        combined_schema["total_files"] = len(file_paths)

        return combined_schema

    def _create_multi_file_prompt(
        self, request: MultiFileAnalysisRequest, schema: Dict[str, Any]
    ) -> str:
        """Create analysis prompt for multi-file operations."""

        prompt_parts = [
            f"MULTI-FILE ANALYSIS REQUEST",
            f"Query: {request.query}",
            f"Analysis Type: {request.analysis_type}",
            f"Files to analyze: {len(request.selected_files)}",
            "",
        ]

        if request.analysis_type == "join":
            prompt_parts.extend(
                [
                    "JOIN ANALYSIS INSTRUCTIONS:",
                    "- Identify common columns between files for joining",
                    "- Use pandas merge operations to combine datasets",
                    "- Handle missing values and data type mismatches",
                    "- Validate join results for data quality",
                    "",
                ]
            )
        elif request.analysis_type == "compare":
            prompt_parts.extend(
                [
                    "COMPARISON ANALYSIS INSTRUCTIONS:",
                    "- Load data from all files",
                    "- Standardize column names and data types",
                    "- Compare key metrics across files",
                    "- Highlight differences and trends",
                    "",
                ]
            )
        elif request.analysis_type == "aggregate":
            prompt_parts.extend(
                [
                    "AGGREGATION ANALYSIS INSTRUCTIONS:",
                    "- Combine data from multiple sources",
                    "- Perform aggregations (sum, mean, count, etc.)",
                    "- Create summary statistics across all files",
                    "- Generate consolidated insights",
                    "",
                ]
            )

        # Add join suggestions if available
        if schema.get("suggested_joins"):
            prompt_parts.append("SUGGESTED JOIN OPPORTUNITIES:")
            for join_info in schema["suggested_joins"][:3]:
                prompt_parts.append(
                    f"- {join_info['file1']} ↔ {join_info['file2']}: {', '.join(join_info['common_columns'])}"
                )
            prompt_parts.append("")

        prompt_parts.extend(
            [
                "IMPLEMENTATION REQUIREMENTS:",
                "- Use pandas for data manipulation and joins",
                "- Include data validation and error handling",
                "- Provide clear summary of results",
                "- Show data quality metrics (null counts, duplicates, etc.)",
                "- Create visualizations where appropriate",
            ]
        )

        return "\n".join(prompt_parts)


# Convenience function for easy usage
async def analyze_multi_file_query(
    query: str, data_directory: Path, schema_file: Path, max_files: int = 5
) -> Dict[str, Any]:
    """
    Convenience function to analyze queries across multiple Excel files.

    Args:
        query: The analysis query
        data_directory: Directory containing Excel files
        schema_file: Path to preprocessed schema file
        max_files: Maximum number of files to analyze together
    """

    analyst = MultiFileExcelAnalyst(schema_file)
    return await analyst.analyze_multi_file_query(
        query=query, data_directory=data_directory, max_files=max_files
    )
