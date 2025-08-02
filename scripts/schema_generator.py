#!/usr/bin/env python
"""Enhanced schema generator with PDF context integration and table extraction."""

import argparse
import asyncio
import glob
import json
import logging
import math
import os
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
from openai import OpenAI, AsyncOpenAI
from loguru import logger

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings
from src.ingestion.document_store import DocumentStore
from src.ingestion.document_processor import DocumentProcessor
from src.knowledge_graph.enhanced_hybrid_retriever import EnhancedHybridRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("enhanced_schema_generator")


class TableExtractor:
    """Extracts tables from Excel files and splits them into separate files."""

    def __init__(self, api_key: str = None, model: str = "gpt-4.1"):
        self.extracted_tables = []
        self.api_key = api_key
        self.model = model
        if api_key:
            self.async_client = AsyncOpenAI(api_key=api_key)
        else:
            self.async_client = None

    async def extract_tables_from_excel(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract individual tables from Excel file with intelligent header detection."""
        extracted_tables = []

        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)

            for sheet_name in excel_file.sheet_names:
                # Read without assuming headers first
                df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

                if df_raw.empty:
                    continue

                # Detect table boundaries and headers intelligently
                tables = await self._detect_table_boundaries_with_headers(
                    df_raw, sheet_name, file_path
                )

                for table_idx, table_data in enumerate(tables):
                    table_info = {
                        "original_file": str(file_path),
                        "sheet_name": sheet_name,
                        "table_index": table_idx,
                        "table_name": f"{sheet_name}_table_{table_idx}",
                        "data": table_data["data"],
                        "header_row": table_data["header_row"],
                        "start_row": table_data["start_row"],
                        "end_row": table_data["end_row"],
                        "start_col": table_data["start_col"],
                        "end_col": table_data["end_col"],
                        "row_count": len(table_data["data"]),
                        "column_count": (
                            len(table_data["data"].columns)
                            if not table_data["data"].empty
                            else 0
                        ),
                    }
                    extracted_tables.append(table_info)

        except Exception as e:
            logger.error(f"Error extracting tables from {file_path}: {e}")

        return extracted_tables

    async def _detect_table_boundaries_with_headers(
        self, df_raw: pd.DataFrame, sheet_name: str, file_path: Path
    ) -> List[Dict[str, Any]]:
        """Detect individual table boundaries within a sheet with intelligent header detection."""
        tables = []

        if df_raw.empty or df_raw.isna().all().all():
            return tables

        # Find the header row using LLM on RAW data (before cleaning)
        header_row_idx = await self._find_header_row_with_llm(
            df_raw, sheet_name, file_path
        )

        if header_row_idx is None:
            # Fallback to traditional approach if LLM fails
            logger.warning(
                f"LLM header detection failed for {sheet_name}, using default approach"
            )
            header_row_idx = 0

        # Now clean the data but preserve the header row index relationship
        # Remove completely empty rows and columns, but keep track of original indices
        non_empty_row_indices = []
        for i in range(len(df_raw)):
            if not df_raw.iloc[i].isna().all():
                non_empty_row_indices.append(i)

        if not non_empty_row_indices:
            return tables

        # Filter to non-empty rows
        df_filtered = df_raw.iloc[non_empty_row_indices].reset_index(drop=True)

        # Remove completely empty columns
        df_cleaned = df_filtered.dropna(how="all", axis=1)

        if df_cleaned.empty:
            return tables

        # Map the original header row index to the cleaned DataFrame index
        if header_row_idx in non_empty_row_indices:
            header_row_idx_in_cleaned = non_empty_row_indices.index(header_row_idx)
        else:
            # Header row was empty, find the next non-empty row
            next_non_empty = None
            for idx in non_empty_row_indices:
                if idx > header_row_idx:
                    next_non_empty = idx
                    break
            if next_non_empty is not None:
                header_row_idx_in_cleaned = non_empty_row_indices.index(next_non_empty)
            else:
                header_row_idx_in_cleaned = 0

        # Create the table with proper headers
        try:
            # Extract header and data
            if header_row_idx_in_cleaned < len(df_cleaned):
                # Use the identified row as header
                headers = (
                    df_cleaned.iloc[header_row_idx_in_cleaned]
                    .fillna("")
                    .astype(str)
                    .tolist()
                )

                # Clean up headers - remove empty or purely numeric headers
                cleaned_headers = []
                for i, header in enumerate(headers):
                    if header.strip() == "" or header.startswith("Unnamed"):
                        cleaned_headers.append(f"Column_{i+1}")
                    else:
                        cleaned_headers.append(header.strip())

                # Get data rows (everything after the header)
                data_rows_raw = df_cleaned.iloc[
                    header_row_idx_in_cleaned + 1 :
                ].reset_index(drop=True)

                if not data_rows_raw.empty:
                    # Create DataFrame with proper headers
                    # Convert raw data to list of lists then back to DataFrame with proper headers
                    data_values = data_rows_raw.values.tolist()

                    # Create new DataFrame with proper column names
                    num_cols = min(
                        len(cleaned_headers), len(data_values[0]) if data_values else 0
                    )
                    if num_cols > 0:
                        data_rows = pd.DataFrame(
                            data_values, columns=cleaned_headers[:num_cols]
                        )
                    else:
                        data_rows = pd.DataFrame()

                    # Remove any completely empty rows in the data
                    data_rows = data_rows.dropna(how="all")

                    if not data_rows.empty:
                        tables.append(
                            {
                                "data": data_rows,
                                "header_row": header_row_idx,  # Keep original header row index
                                "start_row": header_row_idx,
                                "end_row": header_row_idx + len(data_rows) + 1,
                                "start_col": 0,
                                "end_col": len(data_rows.columns),
                            }
                        )

        except Exception as e:
            logger.error(
                f"Error processing table with header row {header_row_idx}: {e}"
            )
            # Fallback to simple approach
            tables.append(
                {
                    "data": df_cleaned,
                    "header_row": 0,
                    "start_row": 0,
                    "end_row": len(df_cleaned),
                    "start_col": 0,
                    "end_col": len(df_cleaned.columns),
                }
            )

        return tables

    async def _find_header_row_with_llm(
        self, df: pd.DataFrame, sheet_name: str, file_path: Path
    ) -> Optional[int]:
        """Use LLM to identify the header row from the first few rows."""

        if not self.async_client:
            logger.warning("No LLM client available for header detection")
            return None

        try:
            # Get first 10 rows for analysis
            analysis_rows = min(10, len(df))
            sample_df = df.head(analysis_rows)

            # Convert to readable format for LLM
            rows_text = []
            for idx, row in sample_df.iterrows():
                row_content = [
                    str(cell).strip() if pd.notna(cell) else "" for cell in row
                ]
                rows_text.append(
                    f"Row {idx}: {' | '.join(row_content[:10])}"
                )  # Limit to first 10 columns

            prompt = f"""
            Analyze the following rows from an Excel sheet and determine which row contains the column headers.
            
            File: {file_path.name}
            Sheet: {sheet_name}
            
            Rows to analyze:
            {chr(10).join(rows_text)}
            
            Look for characteristics of header rows:
            1. Contains descriptive text rather than data values
            2. May contain column names like "Name", "ID", "Amount", "Date", etc.
            3. Usually the first row with meaningful text
            4. Avoids purely numeric values or dates
            5. Contains text that describes what the columns represent
            
            Return ONLY the row index (0, 1, 2, 3, or 4) of the row that contains the headers.
            If no clear header row is found, return 0.
            
            Response format: Just the number (e.g., "1")
            """

            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing spreadsheet data to identify header rows. Return only the row index number.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=10,  # We only need a single number
            )

            result = response.choices[0].message.content.strip()

            # Parse the result
            try:
                header_row_idx = int(result)
                if 0 <= header_row_idx < analysis_rows:
                    logger.info(
                        f"LLM identified header row {header_row_idx} for {sheet_name}"
                    )
                    return header_row_idx
                else:
                    logger.warning(
                        f"LLM returned invalid row index {header_row_idx}, using 0"
                    )
                    return 0
            except ValueError:
                logger.warning(f"LLM returned non-numeric result: {result}, using 0")
                return 0

        except Exception as e:
            logger.error(f"Error in LLM header detection: {e}")
            return None

    def save_table_as_excel(self, table_info: Dict[str, Any], output_dir: Path) -> Path:
        """Save extracted table as separate Excel file."""
        # Create filename for the table
        original_filename = Path(table_info["original_file"]).stem
        table_filename = f"{original_filename}_{table_info['table_name']}.xlsx"
        output_path = output_dir / table_filename

        # Save the table data
        table_info["data"].to_excel(output_path, index=False)

        logger.info(f"Saved table to {output_path}")
        return output_path


class QuestionGenerator:
    """Generates questions from extracted tables."""

    def __init__(self, api_key: str, model: str = "gpt-4.1"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def generate_questions_for_table(
        self, table_info: Dict[str, Any]
    ) -> List[str]:
        """Generate relevant questions for a table."""
        try:
            # Get sample of table data
            table_data = table_info["data"]
            sample_data = table_data.head(5).to_string(index=False)
            columns = table_data.columns.tolist()

            prompt = f"""
            Analyze the following table and generate 5-7 insightful questions that could be answered using this data.
            
            Table: {table_info['table_name']}
            Original file: {Path(table_info['original_file']).name}
            Sheet: {table_info['sheet_name']}
            
            Columns: {', '.join(columns)}
            
            Sample data:
            {sample_data}
            
            Generate questions that:
            1. Explore relationships between columns
            2. Identify trends or patterns
            3. Perform calculations or aggregations
            4. Compare different segments of data
            5. Analyze data quality or completeness
            
            Return the questions as a JSON array of strings.
            """

            response = await self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data analyst that generates insightful questions for tabular data analysis. Return questions as a JSON object with 'questions' key containing an array of strings.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            result = json.loads(response.choices[0].message.content)
            questions = result.get("questions", [])

            logger.info(
                f"Generated {len(questions)} questions for table {table_info['table_name']}"
            )
            return questions

        except Exception as e:
            logger.error(
                f"Error generating questions for table {table_info['table_name']}: {e}"
            )
            return []


class EnhancedSchemaGenerator:
    """Enhanced schema generator with PDF context integration."""

    def __init__(self, api_key: str, model: str = "gpt-4.1"):
        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.model = model

        # Initialize components
        self.document_store = DocumentStore()
        self.document_processor = DocumentProcessor()
        self.retriever = EnhancedHybridRetriever()
        self.table_extractor = TableExtractor(api_key, model)
        self.question_generator = QuestionGenerator(api_key, model)

        self.schema_cache = {}

    async def process_directory_with_context(
        self, data_dir: Path, output_dir: Path, max_context_chunks: int = 10
    ) -> Dict[str, Any]:
        """Process directory with PDF context integration."""

        logger.info(f"Processing directory: {data_dir}")

        # Step 1: Ensure PDFs are ingested into Qdrant
        await self._ensure_pdfs_ingested(data_dir)

        # Step 2: Find Excel files to process
        excel_files = list(data_dir.rglob("*.xlsx")) + list(data_dir.rglob("*.xls"))

        logger.info(f"Found {len(excel_files)} Excel files to process")

        results = {}

        for excel_file in excel_files:
            try:
                result = await self.process_excel_with_context(
                    excel_file, output_dir, max_context_chunks
                )
                results[str(excel_file.relative_to(data_dir))] = result
            except Exception as e:
                logger.error(f"Error processing {excel_file}: {e}")
                results[str(excel_file.relative_to(data_dir))] = {"error": str(e)}

        return results

    async def process_excel_with_context(
        self, excel_file: Path, output_dir: Path, max_context_chunks: int = 10
    ) -> Dict[str, Any]:
        """Process single Excel file with context retrieval."""

        logger.info(f"Processing Excel file: {excel_file}")

        # Step 1: Extract tables from Excel file
        tables = await self.table_extractor.extract_tables_from_excel(excel_file)

        if not tables:
            return {"error": "No tables found in Excel file"}

        # Create output directory for this Excel file
        excel_output_dir = output_dir / excel_file.stem
        excel_output_dir.mkdir(parents=True, exist_ok=True)

        processed_tables = []

        for table_info in tables:
            try:
                # Step 2: Generate questions for the table
                questions = await self.question_generator.generate_questions_for_table(
                    table_info
                )

                # Step 3: Retrieve context using questions
                context_chunks = await self._retrieve_context_for_table(
                    table_info, questions, excel_file.parent, max_context_chunks
                )

                # Step 4: Generate enhanced schema with context
                schema = await self._generate_schema_with_context(
                    table_info, context_chunks, questions
                )

                # Step 5: Save table as separate Excel file
                table_file_path = self.table_extractor.save_table_as_excel(
                    table_info, excel_output_dir
                )

                # Step 6: Create metadata JSON file
                metadata = self._create_table_metadata(
                    table_info, schema, context_chunks, questions, table_file_path
                )

                metadata_path = (
                    excel_output_dir / f"{table_file_path.stem}_metadata.json"
                )
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

                processed_tables.append(
                    {
                        "table_name": table_info["table_name"],
                        "excel_file": str(table_file_path),
                        "metadata_file": str(metadata_path),
                        "schema": schema,
                        "questions": questions,
                        "context_chunks_used": len(context_chunks),
                    }
                )

                logger.info(
                    f"Processed table {table_info['table_name']} with {len(context_chunks)} context chunks"
                )

            except Exception as e:
                logger.error(
                    f"Error processing table {table_info.get('table_name', 'unknown')}: {e}"
                )
                continue

        return {
            "original_file": str(excel_file),
            "output_directory": str(excel_output_dir),
            "tables_processed": len(processed_tables),
            "tables": processed_tables,
        }

    async def _ensure_pdfs_ingested(self, data_dir: Path):
        """Ensure all PDFs in directory are ingested into Qdrant."""
        pdf_files = list(data_dir.rglob("*.pdf"))

        logger.info(f"Found {len(pdf_files)} PDF files in directory")

        for pdf_file in pdf_files:
            try:
                # Check if already ingested by looking for existing documents
                existing_docs = self.document_store.get_document_by_path(str(pdf_file))

                if not existing_docs:
                    logger.info(f"Ingesting PDF: {pdf_file}")
                    # Process and store the PDF
                    processed_doc = await self.document_processor.process_document(
                        pdf_file
                    )
                    if processed_doc:
                        await self.document_store.store_document(processed_doc)
                else:
                    logger.debug(f"PDF already ingested: {pdf_file}")

            except Exception as e:
                logger.error(f"Error ingesting PDF {pdf_file}: {e}")

    async def _retrieve_context_for_table(
        self,
        table_info: Dict[str, Any],
        questions: List[str],
        directory: Path,
        max_chunks: int,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant context for table using questions."""

        all_context_chunks = []

        # Create search queries from table info and questions
        search_queries = self._create_search_queries(table_info, questions)

        for query in search_queries:
            try:
                # Search in document store with preference for same directory
                results = self.document_store.search_documents(
                    query=query,
                    n_results=max_chunks // len(search_queries),
                    file_types=["pdf"],
                )

                # Filter and score results based on file location
                scored_results = self._score_context_by_location(results, directory)
                all_context_chunks.extend(scored_results)

            except Exception as e:
                logger.error(f"Error retrieving context for query '{query}': {e}")

        # Remove duplicates and sort by relevance
        unique_chunks = self._deduplicate_context_chunks(all_context_chunks)
        sorted_chunks = sorted(
            unique_chunks, key=lambda x: x.get("relevance_score", 0), reverse=True
        )

        return sorted_chunks[:max_chunks]

    def _create_search_queries(
        self, table_info: Dict[str, Any], questions: List[str]
    ) -> List[str]:
        """Create search queries from table info and questions."""
        queries = []

        # Add table-based queries
        table_name = table_info.get("table_name", "")
        sheet_name = table_info.get("sheet_name", "")

        # Use column names for context
        if not table_info["data"].empty:
            columns = table_info["data"].columns.tolist()[
                :5
            ]  # Limit to first 5 columns
            queries.append(f"{' '.join(columns)} data analysis")

        # Add sheet/table name queries
        if sheet_name:
            queries.append(f"{sheet_name} information documentation")

        # Add question-based queries (use top 3 questions)
        for question in questions[:3]:
            # Extract key terms from questions
            key_terms = self._extract_key_terms_from_question(question)
            if key_terms:
                queries.append(" ".join(key_terms))

        return queries[:5]  # Limit total queries

    def _extract_key_terms_from_question(self, question: str) -> List[str]:
        """Extract key terms from a question for search."""
        # Simple keyword extraction - could be enhanced with NLP
        stop_words = {
            "what",
            "how",
            "when",
            "where",
            "why",
            "is",
            "are",
            "the",
            "and",
            "or",
            "to",
            "of",
            "in",
            "for",
            "with",
        }
        words = question.lower().split()
        key_terms = [
            word.strip("?.,!")
            for word in words
            if word.strip("?.,!") not in stop_words and len(word) > 3
        ]
        return key_terms[:5]  # Limit to top 5 terms

    def _score_context_by_location(
        self, results: List[Dict[str, Any]], directory: Path
    ) -> List[Dict[str, Any]]:
        """Score context results giving preference to files in same directory."""
        scored_results = []

        for result in results:
            file_path = result.get("metadata", {}).get("file_path", "")
            base_relevance = 1.0 - result.get("distance", 0.5)

            # Boost score for files in same directory
            location_boost = 0.0
            if file_path:
                result_path = Path(file_path)
                if result_path.parent == directory:
                    location_boost = 0.3  # Strong boost for same directory
                elif str(directory) in file_path:
                    location_boost = 0.1  # Smaller boost for subdirectories

            final_score = min(base_relevance + location_boost, 1.0)

            result_copy = result.copy()
            result_copy["relevance_score"] = final_score
            result_copy["location_boost"] = location_boost
            scored_results.append(result_copy)

        return scored_results

    def _deduplicate_context_chunks(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove duplicate context chunks."""
        seen_chunks = set()
        unique_chunks = []

        for chunk in chunks:
            chunk_id = chunk.get("metadata", {}).get("chunk_id", "")
            if chunk_id and chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                unique_chunks.append(chunk)
            elif not chunk_id:
                # If no chunk_id, use content hash
                content_hash = hash(chunk.get("content", ""))
                if content_hash not in seen_chunks:
                    seen_chunks.add(content_hash)
                    unique_chunks.append(chunk)

        return unique_chunks

    async def _generate_schema_with_context(
        self,
        table_info: Dict[str, Any],
        context_chunks: List[Dict[str, Any]],
        questions: List[str],
    ) -> Dict[str, Any]:
        """Generate schema with context information."""

        try:
            # Prepare table data
            table_data = table_info["data"]
            sample_data = table_data.head(5).to_string(index=False)
            columns = table_data.columns.tolist()

            # Prepare context information
            context_text = self._format_context_for_schema_generation(context_chunks)

            prompt = f"""
            Generate a comprehensive schema for this data table using the provided context information.
            
            Table Information:
            - Name: {table_info['table_name']}
            - Original file: {Path(table_info['original_file']).name}
            - Sheet: {table_info['sheet_name']}
            - Dimensions: {table_info['row_count']} rows × {table_info['column_count']} columns
            
            Columns: {', '.join(columns)}
            
            Sample data:
            {sample_data}
            
            Related Questions:
            {chr(10).join(f"- {q}" for q in questions)}
            
            Context from Related Documents:
            {context_text}
            
            Generate a detailed schema that includes:
            1. Column descriptions informed by the context
            2. Data types and constraints
            3. Business meaning derived from context
            4. Relationships to other data mentioned in context
            5. Potential use cases based on the questions
            
            Format as JSON with this structure:
            {{
                "table_name": "string",
                "description": "string",
                "columns": {{
                    "column_name": {{
                        "type": "string",
                        "description": "string",
                        "business_meaning": "string",
                        "constraints": "string",
                        "related_context": "string"
                    }}
                }},
                "relationships": ["string"],
                "use_cases": ["string"],
                "data_quality_notes": ["string"]
            }}
            """

            response = await self.async_client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data analyst that creates comprehensive schemas using contextual information from related documents.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            schema = json.loads(response.choices[0].message.content)
            return schema

        except Exception as e:
            logger.error(f"Error generating schema with context: {e}")
            return {
                "error": str(e),
                "table_name": table_info.get("table_name", "unknown"),
                "basic_columns": {
                    col: {
                        "type": "unknown",
                        "description": "Error generating description",
                    }
                    for col in table_info["data"].columns
                },
            }

    def _format_context_for_schema_generation(
        self, context_chunks: List[Dict[str, Any]]
    ) -> str:
        """Format context chunks for schema generation."""
        if not context_chunks:
            return "No related context found."

        context_parts = []
        for i, chunk in enumerate(context_chunks[:5], 1):  # Use top 5 chunks
            content = chunk.get("content", "").strip()
            file_path = chunk.get("metadata", {}).get("file_path", "Unknown")
            chunk_idx = chunk.get("metadata", {}).get("chunk_index", "Unknown")

            context_parts.append(
                f"Context {i} (from {Path(file_path).name}, chunk {chunk_idx}):\n{content[:500]}..."
            )

        return "\n\n".join(context_parts)

    def _create_table_metadata(
        self,
        table_info: Dict[str, Any],
        schema: Dict[str, Any],
        context_chunks: List[Dict[str, Any]],
        questions: List[str],
        table_file_path: Path,
    ) -> Dict[str, Any]:
        """Create metadata JSON for the table."""

        # Extract context source information
        context_sources = []
        for chunk in context_chunks:
            metadata = chunk.get("metadata", {})
            source_info = {
                "file_path": metadata.get("file_path", "Unknown"),
                "chunk_index": metadata.get("chunk_index", "Unknown"),
                "relevance_score": chunk.get("relevance_score", 0.0),
                "location_boost": chunk.get("location_boost", 0.0),
            }
            context_sources.append(source_info)

        metadata = {
            "table_info": {
                "name": table_info["table_name"],
                "original_file": table_info["original_file"],
                "sheet_name": table_info["sheet_name"],
                "table_index": table_info["table_index"],
                "dimensions": {
                    "rows": table_info["row_count"],
                    "columns": table_info["column_count"],
                },
                "position": {
                    "start_row": table_info["start_row"],
                    "end_row": table_info["end_row"],
                    "start_col": table_info["start_col"],
                    "end_col": table_info["end_col"],
                },
            },
            "extracted_file": str(table_file_path),
            "generated_questions": questions,
            "schema": schema,
            "context_used": {
                "total_chunks": len(context_chunks),
                "sources": context_sources,
            },
            "generation_metadata": {
                "timestamp": pd.Timestamp.now().isoformat(),
                "model_used": self.model,
                "context_preference": "same_directory_files",
            },
        }

        return metadata


async def main():
    """Main entry point for the enhanced schema generator."""
    parser = argparse.ArgumentParser(
        description="Enhanced schema generator with PDF context integration"
    )

    parser.add_argument(
        "--data-dir", required=True, help="Directory containing Excel and PDF files"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for processed tables and schemas",
    )
    parser.add_argument("--model", default="gpt-4.1", help="OpenAI model to use")
    parser.add_argument(
        "--max-context-chunks",
        type=int,
        default=10,
        help="Maximum context chunks per table",
    )

    args = parser.parse_args()

    # Get API key
    api_key = settings.openai_api_key
    if not api_key:
        logger.error("OpenAI API key not found in settings")
        sys.exit(1)

    # Validate paths
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create enhanced schema generator
    generator = EnhancedSchemaGenerator(api_key=api_key, model=args.model)

    try:
        # Process directory
        results = await generator.process_directory_with_context(
            data_dir=data_dir,
            output_dir=output_dir,
            max_context_chunks=args.max_context_chunks,
        )

        # Save summary results
        summary_file = output_dir / "processing_summary.json"
        with open(summary_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Processing complete. Results saved to {output_dir}")
        logger.info(f"Summary: {summary_file}")

        # Print summary
        total_files = len(results)
        successful_files = len([r for r in results.values() if "error" not in r])
        total_tables = sum(
            r.get("tables_processed", 0)
            for r in results.values()
            if isinstance(r, dict) and "tables_processed" in r
        )

        print(f"\nProcessing Summary:")
        print(f"- Files processed: {successful_files}/{total_files}")
        print(f"- Total tables extracted: {total_tables}")
        print(f"- Output directory: {output_dir}")

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
