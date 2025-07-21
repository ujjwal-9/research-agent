"""Document ingestion script for processing and indexing documents."""

import asyncio
from pathlib import Path
from typing import List
import click
from tqdm import tqdm
from loguru import logger

from src.ingestion.document_processor import DocumentProcessor
from src.ingestion.document_store import DocumentStore
from src.config import settings


class DocumentIngestionPipeline:
    """Pipeline for ingesting documents into the system with knowledge graph integration."""
    
    def __init__(self, enable_knowledge_graph: bool = True):
        self.processor = DocumentProcessor(
            chunk_size=settings.document_chunk_size,
            chunk_overlap=settings.document_chunk_overlap
        )
        self.store = DocumentStore(enable_knowledge_graph=enable_knowledge_graph)
        self.enable_kg = enable_knowledge_graph
    
    async def ingest_directory(self, data_dir: Path, force_reindex: bool = False, 
                             max_depth: int = None, exclude_dirs: tuple = ()) -> None:
        """Ingest all documents from a directory and all its subdirectories recursively."""
        logger.info(f"Starting recursive document ingestion from {data_dir}")
        
        # Verify directory exists and is accessible
        if not data_dir.exists():
            raise FileNotFoundError(f"Directory does not exist: {data_dir}")
        if not data_dir.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {data_dir}")
        
        # Find all supported files recursively
        supported_extensions = {'.txt', '.docx', '.xlsx', '.xls', '.pptx', '.ppt', '.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
        files_to_process = []
        
        # Use rglob to recursively find files in all subdirectories
        logger.info("Scanning directory tree for supported file types...")
        for ext in supported_extensions:
            found_files = list(data_dir.rglob(f"*{ext}"))
            
            # Apply filters
            filtered_files = self._filter_files(found_files, data_dir, max_depth, exclude_dirs)
            files_to_process.extend(filtered_files)
            
            if filtered_files:
                logger.debug(f"Found {len(filtered_files)} {ext} files (after filtering)")
        
        # Log directory structure summary
        self._log_directory_summary(data_dir, files_to_process)
        
        logger.info(f"Found {len(files_to_process)} total files to process across all subdirectories")
        
        if not force_reindex:
            # Filter out already processed files
            existing_docs = {doc['file_path'] for doc in self.store.get_unique_documents()}
            original_count = len(files_to_process)
            files_to_process = [f for f in files_to_process if str(f) not in existing_docs]
            skipped_count = original_count - len(files_to_process)
            if skipped_count > 0:
                logger.info(f"Skipping {skipped_count} already processed files")
            logger.info(f"Processing {len(files_to_process)} new/updated files")
        
        if not files_to_process:
            logger.info("No new files to process")
            return
        
        # Process files with concurrency control
        semaphore = asyncio.Semaphore(settings.max_concurrent_tasks)
        tasks = [self._process_file_with_semaphore(semaphore, file_path) for file_path in files_to_process]
        
        # Process with progress bar
        results = []
        with tqdm(total=len(tasks), desc="Processing documents") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                pbar.update(1)
        
        # Summary
        successful = sum(1 for r in results if r)
        failed = len(results) - successful
        
        logger.info(f"Document ingestion completed: {successful} successful, {failed} failed")
        logger.info(f"Total documents in store: {self.store.get_document_count()}")
        
        # Log knowledge graph statistics if enabled
        if self.enable_kg and hasattr(self.store, 'graph_store'):
            try:
                kg_stats = self.store.graph_store.get_graph_statistics()
                logger.info(f"Knowledge graph: {kg_stats.get('total_entities', 0)} entities, {kg_stats.get('total_relationships', 0)} relationships")
            except Exception as e:
                logger.warning(f"Failed to get knowledge graph statistics: {e}")
    
    def _log_directory_summary(self, root_dir: Path, files: List[Path]) -> None:
        """Log a summary of the directory structure and files found."""
        if not files:
            logger.warning(f"No supported files found in {root_dir}")
            return
        
        # Group files by directory
        dir_counts = {}
        file_type_counts = {}
        
        for file_path in files:
            # Count files per directory
            relative_dir = file_path.parent.relative_to(root_dir)
            dir_key = str(relative_dir) if str(relative_dir) != '.' else 'root'
            dir_counts[dir_key] = dir_counts.get(dir_key, 0) + 1
            
            # Count files by type
            ext = file_path.suffix.lower()
            file_type_counts[ext] = file_type_counts.get(ext, 0) + 1
        
        # Log directory structure
        logger.info("Directory structure summary:")
        for directory, count in sorted(dir_counts.items()):
            logger.info(f"  {directory}: {count} files")
        
        # Log file type distribution
        logger.info("File type distribution:")
        for file_type, count in sorted(file_type_counts.items()):
            logger.info(f"  {file_type}: {count} files")
    
    def _filter_files(self, files: List[Path], root_dir: Path, max_depth: int = None, 
                     exclude_dirs: tuple = ()) -> List[Path]:
        """Filter files based on depth and excluded directories."""
        filtered_files = []
        
        for file_path in files:
            # Check directory exclusions
            if exclude_dirs:
                relative_path = file_path.relative_to(root_dir)
                path_parts = relative_path.parts[:-1]  # Exclude filename
                
                # Check if any part of the path matches excluded directories
                if any(excluded_dir in path_parts for excluded_dir in exclude_dirs):
                    logger.debug(f"Excluding file in excluded directory: {file_path}")
                    continue
            
            # Check depth limit
            if max_depth is not None:
                relative_path = file_path.relative_to(root_dir)
                depth = len(relative_path.parts) - 1  # Subtract 1 for the filename
                
                if depth > max_depth:
                    logger.debug(f"Excluding file beyond max depth {max_depth}: {file_path}")
                    continue
            
            # Check if file is accessible
            try:
                if file_path.is_file() and file_path.stat().st_size > 0:
                    filtered_files.append(file_path)
                else:
                    logger.debug(f"Skipping empty or inaccessible file: {file_path}")
            except (OSError, PermissionError) as e:
                logger.warning(f"Cannot access file {file_path}: {e}")
        
        return filtered_files
    
    async def rebuild_knowledge_graph(self) -> dict:
        """Rebuild the knowledge graph from existing documents."""
        if not self.enable_kg:
            return {"success": False, "error": "Knowledge graph not enabled"}
        
        try:
            logger.info("Starting knowledge graph rebuild")
            
            # Clear existing graph
            if hasattr(self.store, 'graph_store'):
                self.store.graph_store.clear_graph()
            
            # Get all documents
            unique_docs = self.store.get_unique_documents()
            
            processed_count = 0
            failed_count = 0
            
            for doc_info in unique_docs:
                try:
                    file_path = doc_info["file_path"]
                    
                    # Get document chunks
                    chunks = self.store.get_document_by_path(file_path)
                    
                    # Process each chunk for knowledge graph extraction
                    for chunk in chunks:
                        chunk_id = str(chunk["metadata"].get("chunk_index", 0))
                        
                        # Extract entities and relationships
                        if hasattr(self.store, 'entity_extractor'):
                            entities, relationships = await self.store.entity_extractor.extract_entities_and_relationships(
                                chunk["content"], chunk_id
                            )
                            
                            # Store in knowledge graph
                            if entities or relationships:
                                await self.store.graph_store.store_entities_and_relationships(
                                    entities, relationships, file_path, chunk_id
                                )
                    
                    processed_count += 1
                    if processed_count % 10 == 0:
                        logger.info(f"Processed {processed_count}/{len(unique_docs)} documents")
                    
                except Exception as e:
                    logger.error(f"Failed to process {doc_info.get('file_path', 'unknown')} for knowledge graph: {e}")
                    failed_count += 1
            
            # Get final statistics
            kg_stats = {}
            if hasattr(self.store, 'graph_store'):
                kg_stats = self.store.graph_store.get_graph_statistics()
            
            logger.info(f"Knowledge graph rebuild completed: {processed_count} documents processed, {failed_count} failed")
            logger.info(f"Final graph: {kg_stats.get('total_entities', 0)} entities, {kg_stats.get('total_relationships', 0)} relationships")
            
            return {
                "success": True,
                "processed_documents": processed_count,
                "failed_documents": failed_count,
                "knowledge_graph_stats": kg_stats
            }
            
        except Exception as e:
            logger.error(f"Knowledge graph rebuild failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _process_file_with_semaphore(self, semaphore: asyncio.Semaphore, file_path: Path) -> bool:
        """Process a single file with concurrency control."""
        async with semaphore:
            return await self._process_single_file(file_path)
    
    async def _process_single_file(self, file_path: Path) -> bool:
        """Process and store a single file."""
        try:
            # Process document
            processed_doc = await self.processor.process_document(file_path)
            
            if processed_doc is None:
                logger.warning(f"Failed to process {file_path}")
                return False
            
            # Store in vector database
            success = await self.store.store_document(processed_doc)
            
            if success:
                logger.debug(f"Successfully processed {file_path}")
                return True
            else:
                logger.error(f"Failed to store {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return False


@click.command()
@click.option("--data-dir", "-d", type=click.Path(exists=True, path_type=Path), 
              default="./data", help="Root directory containing documents to ingest recursively")
@click.option("--force-reindex", "-f", is_flag=True, 
              help="Force reindexing of all documents")
@click.option("--clear-existing", "-c", is_flag=True, 
              help="Clear existing documents before ingestion")
@click.option("--max-depth", type=int, default=None,
              help="Maximum directory depth to recurse (default: unlimited)")
@click.option("--exclude-dirs", multiple=True,
              help="Directory names to exclude from ingestion (can be used multiple times)")
def main(data_dir: Path, force_reindex: bool, clear_existing: bool, max_depth: int, exclude_dirs: tuple):
    """Ingest documents from the specified directory and all subdirectories."""
    
    async def run_ingestion():
        pipeline = DocumentIngestionPipeline()
        
        if clear_existing:
            logger.info("Clearing existing documents...")
            pipeline.store.clear_all_documents()
        
        # Log ingestion parameters
        logger.info(f"Ingestion parameters:")
        logger.info(f"  Root directory: {data_dir}")
        logger.info(f"  Force reindex: {force_reindex}")
        logger.info(f"  Max depth: {max_depth if max_depth else 'unlimited'}")
        logger.info(f"  Excluded directories: {list(exclude_dirs) if exclude_dirs else 'none'}")
        
        await pipeline.ingest_directory(data_dir, force_reindex, max_depth, exclude_dirs)
    
    # Run the ingestion pipeline
    try:
        asyncio.run(run_ingestion())
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise


if __name__ == "__main__":
    main()