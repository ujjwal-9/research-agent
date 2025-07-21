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
    """Pipeline for ingesting documents into the system."""
    
    def __init__(self):
        self.processor = DocumentProcessor(
            chunk_size=settings.document_chunk_size,
            chunk_overlap=settings.document_chunk_overlap
        )
        self.store = DocumentStore()
    
    async def ingest_directory(self, data_dir: Path, force_reindex: bool = False) -> None:
        """Ingest all documents from a directory."""
        logger.info(f"Starting document ingestion from {data_dir}")
        
        # Find all supported files
        supported_extensions = {'.txt', '.docx', '.xlsx', '.xls', '.pptx', '.ppt', '.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
        files_to_process = []
        
        for ext in supported_extensions:
            files_to_process.extend(data_dir.rglob(f"*{ext}"))
        
        logger.info(f"Found {len(files_to_process)} files to process")
        
        if not force_reindex:
            # Filter out already processed files
            existing_docs = {doc['file_path'] for doc in self.store.get_unique_documents()}
            files_to_process = [f for f in files_to_process if str(f) not in existing_docs]
            logger.info(f"Skipping {len(existing_docs)} already processed files")
        
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
              default="./data", help="Directory containing documents to ingest")
@click.option("--force-reindex", "-f", is_flag=True, 
              help="Force reindexing of all documents")
@click.option("--clear-existing", "-c", is_flag=True, 
              help="Clear existing documents before ingestion")
def main(data_dir: Path, force_reindex: bool, clear_existing: bool):
    """Ingest documents from the specified directory."""
    
    async def run_ingestion():
        pipeline = DocumentIngestionPipeline()
        
        if clear_existing:
            logger.info("Clearing existing documents...")
            pipeline.store.clear_all_documents()
        
        await pipeline.ingest_directory(data_dir, force_reindex)
    
    # Run the ingestion pipeline
    asyncio.run(run_ingestion())


if __name__ == "__main__":
    main()