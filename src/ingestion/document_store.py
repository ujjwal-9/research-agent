"""Document storage and retrieval using vector database."""

import os
from typing import List, Dict, Any, Optional
from dataclasses import asdict
import json

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from loguru import logger

from src.config import settings
from src.ingestion.document_processor import ProcessedDocument


class DocumentStore:
    """Manages document storage and retrieval using ChromaDB."""
    
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        logger.info(f"Initialized document store with {self.get_document_count()} documents")
    
    async def store_document(self, document: ProcessedDocument) -> bool:
        """Store a processed document in the vector database."""
        try:
            # Prepare document chunks for storage
            chunk_ids = []
            chunk_texts = []
            chunk_metadatas = []
            
            for i, chunk in enumerate(document.chunks):
                chunk_id = f"{document.file_path}_{i}"
                chunk_ids.append(chunk_id)
                chunk_texts.append(chunk)
                
                chunk_metadata = {
                    "file_path": document.file_path,
                    "file_type": document.file_type,
                    "title": document.title,
                    "chunk_index": i,
                    "total_chunks": len(document.chunks),
                    **document.metadata
                }
                chunk_metadatas.append(chunk_metadata)
            
            if chunk_texts:
                # Generate embeddings
                embeddings = self.embedding_model.encode(chunk_texts).tolist()
                
                # Store in ChromaDB
                self.collection.add(
                    ids=chunk_ids,
                    documents=chunk_texts,
                    metadatas=chunk_metadatas,
                    embeddings=embeddings
                )
                
                logger.info(f"Stored document {document.file_path} with {len(chunk_texts)} chunks")
                return True
            else:
                logger.warning(f"No content to store for {document.file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error storing document {document.file_path}: {e}")
            return False
    
    def search_documents(self, query: str, n_results: int = 10, file_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Search for relevant document chunks."""
        try:
            # Build where clause for filtering
            where_clause = {}
            if file_types:
                where_clause["file_type"] = {"$in": file_types}
            
            # Perform similarity search
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause if where_clause else None
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    result = {
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if results['distances'] else None
                    }
                    formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} results for query: {query[:50]}...")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def get_document_by_path(self, file_path: str) -> List[Dict[str, Any]]:
        """Retrieve all chunks for a specific document."""
        try:
            results = self.collection.get(
                where={"file_path": file_path}
            )
            
            formatted_results = []
            if results['documents']:
                for i in range(len(results['documents'])):
                    result = {
                        'content': results['documents'][i],
                        'metadata': results['metadatas'][i]
                    }
                    formatted_results.append(result)
            
            # Sort by chunk index
            formatted_results.sort(key=lambda x: x['metadata'].get('chunk_index', 0))
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error retrieving document {file_path}: {e}")
            return []
    
    def get_document_count(self) -> int:
        """Get total number of document chunks stored."""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0
    
    def get_unique_documents(self) -> List[Dict[str, Any]]:
        """Get list of unique documents with metadata."""
        try:
            # Get all documents
            results = self.collection.get()
            
            # Group by file path
            unique_docs = {}
            if results['metadatas']:
                for metadata in results['metadatas']:
                    file_path = metadata['file_path']
                    if file_path not in unique_docs:
                        unique_docs[file_path] = {
                            'file_path': file_path,
                            'title': metadata.get('title', ''),
                            'file_type': metadata.get('file_type', ''),
                            'total_chunks': metadata.get('total_chunks', 0)
                        }
            
            return list(unique_docs.values())
            
        except Exception as e:
            logger.error(f"Error getting unique documents: {e}")
            return []
    
    def delete_document(self, file_path: str) -> bool:
        """Delete all chunks for a specific document."""
        try:
            # Get all chunk IDs for this document
            results = self.collection.get(
                where={"file_path": file_path}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted document {file_path} ({len(results['ids'])} chunks)")
                return True
            else:
                logger.warning(f"Document {file_path} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting document {file_path}: {e}")
            return False
    
    def clear_all_documents(self) -> bool:
        """Clear all documents from the store."""
        try:
            self.client.delete_collection("documents")
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Cleared all documents from store")
            return True
        except Exception as e:
            logger.error(f"Error clearing documents: {e}")
            return False