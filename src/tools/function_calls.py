"""OpenAI-compatible function calls for document access."""

from typing import List, Dict, Any, Optional
import json
from loguru import logger

from src.ingestion.document_store import DocumentStore
from src.tools.web_search import WebSearchTool


class DocumentSearchTool:
    """Function call tool for searching internal documents."""
    
    def __init__(self):
        self.document_store = DocumentStore()
    
    @staticmethod
    def get_function_definition() -> Dict[str, Any]:
        """Get OpenAI function definition for document search."""
        return {
            "name": "search_documents",
            "description": "Search internal documents for relevant information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find relevant documents"
                    },
                    "file_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of file types to filter by (e.g., ['docx', 'pdf'])"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 10)"
                    }
                },
                "required": ["query"]
            }
        }
    
    async def execute(self, query: str, file_types: Optional[List[str]] = None, max_results: int = 10) -> Dict[str, Any]:
        """Execute document search."""
        try:
            results = self.document_store.search_documents(
                query=query,
                n_results=max_results,
                file_types=file_types
            )
            
            # Format results for function call response
            formatted_results = []
            for result in results:
                formatted_result = {
                    "content": result["content"],
                    "file_path": result["metadata"].get("file_path", ""),
                    "file_type": result["metadata"].get("file_type", ""),
                    "title": result["metadata"].get("title", ""),
                    "relevance_score": 1.0 - result.get("distance", 0.0) if result.get("distance") else None
                }
                formatted_results.append(formatted_result)
            
            return {
                "success": True,
                "results": formatted_results,
                "total_found": len(formatted_results),
                "query": query
            }
            
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query
            }


class WebSearchFunctionTool:
    """Function call tool for web search."""
    
    def __init__(self):
        self.web_search = WebSearchTool()
    
    @staticmethod
    def get_function_definition() -> Dict[str, Any]:
        """Get OpenAI function definition for web search."""
        return {
            "name": "search_web",
            "description": "Search the web for current information and external sources",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for web search"
                    },
                    "search_type": {
                        "type": "string",
                        "enum": ["general", "news"],
                        "description": "Type of web search to perform"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 10)"
                    }
                },
                "required": ["query"]
            }
        }
    
    async def execute(self, query: str, search_type: str = "general", max_results: int = 10) -> Dict[str, Any]:
        """Execute web search."""
        try:
            if search_type == "news":
                results = await self.web_search.search_news(query, max_results)
            else:
                results = await self.web_search.search(query, max_results)
            
            return {
                "success": True,
                "results": results,
                "total_found": len(results),
                "query": query,
                "search_type": search_type
            }
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query
            }


class DocumentListTool:
    """Function call tool for listing available documents."""
    
    def __init__(self):
        self.document_store = DocumentStore()
    
    @staticmethod
    def get_function_definition() -> Dict[str, Any]:
        """Get OpenAI function definition for listing documents."""
        return {
            "name": "list_documents",
            "description": "List all available documents in the system",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_type": {
                        "type": "string",
                        "description": "Optional file type filter (e.g., 'docx', 'pdf')"
                    }
                }
            }
        }
    
    async def execute(self, file_type: Optional[str] = None) -> Dict[str, Any]:
        """Execute document listing."""
        try:
            documents = self.document_store.get_unique_documents()
            
            if file_type:
                documents = [doc for doc in documents if doc.get("file_type") == file_type]
            
            return {
                "success": True,
                "documents": documents,
                "total_count": len(documents)
            }
            
        except Exception as e:
            logger.error(f"Document listing failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }


class FunctionCallManager:
    """Manager for all function call tools."""
    
    def __init__(self):
        self.document_search = DocumentSearchTool()
        self.web_search = WebSearchFunctionTool()
        self.document_list = DocumentListTool()
        
        self.tools = {
            "search_documents": self.document_search,
            "search_web": self.web_search,
            "list_documents": self.document_list
        }
    
    def get_all_function_definitions(self) -> List[Dict[str, Any]]:
        """Get all function definitions for OpenAI API."""
        return [
            self.document_search.get_function_definition(),
            self.web_search.get_function_definition(),
            self.document_list.get_function_definition()
        ]
    
    async def execute_function(self, function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a function call."""
        if function_name not in self.tools:
            return {
                "success": False,
                "error": f"Unknown function: {function_name}"
            }
        
        tool = self.tools[function_name]
        
        try:
            return await tool.execute(**arguments)
        except Exception as e:
            logger.error(f"Function execution failed for {function_name}: {e}")
            return {
                "success": False,
                "error": str(e)
            }