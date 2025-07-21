"""FastAPI server for the research system."""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import asyncio
from loguru import logger

from src.agents.research_orchestrator import ResearchOrchestrator
from src.tools.function_calls import FunctionCallManager
from src.ingestion.document_store import DocumentStore
from src.knowledge_graph.hybrid_retriever import HybridRetriever
from src.knowledge_graph.graph_store import KnowledgeGraphStore


# Pydantic models for API
class ResearchRequest(BaseModel):
    query: str
    interactive: bool = False
    session_id: Optional[str] = None


class ResearchResponse(BaseModel):
    success: bool
    session_id: str
    status: str
    message: str
    report: Optional[str] = None


class FunctionCallRequest(BaseModel):
    function_name: str
    arguments: Dict[str, Any]


class DocumentSearchRequest(BaseModel):
    query: str
    file_types: Optional[List[str]] = None
    max_results: int = 10


class HybridSearchRequest(BaseModel):
    query: str
    max_results: int = 10
    include_graph: bool = True
    include_rag: bool = True
    graph_depth: int = 2
    entity_boost: float = 1.5


class EntityExpansionRequest(BaseModel):
    entity_text: str
    max_depth: int = 2


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="Research System API",
        description="Multi-agent research system with document ingestion",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize components
    orchestrator = ResearchOrchestrator()
    function_manager = FunctionCallManager()
    document_store = DocumentStore()
    hybrid_retriever = HybridRetriever()
    graph_store = KnowledgeGraphStore()
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        await orchestrator.cleanup()
        document_store.close()
        hybrid_retriever.close()
        graph_store.close()
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "Research System API",
            "version": "1.0.0",
            "status": "running"
        }
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        doc_count = document_store.get_document_count()
        kg_stats = graph_store.get_graph_statistics()
        return {
            "status": "healthy",
            "documents_indexed": doc_count,
            "active_sessions": len(orchestrator.active_sessions),
            "knowledge_graph": {
                "entities": kg_stats.get("total_entities", 0),
                "relationships": kg_stats.get("total_relationships", 0)
            }
        }
    
    @app.post("/research", response_model=ResearchResponse)
    async def start_research(request: ResearchRequest, background_tasks: BackgroundTasks):
        """Start a research session."""
        try:
            logger.info(f"Starting research for query: {request.query}")
            
            if request.interactive:
                # Start interactive research in background
                session = await orchestrator.run_interactive_research(
                    request.query, 
                    request.session_id
                )
            else:
                # Start direct research in background
                session = await orchestrator.run_research(
                    request.query, 
                    request.session_id
                )
            
            return ResearchResponse(
                success=True,
                session_id=session.session_id,
                status=session.status,
                message="Research completed",
                report=session.final_report if session.status == "completed" else None
            )
            
        except Exception as e:
            logger.error(f"Research request failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/research/{session_id}")
    async def get_research_status(session_id: str):
        """Get status of a research session."""
        status = orchestrator.get_session_status(session_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return status
    
    @app.get("/research")
    async def list_research_sessions():
        """List all active research sessions."""
        return {
            "sessions": orchestrator.list_active_sessions()
        }
    
    @app.delete("/research/{session_id}")
    async def cleanup_research_session(session_id: str):
        """Clean up a research session."""
        success = await orchestrator.cleanup_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"message": "Session cleaned up successfully"}
    
    @app.post("/function-call")
    async def execute_function_call(request: FunctionCallRequest):
        """Execute a function call."""
        try:
            result = await function_manager.execute_function(
                request.function_name,
                request.arguments
            )
            return result
        except Exception as e:
            logger.error(f"Function call failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/functions")
    async def list_functions():
        """List available function calls."""
        return {
            "functions": function_manager.get_all_function_definitions()
        }
    
    @app.post("/search/documents")
    async def search_documents(request: DocumentSearchRequest):
        """Search internal documents."""
        try:
            results = document_store.search_documents(
                query=request.query,
                n_results=request.max_results,
                file_types=request.file_types
            )
            return {
                "success": True,
                "results": results,
                "total_found": len(results)
            }
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/documents")
    async def list_documents():
        """List all indexed documents."""
        try:
            documents = document_store.get_unique_documents()
            return {
                "documents": documents,
                "total_count": len(documents)
            }
        except Exception as e:
            logger.error(f"Document listing failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/documents/stats")
    async def get_document_stats():
        """Get document statistics."""
        try:
            total_chunks = document_store.get_document_count()
            unique_docs = document_store.get_unique_documents()
            
            # Group by file type
            file_type_counts = {}
            for doc in unique_docs:
                file_type = doc.get("file_type", "unknown")
                file_type_counts[file_type] = file_type_counts.get(file_type, 0) + 1
            
            return {
                "total_documents": len(unique_docs),
                "total_chunks": total_chunks,
                "file_type_distribution": file_type_counts
            }
        except Exception as e:
            logger.error(f"Document stats failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/search/hybrid")
    async def hybrid_search(request: HybridSearchRequest):
        """Perform hybrid search combining RAG and Knowledge Graph."""
        try:
            results = await hybrid_retriever.search(
                query=request.query,
                n_results=request.max_results,
                include_graph=request.include_graph,
                include_rag=request.include_rag,
                graph_depth=request.graph_depth,
                entity_boost=request.entity_boost
            )
            
            # Convert results to serializable format
            serialized_results = []
            for result in results:
                serialized_results.append({
                    "content": result.content,
                    "metadata": result.metadata,
                    "source_type": result.source_type,
                    "relevance_score": result.relevance_score,
                    "entities": result.entities or [],
                    "relationships": result.relationships or []
                })
            
            return {
                "success": True,
                "results": serialized_results,
                "total_found": len(serialized_results),
                "search_type": "hybrid"
            }
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/knowledge-graph/entity/expand")
    async def expand_entity(request: EntityExpansionRequest):
        """Get expanded information about an entity from the knowledge graph."""
        try:
            expansion = await hybrid_retriever.get_entity_expansion(
                request.entity_text,
                max_depth=request.max_depth
            )
            return {
                "success": True,
                "expansion": expansion
            }
        except Exception as e:
            logger.error(f"Entity expansion failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/knowledge-graph/suggestions/{query}")
    async def get_query_suggestions(query: str, limit: int = 5):
        """Get related query suggestions based on knowledge graph."""
        try:
            suggestions = await hybrid_retriever.suggest_related_queries(query, limit=limit)
            return {
                "success": True,
                "query": query,
                "suggestions": suggestions
            }
        except Exception as e:
            logger.error(f"Query suggestions failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/knowledge-graph/stats")
    async def get_knowledge_graph_stats():
        """Get knowledge graph statistics."""
        try:
            stats = graph_store.get_graph_statistics()
            return {
                "success": True,
                "statistics": stats
            }
        except Exception as e:
            logger.error(f"Knowledge graph stats failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/knowledge-graph/entities/{entity_type}")
    async def get_entities_by_type(entity_type: str, limit: int = 50):
        """Get entities by their type/label."""
        try:
            entities = graph_store.search_entities_by_type(entity_type, limit=limit)
            return {
                "success": True,
                "entity_type": entity_type,
                "entities": entities,
                "total_found": len(entities)
            }
        except Exception as e:
            logger.error(f"Entity search by type failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


if __name__ == "__main__":
    import uvicorn
    from src.config import settings
    
    app = create_app()
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)