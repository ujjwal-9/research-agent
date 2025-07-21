"""Hybrid retrieval combining RAG and Knowledge Graph."""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

from src.ingestion.document_store import DocumentStore
from src.knowledge_graph.graph_store import KnowledgeGraphStore
from src.knowledge_graph.entity_extractor import EntityExtractor


@dataclass
class HybridResult:
    """Combined result from RAG and Knowledge Graph retrieval."""
    content: str
    metadata: Dict[str, Any]
    source_type: str  # "rag", "graph", or "hybrid"
    relevance_score: float
    entities: List[Dict[str, Any]] = None
    relationships: List[Dict[str, Any]] = None


class HybridRetriever:
    """Combines RAG and Knowledge Graph for enhanced retrieval."""
    
    def __init__(self):
        self.document_store = DocumentStore()
        self.graph_store = KnowledgeGraphStore()
        self.entity_extractor = EntityExtractor()
        
    async def search(
        self,
        query: str,
        n_results: int = 10,
        include_graph: bool = True,
        include_rag: bool = True,
        graph_depth: int = 2,
        entity_boost: float = 1.5
    ) -> List[HybridResult]:
        """Perform hybrid search combining RAG and Knowledge Graph."""
        
        results = []
        
        # Extract entities from query for graph search
        query_entities = []
        if include_graph:
            try:
                entities, _ = await self.entity_extractor.extract_entities_and_relationships(query)
                query_entities = [e.normalized_text for e in entities]
                logger.debug(f"Extracted {len(query_entities)} entities from query: {query_entities}")
            except Exception as e:
                logger.warning(f"Failed to extract entities from query: {e}")
        
        # Perform searches in parallel
        tasks = []
        
        if include_rag:
            tasks.append(self._rag_search(query, n_results))
        
        if include_graph and query_entities:
            tasks.append(self._graph_search(query_entities, n_results, graph_depth))
        
        if tasks:
            search_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process RAG results
            if include_rag and len(search_results) > 0 and not isinstance(search_results[0], Exception):
                rag_results = search_results[0]
                for result in rag_results:
                    hybrid_result = HybridResult(
                        content=result["content"],
                        metadata=result["metadata"],
                        source_type="rag",
                        relevance_score=self._calculate_rag_score(result)
                    )
                    results.append(hybrid_result)
            
            # Process Graph results
            graph_idx = 1 if include_rag else 0
            if include_graph and len(search_results) > graph_idx and not isinstance(search_results[graph_idx], Exception):
                graph_results = search_results[graph_idx]
                for result in graph_results:
                    hybrid_result = HybridResult(
                        content=result["content"],
                        metadata=result["metadata"],
                        source_type="graph",
                        relevance_score=result["relevance_score"],
                        entities=result.get("entities", []),
                        relationships=result.get("relationships", [])
                    )
                    results.append(hybrid_result)
        
        # Enhance RAG results with graph information
        if include_graph and include_rag:
            results = await self._enhance_with_graph_context(results, query_entities, entity_boost)
        
        # Sort by relevance score and limit results
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:n_results]
    
    async def _rag_search(self, query: str, n_results: int) -> List[Dict[str, Any]]:
        """Perform RAG search."""
        try:
            return self.document_store.search_documents(query, n_results * 2)  # Get more for filtering
        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            return []
    
    async def _graph_search(
        self, 
        query_entities: List[str], 
        n_results: int, 
        max_depth: int
    ) -> List[Dict[str, Any]]:
        """Perform knowledge graph search."""
        try:
            graph_results = []
            
            for entity_text in query_entities:
                # Find related entities
                related_entities = self.graph_store.find_related_entities(
                    entity_text, max_depth=max_depth, limit=n_results
                )
                
                # Get entity context (documents where entities are mentioned)
                entity_contexts = self.graph_store.find_entity_context(entity_text, limit=n_results)
                
                # Get relationships
                relationships = self.graph_store.get_entity_relationships(entity_text, limit=n_results)
                
                # Convert to results format
                for context in entity_contexts:
                    # Get document content from RAG store
                    doc_chunks = self.document_store.get_document_by_path(context["document_id"])
                    
                    for chunk in doc_chunks:
                        if chunk["metadata"].get("chunk_index") == context.get("chunk_id"):
                            result = {
                                "content": chunk["content"],
                                "metadata": {
                                    **chunk["metadata"],
                                    "source_entity": entity_text,
                                    "entity_label": context["entity_label"]
                                },
                                "relevance_score": 0.8,  # Base score for graph results
                                "entities": related_entities,
                                "relationships": relationships
                            }
                            graph_results.append(result)
                            break
            
            return graph_results
            
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return []
    
    async def _enhance_with_graph_context(
        self, 
        results: List[HybridResult], 
        query_entities: List[str],
        entity_boost: float
    ) -> List[HybridResult]:
        """Enhance RAG results with knowledge graph context."""
        
        enhanced_results = []
        
        for result in results:
            if result.source_type == "rag":
                # Extract entities from the result content
                try:
                    content_entities, content_relationships = await self.entity_extractor.extract_entities_and_relationships(
                        result.content
                    )
                    
                    # Check for entity overlap with query
                    entity_overlap = 0
                    content_entity_texts = [e.normalized_text for e in content_entities]
                    
                    for query_entity in query_entities:
                        if any(query_entity in content_entity for content_entity in content_entity_texts):
                            entity_overlap += 1
                    
                    # Boost score based on entity overlap
                    if entity_overlap > 0:
                        result.relevance_score *= (1 + entity_boost * entity_overlap / len(query_entities))
                        result.source_type = "hybrid"
                    
                    # Add entity and relationship information
                    result.entities = [
                        {
                            "text": e.text,
                            "label": e.label,
                            "confidence": e.confidence
                        } for e in content_entities
                    ]
                    
                    result.relationships = [
                        {
                            "source": r.source_entity.text,
                            "target": r.target_entity.text,
                            "type": r.relation_type,
                            "confidence": r.confidence
                        } for r in content_relationships
                    ]
                    
                except Exception as e:
                    logger.warning(f"Failed to enhance result with graph context: {e}")
            
            enhanced_results.append(result)
        
        return enhanced_results
    
    def _calculate_rag_score(self, result: Dict[str, Any]) -> float:
        """Calculate relevance score for RAG result."""
        # Use distance if available (lower distance = higher relevance)
        if "distance" in result and result["distance"] is not None:
            return max(0.0, 1.0 - result["distance"])
        
        # Default score
        return 0.7
    
    async def get_entity_expansion(self, entity_text: str, max_depth: int = 2) -> Dict[str, Any]:
        """Get expanded information about an entity from the knowledge graph."""
        try:
            # Get related entities
            related_entities = self.graph_store.find_related_entities(
                entity_text, max_depth=max_depth, limit=20
            )
            
            # Get relationships
            relationships = self.graph_store.get_entity_relationships(entity_text, limit=20)
            
            # Get context documents
            contexts = self.graph_store.find_entity_context(entity_text, limit=10)
            
            return {
                "entity": entity_text,
                "related_entities": related_entities,
                "relationships": relationships,
                "contexts": contexts
            }
            
        except Exception as e:
            logger.error(f"Error getting entity expansion: {e}")
            return {}
    
    async def suggest_related_queries(self, query: str, limit: int = 5) -> List[str]:
        """Suggest related queries based on knowledge graph."""
        try:
            # Extract entities from query
            entities, _ = await self.entity_extractor.extract_entities_and_relationships(query)
            
            suggestions = []
            
            for entity in entities[:3]:  # Limit to top 3 entities
                # Find related entities
                related = self.graph_store.find_related_entities(
                    entity.normalized_text, max_depth=1, limit=10
                )
                
                # Generate query suggestions
                for rel_entity in related[:2]:  # Top 2 related entities
                    suggestion = f"{entity.text} and {rel_entity['text']}"
                    if suggestion not in suggestions:
                        suggestions.append(suggestion)
            
            return suggestions[:limit]
            
        except Exception as e:
            logger.error(f"Error generating query suggestions: {e}")
            return []
    
    def close(self):
        """Close connections."""
        self.document_store.close()
        self.graph_store.close()