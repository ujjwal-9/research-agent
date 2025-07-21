"""Knowledge graph storage and retrieval using Neo4j."""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import json
from neo4j import GraphDatabase, AsyncGraphDatabase
from loguru import logger

from src.config import settings
from src.knowledge_graph.entity_extractor import Entity, Relationship


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph."""
    id: str
    label: str
    properties: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            **self.properties
        }


@dataclass
class GraphRelationship:
    """Represents a relationship in the knowledge graph."""
    source_id: str
    target_id: str
    relationship_type: str
    properties: Dict[str, Any]


class KnowledgeGraphStore:
    """Manages knowledge graph storage and retrieval using Neo4j."""
    
    def __init__(self):
        self.driver = None
        self._connect()
    
    def _connect(self):
        """Connect to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_username, settings.neo4j_password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Connected to Neo4j knowledge graph")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
    
    async def store_entities_and_relationships(
        self, 
        entities: List[Entity], 
        relationships: List[Relationship],
        document_id: str,
        chunk_id: str
    ) -> bool:
        """Store entities and relationships in the knowledge graph."""
        try:
            with self.driver.session() as session:
                # Create document node
                session.run(
                    """
                    MERGE (d:Document {id: $doc_id})
                    SET d.chunk_id = $chunk_id, d.updated_at = datetime()
                    """,
                    doc_id=document_id,
                    chunk_id=chunk_id
                )
                
                # Create entity nodes
                for entity in entities:
                    entity_id = self._generate_entity_id(entity)
                    session.run(
                        """
                        MERGE (e:Entity {id: $entity_id})
                        SET e.text = $text,
                            e.normalized_text = $normalized_text,
                            e.label = $label,
                            e.confidence = $confidence,
                            e.updated_at = datetime()
                        """,
                        entity_id=entity_id,
                        text=entity.text,
                        normalized_text=entity.normalized_text,
                        label=entity.label,
                        confidence=entity.confidence
                    )
                    
                    # Link entity to document
                    session.run(
                        """
                        MATCH (e:Entity {id: $entity_id})
                        MATCH (d:Document {id: $doc_id})
                        MERGE (e)-[:MENTIONED_IN]->(d)
                        """,
                        entity_id=entity_id,
                        doc_id=document_id
                    )
                
                # Create relationships
                for rel in relationships:
                    source_id = self._generate_entity_id(rel.source_entity)
                    target_id = self._generate_entity_id(rel.target_entity)
                    
                    session.run(
                        f"""
                        MATCH (s:Entity {{id: $source_id}})
                        MATCH (t:Entity {{id: $target_id}})
                        MERGE (s)-[r:{rel.relation_type}]->(t)
                        SET r.confidence = $confidence,
                            r.context = $context,
                            r.sentence = $sentence,
                            r.updated_at = datetime()
                        """,
                        source_id=source_id,
                        target_id=target_id,
                        confidence=rel.confidence,
                        context=rel.context,
                        sentence=rel.sentence
                    )
                
                logger.info(f"Stored {len(entities)} entities and {len(relationships)} relationships for {document_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error storing knowledge graph data: {e}")
            return False
    
    def _generate_entity_id(self, entity: Entity) -> str:
        """Generate consistent ID for entity."""
        return f"{entity.label}_{entity.normalized_text}".replace(" ", "_").replace("-", "_")
    
    def find_related_entities(
        self, 
        entity_text: str, 
        max_depth: int = 2,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Find entities related to the given entity."""
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (start:Entity)
                    WHERE start.text CONTAINS $entity_text OR start.normalized_text CONTAINS $entity_text
                    MATCH path = (start)-[*1..$max_depth]-(related:Entity)
                    WHERE start <> related
                    RETURN DISTINCT related.id as id,
                           related.text as text,
                           related.label as label,
                           related.confidence as confidence,
                           length(path) as distance
                    ORDER BY distance ASC, related.confidence DESC
                    LIMIT $limit
                    """,
                    entity_text=entity_text.lower(),
                    max_depth=max_depth,
                    limit=limit
                )
                
                related_entities = []
                for record in result:
                    related_entities.append({
                        "id": record["id"],
                        "text": record["text"],
                        "label": record["label"],
                        "confidence": record["confidence"],
                        "distance": record["distance"]
                    })
                
                return related_entities
                
        except Exception as e:
            logger.error(f"Error finding related entities: {e}")
            return []
    
    def find_entity_context(self, entity_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find documents and context where entity is mentioned."""
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (e:Entity)-[:MENTIONED_IN]->(d:Document)
                    WHERE e.text CONTAINS $entity_text OR e.normalized_text CONTAINS $entity_text
                    RETURN e.text as entity_text,
                           e.label as entity_label,
                           d.id as document_id,
                           d.chunk_id as chunk_id
                    LIMIT $limit
                    """,
                    entity_text=entity_text.lower(),
                    limit=limit
                )
                
                contexts = []
                for record in result:
                    contexts.append({
                        "entity_text": record["entity_text"],
                        "entity_label": record["entity_label"],
                        "document_id": record["document_id"],
                        "chunk_id": record["chunk_id"]
                    })
                
                return contexts
                
        except Exception as e:
            logger.error(f"Error finding entity context: {e}")
            return []
    
    def get_entity_relationships(self, entity_text: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get all relationships for a specific entity."""
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (e:Entity)-[r]-(related:Entity)
                    WHERE e.text CONTAINS $entity_text OR e.normalized_text CONTAINS $entity_text
                    RETURN e.text as source_entity,
                           type(r) as relationship_type,
                           related.text as target_entity,
                           r.confidence as confidence,
                           r.context as context
                    ORDER BY r.confidence DESC
                    LIMIT $limit
                    """,
                    entity_text=entity_text.lower(),
                    limit=limit
                )
                
                relationships = []
                for record in result:
                    relationships.append({
                        "source_entity": record["source_entity"],
                        "relationship_type": record["relationship_type"],
                        "target_entity": record["target_entity"],
                        "confidence": record["confidence"],
                        "context": record["context"]
                    })
                
                return relationships
                
        except Exception as e:
            logger.error(f"Error getting entity relationships: {e}")
            return []
    
    def search_entities_by_type(self, entity_type: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search entities by their type/label."""
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (e:Entity {label: $entity_type})
                    RETURN e.id as id,
                           e.text as text,
                           e.normalized_text as normalized_text,
                           e.confidence as confidence
                    ORDER BY e.confidence DESC
                    LIMIT $limit
                    """,
                    entity_type=entity_type,
                    limit=limit
                )
                
                entities = []
                for record in result:
                    entities.append({
                        "id": record["id"],
                        "text": record["text"],
                        "normalized_text": record["normalized_text"],
                        "confidence": record["confidence"]
                    })
                
                return entities
                
        except Exception as e:
            logger.error(f"Error searching entities by type: {e}")
            return []
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        try:
            with self.driver.session() as session:
                # Count entities
                entity_result = session.run("MATCH (e:Entity) RETURN count(e) as count")
                entity_count = entity_result.single()["count"]
                
                # Count relationships
                rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                relationship_count = rel_result.single()["count"]
                
                # Count documents
                doc_result = session.run("MATCH (d:Document) RETURN count(d) as count")
                document_count = doc_result.single()["count"]
                
                # Get entity type distribution
                type_result = session.run(
                    """
                    MATCH (e:Entity)
                    RETURN e.label as type, count(e) as count
                    ORDER BY count DESC
                    """
                )
                entity_types = [{"type": record["type"], "count": record["count"]} 
                              for record in type_result]
                
                return {
                    "total_entities": entity_count,
                    "total_relationships": relationship_count,
                    "total_documents": document_count,
                    "entity_types": entity_types
                }
                
        except Exception as e:
            logger.error(f"Error getting graph statistics: {e}")
            return {}
    
    def clear_graph(self) -> bool:
        """Clear all data from the knowledge graph."""
        try:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
                logger.info("Cleared knowledge graph")
                return True
        except Exception as e:
            logger.error(f"Error clearing graph: {e}")
            return False