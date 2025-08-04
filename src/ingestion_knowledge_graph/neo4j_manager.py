"""Neo4j Database Manager for Knowledge Graph Storage."""

import logging
from typing import Dict, Any, Optional
from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable, AuthError, ClientError

from .config import KnowledgeGraphConfig
from .extractor import Entity, Relationship, KnowledgeGraphData


class Neo4jManager:
    """Manages Neo4j database operations for knowledge graph storage."""

    def __init__(self, config: KnowledgeGraphConfig):
        """Initialize Neo4j manager."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.driver: Optional[Driver] = None

    def connect(self) -> bool:
        """
        Establish connection to Neo4j database.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.logger.info(f"🔌 Connecting to Neo4j at {self.config.neo4j_uri}")

            self.driver = GraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_username, self.config.neo4j_password),
                database=self.config.neo4j_database,
            )

            # Test the connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]

            if test_value == 1:
                self.logger.info("✅ Successfully connected to Neo4j")
                return True
            else:
                self.logger.error("❌ Connection test failed")
                return False

        except AuthError as e:
            self.logger.error(f"❌ Authentication failed: {e}")
            return False
        except ServiceUnavailable as e:
            self.logger.error(f"❌ Neo4j service unavailable: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Neo4j: {e}")
            return False

    def disconnect(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
            self.logger.info("🔌 Disconnected from Neo4j")

    def setup_schema(self):
        """
        Set up Neo4j schema with constraints and indexes.
        This should be called once to initialize the database.
        """
        self.logger.info("🏗️  Setting up Neo4j schema")

        schema_queries = [
            # Create uniqueness constraints for entity IDs
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            # Create indexes for better performance
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX document_source_index IF NOT EXISTS FOR (d:Document) ON (d.source)",
            # Create constraint for document sources
            "CREATE CONSTRAINT document_source_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.source IS UNIQUE",
        ]

        with self.driver.session() as session:
            for query in schema_queries:
                try:
                    session.run(query)
                    self.logger.debug(f"✅ Executed: {query}")
                except ClientError as e:
                    if "already exists" in str(e).lower():
                        self.logger.debug(f"⚠️  Schema element already exists: {query}")
                    else:
                        self.logger.warning(f"⚠️  Schema setup warning: {e}")

        self.logger.info("✅ Schema setup completed")

    def store_knowledge_graph(self, kg_data: KnowledgeGraphData) -> bool:
        """
        Store knowledge graph data in Neo4j.

        Args:
            kg_data: KnowledgeGraphData object containing entities and relationships

        Returns:
            bool: True if storage successful, False otherwise
        """
        if not self.driver:
            self.logger.error("❌ No Neo4j connection available")
            return False

        try:
            with self.driver.session() as session:
                # Start a transaction
                with session.begin_transaction() as tx:
                    # Store document node
                    self._store_document(tx, kg_data.source_document)

                    # Store entities
                    for entity in kg_data.entities:
                        self._store_entity(tx, entity, kg_data)

                    # Store relationships
                    for relationship in kg_data.relationships:
                        self._store_relationship(tx, relationship, kg_data)

                    # Commit transaction
                    tx.commit()

            # After successful commit, try to add APOC labels in separate transactions
            # This won't affect the main data if APOC is not available
            for entity in kg_data.entities:
                self._add_entity_label_if_apoc_available(entity.id, entity.type)

            self.logger.info(
                f"✅ Stored knowledge graph: {len(kg_data.entities)} entities, "
                f"{len(kg_data.relationships)} relationships"
            )
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to store knowledge graph: {e}")
            return False

    def _store_document(self, tx, source_document: str):
        """Store or update document node."""
        query = """
        MERGE (d:Document {source: $source})
        ON CREATE SET d.created_at = datetime(), d.updated_at = datetime()
        ON MATCH SET d.updated_at = datetime()
        """
        tx.run(query, source=source_document)

    def _store_entity(self, tx, entity: Entity, kg_data: KnowledgeGraphData):
        """Store entity in Neo4j."""

        try:
            # Serialize properties to ensure Neo4j compatibility
            # Always serialize, even if empty, to catch any nested structures
            raw_properties = (
                entity.properties
                if hasattr(entity, "properties") and entity.properties
                else {}
            )
            properties = self._serialize_properties(raw_properties)

            # Debug logging to catch problematic properties
            if raw_properties:
                self.logger.debug(
                    f"Entity {entity.id} raw properties: {raw_properties}"
                )
                self.logger.debug(
                    f"Entity {entity.id} serialized properties: {properties}"
                )

            # Prepare parameters, excluding properties if empty to avoid Neo4j Map issues
            params = {
                "id": entity.id,
                "name": entity.name,
                "type": entity.type,
                "description": entity.description,
            }

            # Add flattened properties directly to params
            # This avoids Neo4j Map issues by storing each property as a separate field
            for prop_key, prop_value in properties.items():
                # Prefix property keys to avoid conflicts with standard fields
                safe_key = f"prop_{prop_key}"
                params[safe_key] = prop_value

            # Build dynamic SET clauses for properties
            if properties:
                create_props = ", ".join(
                    [f"e.{key} = ${key}" for key in params if key.startswith("prop_")]
                )
                match_props = ", ".join(
                    [f"e.{key} = ${key}" for key in params if key.startswith("prop_")]
                )

                query = f"""
                MERGE (e:Entity {{id: $id}})
                ON CREATE SET 
                    e.name = $name,
                    e.type = $type,
                    e.description = $description,
                    {create_props},
                    e.created_at = datetime(),
                    e.updated_at = datetime()
                ON MATCH SET
                    e.name = $name,
                    e.description = CASE WHEN $description <> '' THEN $description ELSE e.description END,
                    {match_props},
                    e.updated_at = datetime()
                """
            else:
                # Query without properties
                query = """
                MERGE (e:Entity {id: $id})
                ON CREATE SET 
                    e.name = $name,
                    e.type = $type,
                    e.description = $description,
                    e.created_at = datetime(),
                    e.updated_at = datetime()
                ON MATCH SET
                    e.name = $name,
                    e.description = CASE WHEN $description <> '' THEN $description ELSE e.description END,
                    e.updated_at = datetime()
                """

            tx.run(query, **params)
        except Exception as e:
            self.logger.error(f"Failed to store entity {entity.id}: {e}")
            raise

        # Link entity to document and chunk
        link_query = """
        MATCH (e:Entity {id: $entity_id})
        MATCH (d:Document {source: $source_document})
        MERGE (e)-[r:MENTIONED_IN]->(d)
        ON CREATE SET r.chunk_id = $chunk_id, r.created_at = datetime()
        ON MATCH SET r.updated_at = datetime()
        """

        tx.run(
            link_query,
            entity_id=entity.id,
            source_document=kg_data.source_document,
            chunk_id=kg_data.chunk_id,
        )

    def _add_entity_label_if_apoc_available(self, entity_id: str, entity_type: str):
        """Add type-specific label using APOC if available, in a separate transaction."""
        try:
            with self.driver.session() as session:
                label_query = f"MATCH (e:Entity {{id: $id}}) CALL apoc.create.addLabels(e, ['{entity_type}']) YIELD node RETURN node"
                session.run(label_query, id=entity_id)
                self.logger.debug(f"Added {entity_type} label to entity {entity_id}")
        except Exception as e:
            # APOC is not available, which is fine - we'll use type property instead
            self.logger.debug(f"APOC not available for entity labeling: {e}")
            pass

    def _store_relationship(
        self, tx, relationship: Relationship, kg_data: KnowledgeGraphData
    ):
        """Store relationship in Neo4j."""

        try:
            # Serialize properties to ensure Neo4j compatibility
            properties = self._serialize_properties(relationship.properties or {})

            # Prepare base parameters
            params = {
                "source_id": relationship.source_entity_id,
                "target_id": relationship.target_entity_id,
                "description": relationship.description,
                "confidence": relationship.confidence,
                "source_document": kg_data.source_document,
                "chunk_id": kg_data.chunk_id,
            }

            # Create relationship between entities with conditional properties
            if properties:
                # Query with properties
                query_with_props = f"""
                MATCH (source:Entity {{id: $source_id}})
                MATCH (target:Entity {{id: $target_id}})
                MERGE (source)-[r:{relationship.relationship_type}]->(target)
                ON CREATE SET 
                    r.description = $description,
                    r.confidence = $confidence,
                    r.properties = $properties,
                    r.source_document = $source_document,
                    r.chunk_id = $chunk_id,
                    r.created_at = datetime(),
                    r.updated_at = datetime()
                ON MATCH SET
                    r.description = CASE WHEN $description <> '' THEN $description ELSE r.description END,
                    r.confidence = CASE WHEN $confidence > r.confidence THEN $confidence ELSE r.confidence END,
                    r.properties = $properties,
                    r.updated_at = datetime()
                """
                params["properties"] = properties
                tx.run(query_with_props, **params)
            else:
                # Query without properties
                query_no_props = f"""
                MATCH (source:Entity {{id: $source_id}})
                MATCH (target:Entity {{id: $target_id}})
                MERGE (source)-[r:{relationship.relationship_type}]->(target)
                ON CREATE SET 
                    r.description = $description,
                    r.confidence = $confidence,
                    r.source_document = $source_document,
                    r.chunk_id = $chunk_id,
                    r.created_at = datetime(),
                    r.updated_at = datetime()
                ON MATCH SET
                    r.description = CASE WHEN $description <> '' THEN $description ELSE r.description END,
                    r.confidence = CASE WHEN $confidence > r.confidence THEN $confidence ELSE r.confidence END,
                    r.updated_at = datetime()
                """
                tx.run(query_no_props, **params)
        except Exception as e:
            self.logger.error(
                f"Failed to store relationship {relationship.relationship_type}: {e}"
            )
            raise

    def get_entity_count(self) -> int:
        """Get total number of entities in the graph."""
        if not self.driver:
            return 0

        with self.driver.session() as session:
            result = session.run("MATCH (e:Entity) RETURN count(e) as count")
            return result.single()["count"]

    def get_relationship_count(self) -> int:
        """Get total number of relationships in the graph."""
        if not self.driver:
            return 0

        with self.driver.session() as session:
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            return result.single()["count"]

    def get_document_count(self) -> int:
        """Get total number of documents in the graph."""
        if not self.driver:
            return 0

        with self.driver.session() as session:
            result = session.run("MATCH (d:Document) RETURN count(d) as count")
            return result.single()["count"]

    def clear_database(self) -> bool:
        """
        Clear all data from the database. Use with caution!

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self.driver.session() as session:
                # Delete all relationships first, then nodes
                session.run("MATCH ()-[r]->() DELETE r")
                session.run("MATCH (n) DELETE n")

            self.logger.info("✅ Database cleared successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to clear database: {e}")
            return False

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the knowledge graph."""
        if not self.driver:
            return {}

        stats = {}

        with self.driver.session() as session:
            # Basic counts
            stats["entities"] = session.run(
                "MATCH (e:Entity) RETURN count(e) as count"
            ).single()["count"]
            stats["relationships"] = session.run(
                "MATCH ()-[r]->() RETURN count(r) as count"
            ).single()["count"]
            stats["documents"] = session.run(
                "MATCH (d:Document) RETURN count(d) as count"
            ).single()["count"]

            # Entity types distribution
            entity_types_result = session.run(
                """
                MATCH (e:Entity) 
                RETURN e.type as type, count(e) as count 
                ORDER BY count DESC
            """
            )
            stats["entity_types"] = {
                record["type"]: record["count"] for record in entity_types_result
            }

            # Relationship types distribution
            rel_types_result = session.run(
                """
                MATCH ()-[r]->() 
                RETURN type(r) as type, count(r) as count 
                ORDER BY count DESC
            """
            )
            stats["relationship_types"] = {
                record["type"]: record["count"] for record in rel_types_result
            }

            # Most connected entities
            connected_entities_result = session.run(
                """
                MATCH (e:Entity)
                OPTIONAL MATCH (e)-[r]-()
                RETURN e.name as name, e.type as type, count(r) as connections
                ORDER BY connections DESC
                LIMIT 10
            """
            )
            stats["most_connected_entities"] = [
                {
                    "name": record["name"],
                    "type": record["type"],
                    "connections": record["connections"],
                }
                for record in connected_entities_result
            ]

        return stats

    def _serialize_properties(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize properties to ensure Neo4j compatibility.

        Neo4j only supports primitive types and arrays of primitives as properties.
        This method converts complex objects to JSON strings.

        Args:
            properties: Dictionary of properties

        Returns:
            Dictionary with Neo4j-compatible properties
        """
        import json

        # If empty dictionary, return empty dict (Neo4j handles this fine)
        if not properties:
            return {}

        serialized = {}

        def flatten_and_serialize(obj, prefix=""):
            """Recursively flatten and serialize any object to Neo4j-compatible format."""
            if obj is None:
                return None
            elif isinstance(obj, (str, int, float, bool)):
                return obj
            elif isinstance(obj, dict):
                if not obj:  # Empty dict
                    return None
                # Flatten dictionary keys with dot notation
                flattened = {}
                for k, v in obj.items():
                    new_key = f"{prefix}.{k}" if prefix else k
                    flat_value = flatten_and_serialize(v, new_key)
                    if flat_value is not None:
                        if isinstance(flat_value, dict):
                            flattened.update(flat_value)
                        else:
                            flattened[new_key] = flat_value
                return flattened
            elif isinstance(obj, (list, tuple)):
                # Handle arrays
                if not obj:  # Empty list
                    return None
                # Check if all elements are primitives
                if all(
                    isinstance(item, (str, int, float, bool, type(None)))
                    for item in obj
                ):
                    clean_list = [item for item in obj if item is not None]
                    return clean_list if clean_list else None
                else:
                    # Convert to JSON string for complex arrays
                    try:
                        return json.dumps([str(item) for item in obj])
                    except:
                        return str(obj)
            else:
                # For any other object type, convert to string
                return str(obj)

        for key, value in properties.items():
            result = flatten_and_serialize(value)
            if result is not None:
                if isinstance(result, dict):
                    # If flattening returned a dict, add all its key-value pairs with prefixed keys
                    for flat_key, flat_value in result.items():
                        final_key = f"{key}_{flat_key}" if flat_key != key else key
                        serialized[final_key] = flat_value
                else:
                    serialized[key] = result

        return serialized
