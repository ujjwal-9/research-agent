"""Tests for Knowledge Graph Ingestion System."""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ingestion_knowledge_graph.config import KnowledgeGraphConfig
from ingestion_knowledge_graph.extractor import (
    Entity,
    Relationship,
    KnowledgeGraphData,
    KnowledgeGraphExtractor,
)
from ingestion_knowledge_graph.neo4j_manager import Neo4jManager


class TestKnowledgeGraphConfig:
    """Test Knowledge Graph Configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test_key", "NEO4J_PASSWORD": "test_password"},
        ):
            config = KnowledgeGraphConfig()

            assert config.neo4j_uri == "neo4j://127.0.0.1:7687"
            assert config.neo4j_username == "neo4j"
            assert config.neo4j_password == "test_password"
            assert config.neo4j_database == "neo4j"
            assert config.openai_api_key == "test_key"
            assert config.knowledge_graph_model == "gpt-4o"
            assert config.max_chunk_size == 4000
            assert config.batch_size == 5

    def test_config_validation(self):
        """Test configuration validation."""
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test_key", "NEO4J_PASSWORD": "test_password"},
        ):
            config = KnowledgeGraphConfig()
            assert config.validate() is True

        # Test missing API key
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                KnowledgeGraphConfig()

    def test_entity_types(self):
        """Test default entity types."""
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test_key", "NEO4J_PASSWORD": "test_password"},
        ):
            config = KnowledgeGraphConfig()

            assert "PERSON" in config.entity_types
            assert "ORGANIZATION" in config.entity_types
            assert "LOCATION" in config.entity_types
            assert len(config.entity_types) > 5


class TestEntity:
    """Test Entity class."""

    def test_entity_creation(self):
        """Test entity creation and properties."""
        entity = Entity(
            id="person_john_doe",
            name="John Doe",
            type="PERSON",
            description="Software engineer",
            properties={"age": 30},
        )

        assert entity.id == "person_john_doe"
        assert entity.name == "John Doe"
        assert entity.type == "PERSON"
        assert entity.description == "Software engineer"
        assert entity.properties["age"] == 30

    def test_entity_default_properties(self):
        """Test entity with default properties."""
        entity = Entity(
            id="org_acme",
            name="ACME Corp",
            type="ORGANIZATION",
            description="Technology company",
        )

        assert entity.properties == {}


class TestRelationship:
    """Test Relationship class."""

    def test_relationship_creation(self):
        """Test relationship creation."""
        relationship = Relationship(
            source_entity_id="person_john_doe",
            target_entity_id="org_acme",
            relationship_type="WORKS_FOR",
            description="John works for ACME",
            confidence=0.95,
            properties={"since": "2020"},
        )

        assert relationship.source_entity_id == "person_john_doe"
        assert relationship.target_entity_id == "org_acme"
        assert relationship.relationship_type == "WORKS_FOR"
        assert relationship.confidence == 0.95
        assert relationship.properties["since"] == "2020"

    def test_relationship_defaults(self):
        """Test relationship with default values."""
        relationship = Relationship(
            source_entity_id="entity1",
            target_entity_id="entity2",
            relationship_type="RELATED_TO",
            description="Generic relationship",
        )

        assert relationship.confidence == 1.0
        assert relationship.properties == {}


class TestKnowledgeGraphData:
    """Test KnowledgeGraphData container."""

    def test_knowledge_graph_data_creation(self):
        """Test knowledge graph data container."""
        entities = [
            Entity("person_john", "John", "PERSON", "A person"),
            Entity("org_acme", "ACME", "ORGANIZATION", "A company"),
        ]

        relationships = [
            Relationship("person_john", "org_acme", "WORKS_FOR", "Employment")
        ]

        kg_data = KnowledgeGraphData(
            entities=entities,
            relationships=relationships,
            source_document="test.pdf",
            chunk_id="test_chunk_1",
        )

        assert len(kg_data.entities) == 2
        assert len(kg_data.relationships) == 1
        assert kg_data.source_document == "test.pdf"
        assert kg_data.chunk_id == "test_chunk_1"

    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        entities = [Entity("person_john", "John", "PERSON", "A person")]
        relationships = [
            Relationship(
                "person_john", "org_acme", "WORKS_FOR", "Employment", confidence=0.9
            )
        ]

        kg_data = KnowledgeGraphData(
            entities=entities,
            relationships=relationships,
            source_document="test.pdf",
            chunk_id="test_chunk_1",
        )

        data_dict = kg_data.to_dict()

        assert "entities" in data_dict
        assert "relationships" in data_dict
        assert data_dict["source_document"] == "test.pdf"
        assert data_dict["chunk_id"] == "test_chunk_1"
        assert len(data_dict["entities"]) == 1
        assert data_dict["entities"][0]["name"] == "John"
        assert data_dict["relationships"][0]["confidence"] == 0.9


class TestKnowledgeGraphExtractor:
    """Test Knowledge Graph Extractor."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test_key", "NEO4J_PASSWORD": "test_password"},
        ):
            return KnowledgeGraphConfig()

    @pytest.fixture
    def extractor(self, mock_config):
        """Create extractor with mock configuration."""
        with patch("ingestion_knowledge_graph.extractor.OpenAI"):
            return KnowledgeGraphExtractor(mock_config)

    def test_extractor_initialization(self, mock_config):
        """Test extractor initialization."""
        with patch("ingestion_knowledge_graph.extractor.OpenAI") as mock_openai:
            extractor = KnowledgeGraphExtractor(mock_config)

            assert extractor.config == mock_config
            mock_openai.assert_called_once_with(api_key="test_key")

    def test_system_prompt_generation(self, extractor):
        """Test system prompt generation."""
        prompt = extractor._get_system_prompt()

        assert "knowledge graph extraction expert" in prompt.lower()
        assert "PERSON" in prompt
        assert "ORGANIZATION" in prompt
        assert "JSON" in prompt
        assert str(extractor.config.max_entities_per_chunk) in prompt

    def test_extraction_prompt_creation(self, extractor):
        """Test extraction prompt creation."""
        text = "John Smith works for Microsoft Corporation."
        context = "Business document"

        prompt = extractor._create_extraction_prompt(text, context)

        assert text in prompt
        assert context in prompt
        assert "extract entities and relationships" in prompt.lower()

    def test_parse_entities(self, extractor):
        """Test entity parsing from API response."""
        entities_data = [
            {
                "id": "person_john_smith",
                "name": "John Smith",
                "type": "PERSON",
                "description": "Software engineer",
                "properties": {"role": "developer"},
            },
            {
                "id": "org_microsoft",
                "name": "Microsoft Corporation",
                "type": "ORGANIZATION",
                "description": "Technology company",
            },
        ]

        entities = extractor._parse_entities(entities_data)

        assert len(entities) == 2
        assert entities[0].name == "John Smith"
        assert entities[0].type == "PERSON"
        assert entities[1].name == "Microsoft Corporation"
        assert entities[1].type == "ORGANIZATION"

    def test_parse_relationships(self, extractor):
        """Test relationship parsing from API response."""
        relationships_data = [
            {
                "source_entity_id": "person_john_smith",
                "target_entity_id": "org_microsoft",
                "relationship_type": "WORKS_FOR",
                "description": "Employment relationship",
                "confidence": 0.95,
            }
        ]

        relationships = extractor._parse_relationships(relationships_data)

        assert len(relationships) == 1
        assert relationships[0].source_entity_id == "person_john_smith"
        assert relationships[0].target_entity_id == "org_microsoft"
        assert relationships[0].relationship_type == "WORKS_FOR"
        assert relationships[0].confidence == 0.95

    def test_validate_and_clean(self, extractor):
        """Test validation and cleaning of extracted data."""
        entities = [
            Entity("entity1", "Entity 1", "PERSON", "Description 1"),
            Entity("entity2", "Entity 2", "ORGANIZATION", "Description 2"),
            Entity(
                "entity1", "Entity 1 Duplicate", "PERSON", "Duplicate"
            ),  # Duplicate ID
        ]

        relationships = [
            Relationship("entity1", "entity2", "WORKS_FOR", "Valid relationship"),
            Relationship("entity1", "entity3", "KNOWS", "Invalid - entity3 not found"),
            Relationship("entity1", "entity1", "SELF_REF", "Invalid - self reference"),
        ]

        clean_entities, clean_relationships = extractor._validate_and_clean(
            entities, relationships
        )

        # Should remove duplicate entity
        assert len(clean_entities) == 2
        entity_ids = {e.id for e in clean_entities}
        assert "entity1" in entity_ids
        assert "entity2" in entity_ids

        # Should keep only valid relationship
        assert len(clean_relationships) == 1
        assert clean_relationships[0].source_entity_id == "entity1"
        assert clean_relationships[0].target_entity_id == "entity2"


class TestNeo4jManager:
    """Test Neo4j Manager."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test_key", "NEO4J_PASSWORD": "test_password"},
        ):
            return KnowledgeGraphConfig()

    @pytest.fixture
    def neo4j_manager(self, mock_config):
        """Create Neo4j manager with mock configuration."""
        return Neo4jManager(mock_config)

    def test_neo4j_manager_initialization(self, mock_config):
        """Test Neo4j manager initialization."""
        manager = Neo4jManager(mock_config)

        assert manager.config == mock_config
        assert manager.driver is None

    @patch("ingestion_knowledge_graph.neo4j_manager.GraphDatabase")
    def test_connection_success(self, mock_graph_db, neo4j_manager):
        """Test successful Neo4j connection."""
        # Mock driver and session
        mock_driver = Mock()
        mock_session = Mock()
        mock_result = Mock()
        mock_result.single.return_value = {"test": 1}

        mock_session.run.return_value = mock_result
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_graph_db.driver.return_value = mock_driver

        success = neo4j_manager.connect()

        assert success is True
        assert neo4j_manager.driver == mock_driver
        mock_graph_db.driver.assert_called_once()

    @patch("ingestion_knowledge_graph.neo4j_manager.GraphDatabase")
    def test_connection_failure(self, mock_graph_db, neo4j_manager):
        """Test Neo4j connection failure."""
        from neo4j.exceptions import ServiceUnavailable

        mock_graph_db.driver.side_effect = ServiceUnavailable("Connection failed")

        success = neo4j_manager.connect()

        assert success is False
        assert neo4j_manager.driver is None

    def test_disconnect(self, neo4j_manager):
        """Test Neo4j disconnection."""
        mock_driver = Mock()
        neo4j_manager.driver = mock_driver

        neo4j_manager.disconnect()

        mock_driver.close.assert_called_once()


# Integration test (requires running Neo4j)
@pytest.mark.integration
class TestKnowledgeGraphIntegration:
    """Integration tests for the full knowledge graph system."""

    def test_end_to_end_processing(self):
        """Test end-to-end processing with a sample document."""
        # This test requires a running Neo4j instance
        # Skip if Neo4j is not available
        pytest.skip("Integration test requires running Neo4j instance")

    def test_document_processing_pipeline(self):
        """Test full document processing pipeline."""
        # This test would create a temporary document and process it
        pytest.skip("Integration test requires running Neo4j instance")


if __name__ == "__main__":
    pytest.main([__file__])
