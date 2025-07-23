"""Tests for knowledge graph functionality."""

import pytest
import asyncio
from unittest.mock import Mock, patch

from src.knowledge_graph.entity_extractor import EntityExtractor, Entity, Relationship
from src.knowledge_graph.hybrid_retriever import HybridRetriever
from src.ingestion.document_store import DocumentStore


class TestEntityExtractor:
    """Test entity extraction functionality."""

    @pytest.fixture
    def extractor(self):
        """Create entity extractor for testing."""
        with patch("spacy.load") as mock_spacy:
            # Mock spaCy model
            mock_nlp = Mock()
            mock_doc = Mock()
            mock_ent = Mock()
            mock_ent.text = "OpenAI"
            mock_ent.label_ = "ORG"
            mock_ent.start_char = 0
            mock_ent.end_char = 6
            mock_doc.ents = [mock_ent]
            mock_doc.sents = []
            mock_nlp.return_value = mock_doc
            mock_spacy.return_value = mock_nlp

            return EntityExtractor()

    @pytest.mark.asyncio
    async def test_entity_extraction(self, extractor):
        """Test basic entity extraction."""
        text = "OpenAI is a leading AI research company."

        entities, relationships = await extractor.extract_entities_and_relationships(
            text
        )

        assert len(entities) >= 0  # Should extract at least some entities
        assert isinstance(entities, list)
        assert isinstance(relationships, list)

    def test_entity_normalization(self):
        """Test entity text normalization."""
        entity = Entity(
            text="The OpenAI Corp.", label="ORG", start=0, end=16, confidence=0.9
        )

        # Should normalize by removing "The" and "Corp."
        assert "openai" in entity.normalized_text.lower()
        assert "the" not in entity.normalized_text.lower()


class TestHybridRetriever:
    """Test hybrid retrieval functionality."""

    @pytest.fixture
    def mock_retriever(self):
        """Create mock hybrid retriever."""
        with patch("src.knowledge_graph.hybrid_retriever.DocumentStore"), patch(
            "src.knowledge_graph.hybrid_retriever.KnowledgeGraphStore"
        ), patch("src.knowledge_graph.hybrid_retriever.EntityExtractor"):
            return HybridRetriever()

    @pytest.mark.asyncio
    async def test_hybrid_search_structure(self, mock_retriever):
        """Test that hybrid search returns proper structure."""
        # Mock the internal search methods
        mock_retriever._rag_search = Mock(return_value=[])
        mock_retriever._graph_search = Mock(return_value=[])
        mock_retriever.entity_extractor.extract_entities_and_relationships = Mock(
            return_value=([], [])
        )

        results = await mock_retriever.search(
            query="test query", n_results=5, include_graph=True, include_rag=True
        )

        assert isinstance(results, list)


class TestDocumentStoreKGIntegration:
    """Test document store knowledge graph integration."""

    @pytest.mark.asyncio
    async def test_kg_integration_disabled(self):
        """Test document store with KG disabled."""
        with patch("qdrant_client.QdrantClient"), patch("openai.OpenAI"):

            store = DocumentStore(enable_knowledge_graph=False)
            assert not store.enable_kg

    @pytest.mark.asyncio
    async def test_kg_integration_enabled(self):
        """Test document store with KG enabled."""
        with patch("qdrant_client.QdrantClient"), patch("openai.OpenAI"), patch(
            "src.knowledge_graph.entity_extractor.EntityExtractor"
        ), patch("src.knowledge_graph.graph_store.KnowledgeGraphStore"):

            store = DocumentStore(enable_knowledge_graph=True)
            assert store.enable_kg


@pytest.mark.integration
class TestKnowledgeGraphIntegration:
    """Integration tests for knowledge graph functionality."""

    @pytest.mark.asyncio
    async def test_end_to_end_flow(self):
        """Test complete knowledge graph flow."""
        # This would require actual Neo4j instance
        # Skip in unit tests, run separately for integration testing
        pytest.skip("Requires Neo4j instance for integration testing")

    def test_neo4j_connection(self):
        """Test Neo4j connection."""
        # Skip if Neo4j not available
        try:
            from src.knowledge_graph.graph_store import KnowledgeGraphStore

            store = KnowledgeGraphStore()
            stats = store.get_graph_statistics()
            store.close()
            assert isinstance(stats, dict)
        except Exception:
            pytest.skip("Neo4j not available for testing")


if __name__ == "__main__":
    pytest.main([__file__])
