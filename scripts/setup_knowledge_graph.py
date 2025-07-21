#!/usr/bin/env python3
"""Setup script for knowledge graph components."""

import asyncio
import subprocess
import sys
import os
from pathlib import Path
from loguru import logger

# Add the project root to Python path so we can import src modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def install_spacy_model():
    """Install the required spaCy model."""
    try:
        logger.info("Installing spaCy English model...")
        
        # Try different approaches for spaCy model installation
        commands_to_try = [
            ["uv", "run", "python", "-m", "spacy", "download", "en_core_web_sm"],
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
            ["python", "-m", "spacy", "download", "en_core_web_sm"]
        ]
        
        for cmd in commands_to_try:
            try:
                logger.debug(f"Trying command: {' '.join(cmd)}")
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                logger.info("spaCy model installed successfully")
                return True
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                logger.debug(f"Command {' '.join(cmd)} failed: {e}")
                continue
        
        # If all automated methods fail, try to check if model is already installed
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model already available")
            return True
        except (ImportError, OSError):
            pass
        
        logger.error("All spaCy installation methods failed")
        logger.info("Please install manually with: uv run python -m spacy download en_core_web_sm")
        return False
        
    except Exception as e:
        logger.error(f"Failed to install spaCy model: {e}")
        return False

def check_neo4j_connection():
    """Check if Neo4j is accessible."""
    try:
        from src.knowledge_graph.graph_store import KnowledgeGraphStore
        
        logger.info("Testing Neo4j connection...")
        graph_store = KnowledgeGraphStore()
        stats = graph_store.get_graph_statistics()
        graph_store.close()
        
        logger.info("Neo4j connection successful")
        logger.info(f"Current graph stats: {stats}")
        return True
        
    except Exception as e:
        logger.error(f"Neo4j connection failed: {e}")
        logger.error("Please ensure Neo4j is running and configured correctly")
        logger.error("Default connection: bolt://localhost:7687 (neo4j/password)")
        return False

def setup_neo4j_constraints():
    """Set up Neo4j constraints and indexes for better performance."""
    try:
        from src.knowledge_graph.graph_store import KnowledgeGraphStore
        
        logger.info("Setting up Neo4j constraints and indexes...")
        graph_store = KnowledgeGraphStore()
        
        with graph_store.driver.session() as session:
            # Create constraints for unique entity IDs
            try:
                session.run("CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
                logger.info("Created entity ID constraint")
            except Exception as e:
                logger.debug(f"Entity constraint may already exist: {e}")
            
            # Create constraints for unique document IDs
            try:
                session.run("CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
                logger.info("Created document ID constraint")
            except Exception as e:
                logger.debug(f"Document constraint may already exist: {e}")
            
            # Create indexes for better search performance
            try:
                session.run("CREATE INDEX entity_text_index IF NOT EXISTS FOR (e:Entity) ON (e.text)")
                session.run("CREATE INDEX entity_normalized_text_index IF NOT EXISTS FOR (e:Entity) ON (e.normalized_text)")
                session.run("CREATE INDEX entity_label_index IF NOT EXISTS FOR (e:Entity) ON (e.label)")
                logger.info("Created search indexes")
            except Exception as e:
                logger.debug(f"Indexes may already exist: {e}")
        
        graph_store.close()
        logger.info("Neo4j setup completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Neo4j setup failed: {e}")
        return False

async def test_hybrid_retrieval():
    """Test the hybrid retrieval system."""
    try:
        from src.knowledge_graph.hybrid_retriever import HybridRetriever
        
        logger.info("Testing hybrid retrieval system...")
        retriever = HybridRetriever()
        
        # Test with a simple query
        results = await retriever.search(
            query="test query",
            n_results=5,
            include_graph=True,
            include_rag=True
        )
        
        logger.info(f"Hybrid retrieval test successful - found {len(results)} results")
        retriever.close()
        return True
        
    except Exception as e:
        logger.error(f"Hybrid retrieval test failed: {e}")
        return False

def main():
    """Main setup function."""
    logger.info("Setting up Knowledge Graph + RAG system...")
    
    success = True
    
    # Step 1: Install spaCy model
    if not install_spacy_model():
        success = False
    
    # Step 2: Check Neo4j connection
    if not check_neo4j_connection():
        success = False
        logger.error("Please install and start Neo4j before continuing")
        logger.error("Installation instructions:")
        logger.error("1. Download Neo4j Desktop from https://neo4j.com/download/")
        logger.error("2. Create a new database with username 'neo4j' and password 'password'")
        logger.error("3. Start the database")
        logger.error("4. Update .env file with your Neo4j credentials if different")
        return False
    
    # Step 3: Set up Neo4j constraints and indexes
    if not setup_neo4j_constraints():
        success = False
    
    # Step 4: Test hybrid retrieval
    if not asyncio.run(test_hybrid_retrieval()):
        success = False
    
    if success:
        logger.info("✅ Knowledge Graph + RAG setup completed successfully!")
        logger.info("You can now:")
        logger.info("1. Ingest documents: python -m src.ingestion.ingest_documents --data-dir ./data")
        logger.info("2. Start the API server: python -m src.api.server")
        logger.info("3. Use hybrid search: POST /search/hybrid")
        logger.info("4. Explore entities: GET /knowledge-graph/stats")
    else:
        logger.error("❌ Setup completed with errors. Please check the logs above.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)