# Conversational Agentic Research System

A scalable system for document ingestion, indexing, and multi-agent research workflows with natural language interface.

## Overview

This system ingests documents (text, Word, spreadsheets, images), indexes content, and provides a conversational research interface powered by orchestrated multi-agent workflows. Built for the Redesign Health takehome assignment.

## Features

- **Document Ingestion**: Supports text, Word, Excel, PowerPoint, PDF, and image files with OCR
- **Knowledge Graph + RAG**: Hybrid retrieval combining vector search with entity relationships
- **Multi-Agent Research**: Structured workflow with planning, execution, and synthesis agents
- **LLM-Enhanced Entity Extraction**: Advanced entity and relationship extraction using OpenAI LLMs combined with spaCy
- **Graph Database**: Neo4j integration for storing and querying knowledge graphs
- **Tool Integration**: OpenAI-compatible function calls and MCP support
- **Async Processing**: Efficient parallelization and state management
- **Web Search Integration**: Combines internal documents with external DuckDuckGo search
- **REST API**: FastAPI server with comprehensive endpoints including graph queries
- **Evaluation Framework**: Built-in system performance evaluation

## Quick Start

### 1. Setup Environment

```bash
# Clone and navigate to the repository
cd redesign-research-system

# Install dependencies
pip install -r requirements.txt

# Set up knowledge graph components
python scripts/setup_knowledge_graph.py

# Configure environment variables
cp .env.example .env
# Edit .env with your OpenAI API key and Neo4j credentials
```

### Prerequisites

- **Neo4j Database**: Install and run Neo4j for knowledge graph storage
  - Download from [neo4j.com/download](https://neo4j.com/download/)
  - Create database with username `neo4j` and password `password` (or update .env)
  - Start the database before running the system

- **spaCy Model**: English language model for entity extraction
  - Automatically installed by setup script
  - Manual install: `python -m spacy download en_core_web_sm`

### 2. Ingest Documents

```bash
# Ingest documents recursively from the data directory and all subdirectories
python -m src.main ingest --data-dir ./data

# Advanced ingestion options
python -m src.main ingest --data-dir ./data --max-depth 3 --exclude-dirs temp --exclude-dirs cache

# Check ingestion status
python -m src.main status
```

### 3. Run Research

```bash
# Run a research query
python -m src.main research -q "What are the main challenges in clinical trial communication?"

# Interactive mode with plan confirmation
python -m src.main research -q "Summarize the UPMC pilot study findings" --interactive

# Save report to file
python -m src.main research -q "Analyze competitor landscape" -o report.md
```

### 4. Start API Server

```bash
# Start the REST API server
python -m src.main serve

# API will be available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

## Architecture

```
src/
├── agents/              # Multi-agent workflow
│   ├── base_agent.py           # Base agent class
│   ├── planner_agent.py        # Research planning
│   ├── researcher_agent.py     # Search execution (hybrid RAG+KG)
│   ├── synthesizer_agent.py    # Report synthesis
│   └── research_orchestrator.py # Workflow coordination
├── ingestion/           # Document processing
│   ├── document_processor.py   # File type handlers
│   ├── document_store.py       # Vector database + KG integration
│   └── ingest_documents.py     # Ingestion pipeline
├── knowledge_graph/     # Knowledge Graph components
│   ├── entity_extractor.py    # Entity and relationship extraction
│   ├── graph_store.py          # Neo4j graph database
│   └── hybrid_retriever.py     # Combined RAG + KG retrieval
├── tools/               # External integrations
│   ├── function_calls.py       # OpenAI function calls
│   └── web_search.py           # DuckDuckGo search
├── api/                 # API interfaces
│   ├── server.py               # FastAPI server with KG endpoints
│   └── mcp_server.py           # MCP protocol
├── evaluation/          # Performance assessment
│   └── evaluate_system.py      # Evaluation framework
└── main.py              # CLI interface
```

## Usage Examples

### CLI Commands

```bash
# System management
python -m src.main setup          # Initial setup
python -m src.main status         # Check system status
python -m src.main test           # Run system tests

# Document management (recursive by default)
python -m src.main ingest -d ./data                    # Ingest all documents recursively
python -m src.main ingest -d ./data --force-reindex    # Force reindexing all files
python -m src.main ingest -d ./data --max-depth 2      # Limit recursion depth
python -m src.main ingest -d ./data --exclude-dirs temp --exclude-dirs .git  # Exclude directories
python -m src.main search -q "clinical trials" -n 5    # Search documents

# Research operations
python -m src.main research -q "Your research question"
python -m src.main research -q "Question" --interactive
python -m src.main evaluate                             # Run evaluation

# API server
python -m src.main serve --host 0.0.0.0 --port 8000
```

## Knowledge Graph + RAG

The system combines traditional RAG (Retrieval-Augmented Generation) with a knowledge graph for enhanced search and discovery.

### How It Works

1. **LLM-Enhanced Entity Extraction**: During document ingestion, entities and relationships are extracted using a hybrid approach combining spaCy with OpenAI's LLM for superior accuracy and domain adaptation
2. **Graph Storage**: Entities and relationships are stored in Neo4j, creating a knowledge graph of your documents
3. **Hybrid Retrieval**: Search queries use both vector similarity (RAG) and graph traversal to find relevant information
4. **Enhanced Context**: Results include not just similar text, but related entities and their connections

### Knowledge Graph Features

- **LLM-Enhanced Entity Recognition**: Extracts 12+ entity types with superior accuracy using OpenAI models
- **Domain-Aware Extraction**: Adapts extraction based on document domain (healthcare, finance, technology, etc.)
- **Advanced Relationship Detection**: Uses LLM understanding to identify complex relationships
- **Hybrid Approach**: Combines traditional NLP (spaCy) with LLM capabilities for best results
- **Entity Linking**: Connects the same entities mentioned across different documents
- **Graph Traversal**: Finds information through entity relationships, not just text similarity
- **Query Expansion**: Suggests related queries based on entity connections

### LLM Entity Extraction

The system now includes advanced LLM-based entity extraction:

```bash
# Extract entities from text with domain context
python -m src.main extract-entities -t "Dr. Smith prescribed metformin for diabetes." -d "Healthcare"

# Compare extraction methods
python -m src.main extract-entities -t "Your text" --use-hybrid  # spaCy + LLM
python -m src.main extract-entities -t "Your text" --llm-only    # LLM only

# Batch processing and API integration available
```

**Benefits of LLM Extraction:**
- Better recognition of domain-specific entities
- Understanding of context and relationships
- Improved accuracy over traditional NLP
- Adaptable to different document types

### API Usage

```python
import requests

# Hybrid search (RAG + Knowledge Graph)
response = requests.post("http://localhost:8000/search/hybrid", json={
    "query": "What companies are mentioned in relation to AI research?",
    "max_results": 10,
    "include_graph": True,
    "include_rag": True,
    "entity_boost": 1.5
})

# Get entity expansion
response = requests.post("http://localhost:8000/knowledge-graph/entity/expand", json={
    "entity_text": "OpenAI",
    "max_depth": 2
})

# LLM-based entity extraction
response = requests.post("http://localhost:8000/extract/entities", json={
    "text": "Apple Inc. partners with Intel for processor development.",
    "domain_context": "Technology and business partnerships",
    "use_hybrid": True
})

# Get knowledge graph statistics
response = requests.get("http://localhost:8000/knowledge-graph/stats")

# Get query suggestions based on entities
response = requests.get("http://localhost:8000/knowledge-graph/suggestions/artificial intelligence")

# Start research (now uses hybrid retrieval)
response = requests.post("http://localhost:8000/research", json={
    "query": "What are the key findings from consumer insights?",
    "interactive": False
})

# Traditional document search
response = requests.post("http://localhost:8000/search/documents", json={
    "query": "clinical trial efficiency",
    "max_results": 10
})

# List available functions
response = requests.get("http://localhost:8000/functions")
```

### MCP Integration

Add to your `.kiro/settings/mcp.json`:

```json
{
  "mcpServers": {
    "research-system": {
      "command": "python",
      "args": ["-m", "src.api.mcp_server"],
      "env": {},
      "disabled": false,
      "autoApprove": ["search_documents", "list_documents"]
    }
  }
}
```

## Multi-Agent Workflow

The system implements a structured research workflow:

1. **Planning Agent**: Analyzes queries, identifies ambiguities, creates research plans
2. **Researcher Agent**: Executes parallel search tasks (internal + external)
3. **Synthesizer Agent**: Combines findings into coherent reports

### Workflow Features

- **Clarification Handling**: Proactively identifies query ambiguities
- **Plan Confirmation**: Interactive mode allows user review/modification
- **Parallel Execution**: Internal and external searches run concurrently
- **Source Integration**: Seamlessly combines document and web sources
- **Quality Assessment**: Built-in confidence scoring and validation

## Performance Evaluation

The system includes comprehensive evaluation capabilities:

```bash
# Run default evaluation
python -m src.main evaluate

# Custom query set
python -m src.main evaluate --queries-file custom_queries.json

# Save detailed results
python -m src.main evaluate --output detailed_results.json
```

### Evaluation Metrics

- **Response Time**: Query processing speed
- **Document Coverage**: Percentage of relevant docs found
- **Source Diversity**: Variety of sources utilized
- **Answer Completeness**: How thoroughly queries are addressed
- **Factual Accuracy**: Accuracy assessment of findings
- **Coherence Score**: Report structure and flow quality
- **Confidence Score**: System confidence in results

## Configuration

Key environment variables in `.env`:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4-1106-preview

# LLM Entity Extraction Configuration
USE_LLM_EXTRACTION=true
LLM_EXTRACTION_MODEL=gpt-4-turbo-preview
LLM_MAX_TEXT_LENGTH=4000
LLM_EXTRACTION_TEMPERATURE=0.1

# Vector Database
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=documents
QDRANT_STORAGE_PATH=./qdrant_storage
EMBEDDING_MODEL=text-embedding-3-small

# Processing Settings
DOCUMENT_CHUNK_SIZE=1000
DOCUMENT_CHUNK_OVERLAP=200
MAX_CONCURRENT_TASKS=5

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
```

## Supported File Types

- **Text**: `.txt`
- **Word**: `.docx`
- **Excel**: `.xlsx`, `.xls`
- **PowerPoint**: `.pptx`, `.ppt`
- **PDF**: `.pdf`
- **Images**: `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp` (with OCR)

## Testing

```bash
# Run unit tests
pytest tests/

# Quick system test
python -m src.main test

# Performance evaluation
python -m src.main evaluate
```

## Development

### Adding New File Types

1. Extend `DocumentProcessor` in `src/ingestion/document_processor.py`
2. Add processing method for new file type
3. Update supported extensions list

### Adding New Agents

1. Inherit from `BaseAgent` in `src/agents/base_agent.py`
2. Implement `process_task` method
3. Register with orchestrator

### Extending Function Calls

1. Create new tool class in `src/tools/`
2. Add to `FunctionCallManager`
3. Update MCP server if needed

## Presentation Notes

This system demonstrates:

- **Technical Excellence**: Robust async architecture, comprehensive error handling
- **AI Integration**: Thoughtful LLM orchestration with function calling
- **Scalability**: Modular design supporting easy extension
- **User Experience**: Both programmatic and interactive interfaces
- **Quality Assurance**: Built-in evaluation and testing frameworks

The implementation showcases modern AI system design principles while solving real research workflow challenges.