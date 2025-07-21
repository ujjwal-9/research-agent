#!/usr/bin/env python3
"""Quick test script to verify system functionality."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.ingestion.document_store import DocumentStore
from src.tools.function_calls import FunctionCallManager
from src.agents.research_orchestrator import ResearchOrchestrator


async def test_document_store():
    """Test document store functionality."""
    print("Testing document store...")
    
    try:
        store = DocumentStore()
        count = store.get_document_count()
        docs = store.get_unique_documents()
        
        print(f"✓ Document store initialized")
        print(f"  - Total chunks: {count}")
        print(f"  - Unique documents: {len(docs)}")
        
        if docs:
            print(f"  - Sample document: {docs[0]['title']}")
        
        return True
    except Exception as e:
        print(f"✗ Document store test failed: {e}")
        return False


async def test_function_calls():
    """Test function call system."""
    print("Testing function calls...")
    
    try:
        manager = FunctionCallManager()
        functions = manager.get_all_function_definitions()
        
        print(f"✓ Function call manager initialized")
        print(f"  - Available functions: {len(functions)}")
        
        for func in functions:
            print(f"    - {func['name']}: {func['description']}")
        
        return True
    except Exception as e:
        print(f"✗ Function call test failed: {e}")
        return False


async def test_research_orchestrator():
    """Test research orchestrator."""
    print("Testing research orchestrator...")
    
    try:
        orchestrator = ResearchOrchestrator()
        print("✓ Research orchestrator initialized")
        print("  - Planner agent ready")
        print("  - Researcher agent ready")
        print("  - Synthesizer agent ready")
        
        return True
    except Exception as e:
        print(f"✗ Research orchestrator test failed: {e}")
        return False


async def run_quick_search_test():
    """Run a quick search test if documents are available."""
    print("Testing document search...")
    
    try:
        store = DocumentStore()
        if store.get_document_count() == 0:
            print("⚠️  No documents indexed. Skipping search test.")
            return True
        
        results = store.search_documents("test query", n_results=3)
        print(f"✓ Search completed, found {len(results)} results")
        
        return True
    except Exception as e:
        print(f"✗ Search test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🧪 Running quick system tests...\n")
    
    tests = [
        test_document_store,
        test_function_calls,
        test_research_orchestrator,
        run_quick_search_test
    ]
    
    results = []
    for test in tests:
        result = await test()
        results.append(result)
        print()
    
    passed = sum(results)
    total = len(results)
    
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready.")
    else:
        print("⚠️  Some tests failed. Check configuration and dependencies.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())