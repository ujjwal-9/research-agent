#!/usr/bin/env python3
"""Test script to verify connection cleanup works properly."""

import asyncio
import warnings
from src.agents.research_orchestrator import ResearchOrchestrator

async def test_cleanup():
    """Test that connections are properly cleaned up."""
    print("Testing connection cleanup...")
    
    # Enable ResourceWarning to see if we still get them
    warnings.filterwarnings("default", category=ResourceWarning)
    
    async with ResearchOrchestrator() as orchestrator:
        print("✓ Orchestrator created and will be cleaned up automatically")
        
        # You could run a simple task here if needed
        # session = await orchestrator.run_research("test query")
    
    print("✓ Orchestrator cleaned up")
    print("If you don't see ResourceWarnings above, the fix is working!")

if __name__ == "__main__":
    asyncio.run(test_cleanup())