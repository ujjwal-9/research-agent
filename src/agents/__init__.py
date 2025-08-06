"""
Multi-agent research workflow system.

This module provides specialized agents for different aspects of research:
- Research planning and strategy
- Internal document analysis
- External web research
- Content synthesis and report generation
- Workflow orchestration
"""

from .base_agent import BaseAgent, AgentState
from .research_planner import ResearchPlannerAgent
from .document_analyst import DocumentAnalystAgent
from .web_researcher import WebResearcherAgent
from .synthesis_agent import SynthesisAgent
from .orchestrator import ResearchOrchestrator
from .chat_planner import ChatPlannerAgent

__all__ = [
    "BaseAgent",
    "AgentState",
    "ResearchPlannerAgent",
    "DocumentAnalystAgent",
    "WebResearcherAgent",
    "SynthesisAgent",
    "ResearchOrchestrator",
    "ChatPlannerAgent",
]
