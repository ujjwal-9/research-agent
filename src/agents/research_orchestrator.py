"""Main orchestrator for the multi-agent research workflow."""

import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import json
from loguru import logger

from src.agents.planner_agent import PlannerAgent, ResearchPlan
from src.agents.researcher_agent import ResearcherAgent
from src.agents.synthesizer_agent import SynthesizerAgent
from src.config import settings


@dataclass
class ResearchSession:
    """Represents a complete research session."""
    session_id: str
    query: str
    plan: Optional[ResearchPlan] = None
    research_results: List[Any] = None
    final_report: str = ""
    status: str = "initialized"
    
    def __post_init__(self):
        if self.research_results is None:
            self.research_results = []


class ResearchOrchestrator:
    """Orchestrates the multi-agent research workflow."""
    
    def __init__(self):
        self.planner = PlannerAgent()
        self.researcher = ResearcherAgent()
        self.synthesizer = SynthesizerAgent()
        self.active_sessions: Dict[str, ResearchSession] = {}
    
    async def run_research(self, query: str, session_id: str = None) -> ResearchSession:
        """Run complete research workflow without user interaction."""
        if session_id is None:
            import uuid
            session_id = str(uuid.uuid4())
        
        session = ResearchSession(session_id=session_id, query=query)
        self.active_sessions[session_id] = session
        
        try:
            # Step 1: Create research plan
            logger.info(f"Starting research for query: {query}")
            session.status = "planning"
            
            plan_result = await self.planner.process_task({
                "type": "create_plan",
                "query": query,
                "context": {}
            })
            
            if not plan_result["success"]:
                session.status = "failed"
                session.final_report = f"Planning failed: {plan_result.get('error', 'Unknown error')}"
                return session
            
            session.plan = plan_result["plan"]
            
            # Step 2: Execute research
            session.status = "researching"
            research_result = await self.researcher.process_task({
                "type": "execute_research_plan",
                "plan": session.plan
            })
            
            if not research_result["success"]:
                session.status = "failed"
                session.final_report = f"Research failed: {research_result.get('error', 'Unknown error')}"
                return session
            
            session.research_results = research_result["results"]
            
            # Step 3: Synthesize results
            session.status = "synthesizing"
            synthesis_result = await self.synthesizer.process_task({
                "type": "synthesize_report",
                "query": query,
                "plan": session.plan,
                "research_results": session.research_results
            })
            
            if not synthesis_result["success"]:
                session.status = "failed"
                session.final_report = f"Synthesis failed: {synthesis_result.get('error', 'Unknown error')}"
                return session
            
            session.final_report = synthesis_result["report"]
            session.status = "completed"
            
            logger.info(f"Research completed successfully for session {session_id}")
            return session
            
        except Exception as e:
            logger.error(f"Research workflow failed: {e}")
            session.status = "failed"
            session.final_report = f"Workflow error: {str(e)}"
            return session
    
    async def run_interactive_research(self, query: str, session_id: str = None) -> ResearchSession:
        """Run research workflow with user interaction and confirmation."""
        if session_id is None:
            import uuid
            session_id = str(uuid.uuid4())
        
        session = ResearchSession(session_id=session_id, query=query)
        self.active_sessions[session_id] = session
        
        try:
            # Step 1: Create initial research plan
            logger.info(f"Creating research plan for: {query}")
            session.status = "planning"
            
            plan_result = await self.planner.process_task({
                "type": "create_plan",
                "query": query,
                "context": {}
            })
            
            if not plan_result["success"]:
                session.status = "failed"
                session.final_report = f"Planning failed: {plan_result.get('error', 'Unknown error')}"
                return session
            
            session.plan = plan_result["plan"]
            
            # Step 2: Check if clarifications are needed
            if plan_result.get("requires_clarification", False):
                print("\\n" + "="*60)
                print("CLARIFICATIONS NEEDED")
                print("="*60)
                for clarification in session.plan.clarifications_needed:
                    print(f"• {clarification}")
                
                # In a real interactive system, you would wait for user input here
                print("\\nProceeding with current understanding...")
            
            # Step 3: Present research plan for confirmation
            print("\\n" + "="*60)
            print("RESEARCH PLAN")
            print("="*60)
            print(f"Query: {session.plan.query}")
            print(f"\\nObjectives:")
            for obj in session.plan.research_objectives:
                print(f"• {obj}")
            
            print(f"\\nInternal Search Tasks ({len(session.plan.internal_search_tasks)}):")
            for task in session.plan.internal_search_tasks:
                print(f"• {task['description']} (Priority: {task.get('priority', 'medium')})")
            
            print(f"\\nExternal Search Tasks ({len(session.plan.external_search_tasks)}):")
            for task in session.plan.external_search_tasks:
                print(f"• {task['description']} (Priority: {task.get('priority', 'medium')})")
            
            print(f"\\nEstimated Time: {session.plan.estimated_time}")
            
            # In a real interactive system, you would ask for user confirmation here
            print("\\nProceeding with research execution...")
            
            # Step 4: Execute research
            session.status = "researching"
            research_result = await self.researcher.process_task({
                "type": "execute_research_plan",
                "plan": session.plan
            })
            
            if not research_result["success"]:
                session.status = "failed"
                session.final_report = f"Research failed: {research_result.get('error', 'Unknown error')}"
                return session
            
            session.research_results = research_result["results"]
            
            # Step 5: Synthesize results
            session.status = "synthesizing"
            synthesis_result = await self.synthesizer.process_task({
                "type": "synthesize_report",
                "query": query,
                "plan": session.plan,
                "research_results": session.research_results
            })
            
            if not synthesis_result["success"]:
                session.status = "failed"
                session.final_report = f"Synthesis failed: {synthesis_result.get('error', 'Unknown error')}"
                return session
            
            session.final_report = synthesis_result["report"]
            session.status = "completed"
            
            logger.info(f"Interactive research completed successfully for session {session_id}")
            return session
            
        except Exception as e:
            logger.error(f"Interactive research workflow failed: {e}")
            session.status = "failed"
            session.final_report = f"Workflow error: {str(e)}"
            return session
    
    async def refine_plan(self, session_id: str, feedback: str) -> bool:
        """Refine research plan based on user feedback."""
        if session_id not in self.active_sessions:
            logger.error(f"Session {session_id} not found")
            return False
        
        session = self.active_sessions[session_id]
        
        if not session.plan:
            logger.error(f"No plan found for session {session_id}")
            return False
        
        try:
            refinement_result = await self.planner.process_task({
                "type": "refine_plan",
                "plan": session.plan,
                "feedback": feedback
            })
            
            if refinement_result["success"]:
                session.plan = refinement_result["plan"]
                logger.info(f"Plan refined for session {session_id}")
                return True
            else:
                logger.error(f"Plan refinement failed: {refinement_result.get('error')}")
                return False
                
        except Exception as e:
            logger.error(f"Error refining plan: {e}")
            return False
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a research session."""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        return {
            "session_id": session.session_id,
            "query": session.query,
            "status": session.status,
            "has_plan": session.plan is not None,
            "results_count": len(session.research_results),
            "has_report": bool(session.final_report)
        }
    
    def list_active_sessions(self) -> List[Dict[str, Any]]:
        """List all active research sessions."""
        return [self.get_session_status(sid) for sid in self.active_sessions.keys()]
    
    async def cleanup_session(self, session_id: str) -> bool:
        """Clean up a research session."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Cleaned up session {session_id}")
            return True
        return False