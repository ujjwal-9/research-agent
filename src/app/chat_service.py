"""
Chat service integration for the research workflow.
"""

import asyncio
import uuid
import logging
from typing import Dict, Optional
from datetime import datetime

from ..agents.chat_planner import ChatPlannerAgent, ChatSession
from ..research_workflow import ResearchWorkflowManager
from .models import ChatResponse, ResearchPlanResponse


class ChatService:
    """Service for managing chat sessions and research workflows."""

    def __init__(self):
        self.sessions: Dict[str, ChatSession] = {}
        self.chat_agents: Dict[str, ChatPlannerAgent] = {}
        self.research_managers: Dict[str, ResearchWorkflowManager] = {}

        # Setup logging
        self.logger = logging.getLogger(__name__)

    async def start_chat_session(
        self, initial_question: str
    ) -> tuple[str, ChatResponse]:
        """Start a new chat session."""
        try:
            session_id = str(uuid.uuid4())

            # Create chat agent
            chat_agent = ChatPlannerAgent()
            self.chat_agents[session_id] = chat_agent

            # Start chat session
            chat_session = await chat_agent.start_chat_session(initial_question)
            self.sessions[session_id] = chat_session

            # Get the initial response
            assistant_message = ""
            if chat_session.messages:
                assistant_messages = [
                    msg for msg in chat_session.messages if msg.role == "assistant"
                ]
                if assistant_messages:
                    assistant_message = assistant_messages[-1].content

            response = ChatResponse(
                message=assistant_message,
                session_id=session_id,
                plan_available=False,
                plan_approved=False,
            )

            return session_id, response

        except Exception as e:
            self.logger.error(f"Error starting chat session: {e}")
            raise

    async def send_message(self, session_id: str, user_message: str) -> ChatResponse:
        """Send a message to an existing chat session."""
        try:
            if session_id not in self.chat_agents:
                raise ValueError(f"Session {session_id} not found")

            chat_agent = self.chat_agents[session_id]
            response_message = await chat_agent.continue_chat(user_message)

            # Check if a plan was generated
            plan = chat_agent.get_research_plan()
            plan_available = plan is not None
            plan_approved = (
                self.sessions[session_id].plan_approved
                if session_id in self.sessions
                else False
            )

            response = ChatResponse(
                message=response_message,
                session_id=session_id,
                plan_available=plan_available,
                plan_approved=plan_approved,
            )

            return response

        except Exception as e:
            self.logger.error(f"Error sending message: {e}")
            raise

    def get_research_plan(self, session_id: str) -> Optional[ResearchPlanResponse]:
        """Get the current research plan for a session."""
        try:
            if session_id not in self.chat_agents:
                raise ValueError(f"Session {session_id} not found")

            chat_agent = self.chat_agents[session_id]
            plan = chat_agent.get_research_plan()

            if not plan:
                return None

            return ResearchPlanResponse(
                research_question=plan.research_question,
                research_methodology=plan.research_methodology,
                internal_search_queries=plan.internal_search_queries,
                external_search_topics=plan.external_search_topics,
                estimated_timeline=plan.estimated_timeline,
                success_criteria=plan.success_criteria,
            )

        except Exception as e:
            self.logger.error(f"Error getting research plan: {e}")
            raise

    async def start_research(
        self, session_id: str, use_code_interpreter: bool = True
    ) -> Dict:
        """Start the research workflow for a session."""
        try:
            if session_id not in self.chat_agents:
                raise ValueError(f"Session {session_id} not found")

            chat_agent = self.chat_agents[session_id]
            plan = chat_agent.get_research_plan()

            if not plan:
                raise ValueError("No research plan available")

            # Initialize research manager
            research_manager = ResearchWorkflowManager()
            self.research_managers[session_id] = research_manager

            # Get user requirements from session
            session = self.sessions.get(session_id)
            user_requirements = session.user_requirements or {} if session else {}

            # Execute research
            results = await research_manager.conduct_research(
                research_question=plan.research_question,
                user_requirements=user_requirements,
                export_format="markdown",
                research_plan=plan,
                use_code_interpreter=use_code_interpreter,
            )

            return results

        except Exception as e:
            self.logger.error(f"Error starting research: {e}")
            raise

    def get_session_status(self, session_id: str) -> Dict:
        """Get the status of a session."""
        try:
            if session_id not in self.sessions:
                return {"exists": False}

            session = self.sessions[session_id]
            chat_agent = self.chat_agents.get(session_id)

            plan = None
            if chat_agent:
                plan = chat_agent.get_research_plan()

            return {
                "exists": True,
                "research_question": session.research_question,
                "refined_question": session.refined_question,
                "plan_available": plan is not None,
                "plan_approved": session.plan_approved,
                "message_count": len(session.messages),
            }

        except Exception as e:
            self.logger.error(f"Error getting session status: {e}")
            raise

    def cleanup_session(self, session_id: str):
        """Clean up a session and its resources."""
        try:
            if session_id in self.sessions:
                del self.sessions[session_id]
            if session_id in self.chat_agents:
                del self.chat_agents[session_id]
            if session_id in self.research_managers:
                del self.research_managers[session_id]

        except Exception as e:
            self.logger.error(f"Error cleaning up session: {e}")
