"""Base agent class for the multi-agent research system."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio

from openai import AsyncOpenAI
from loguru import logger

from src.config import settings


class AgentRole(Enum):
    """Defines different agent roles in the research workflow."""
    PLANNER = "planner"
    RESEARCHER = "researcher"
    SYNTHESIZER = "synthesizer"
    VALIDATOR = "validator"


@dataclass
class AgentMessage:
    """Represents a message between agents."""
    sender: str
    recipient: str
    content: str
    message_type: str
    metadata: Dict[str, Any] = None


@dataclass
class AgentState:
    """Represents the current state of an agent."""
    agent_id: str
    role: AgentRole
    status: str
    current_task: Optional[str] = None
    context: Dict[str, Any] = None
    messages: List[AgentMessage] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}
        if self.messages is None:
            self.messages = []


class BaseAgent(ABC):
    """Abstract base class for all agents in the research system."""
    
    def __init__(self, agent_id: str, role: AgentRole):
        self.agent_id = agent_id
        self.role = role
        self.state = AgentState(agent_id=agent_id, role=role, status="initialized")
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        
        logger.info(f"Initialized {role.value} agent: {agent_id}")
    
    @abstractmethod
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task and return results."""
        pass
    
    async def send_message(self, recipient: str, content: str, message_type: str, metadata: Dict[str, Any] = None):
        """Send a message to another agent."""
        message = AgentMessage(
            sender=self.agent_id,
            recipient=recipient,
            content=content,
            message_type=message_type,
            metadata=metadata or {}
        )
        self.state.messages.append(message)
        logger.debug(f"Agent {self.agent_id} sent message to {recipient}: {message_type}")
        return message
    
    async def receive_message(self, message: AgentMessage):
        """Receive and process a message from another agent."""
        self.state.messages.append(message)
        logger.debug(f"Agent {self.agent_id} received message from {message.sender}: {message.message_type}")
    
    async def update_status(self, status: str, current_task: str = None):
        """Update agent status."""
        self.state.status = status
        if current_task:
            self.state.current_task = current_task
        logger.debug(f"Agent {self.agent_id} status: {status}")
    
    async def call_llm(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """Make a call to the language model."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling LLM for agent {self.agent_id}: {e}")
            raise
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        base_prompt = f"""You are a {self.role.value} agent in a multi-agent research system.
Your agent ID is {self.agent_id}.

Your role is to {self._get_role_description()}.

Always be thorough, accurate, and collaborative with other agents.
Provide clear reasoning for your decisions and actions.
"""
        return base_prompt
    
    def _get_role_description(self) -> str:
        """Get description of the agent's role."""
        role_descriptions = {
            AgentRole.PLANNER: "analyze research requests, identify ambiguities, ask clarifying questions, and create comprehensive research plans",
            AgentRole.RESEARCHER: "execute research tasks by searching internal documents and external sources to gather relevant information",
            AgentRole.SYNTHESIZER: "combine and synthesize research findings into coherent, well-structured reports",
            AgentRole.VALIDATOR: "review and validate research outputs for accuracy, completeness, and quality"
        }
        return role_descriptions.get(self.role, "perform specialized research tasks")
    
    async def cleanup(self):
        """Cleanup resources when agent is done."""
        await self.update_status("completed")
        logger.info(f"Agent {self.agent_id} cleanup completed")