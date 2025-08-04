"""
Base agent class and shared utilities for the multi-agent research system.
"""

import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class AgentState(Enum):
    """Possible states for agents during execution."""

    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class AgentContext:
    """Shared context between agents."""

    research_question: str
    user_requirements: Dict[str, Any] = field(default_factory=dict)
    shared_data: Dict[str, Any] = field(default_factory=dict)
    agent_results: Dict[str, Any] = field(default_factory=dict)
    execution_metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)


@dataclass
class AgentResult:
    """Result from an agent's execution."""

    agent_name: str
    status: AgentState
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    error: Optional[str] = None


class BaseAgent(ABC):
    """Base class for all research agents."""

    def __init__(self, name: str):
        """Initialize the base agent.

        Args:
            name: Unique name for this agent
        """
        self.name = name
        self.logger = logging.getLogger(f"agent.{name}")
        self.state = AgentState.IDLE
        self.context: Optional[AgentContext] = None

    @abstractmethod
    async def execute(self, context: AgentContext) -> AgentResult:
        """Execute the agent's main functionality.

        Args:
            context: Shared context between agents

        Returns:
            AgentResult with execution outcome
        """
        pass

    def update_state(self, new_state: AgentState):
        """Update the agent's current state.

        Args:
            new_state: New state to transition to
        """
        old_state = self.state
        self.state = new_state
        self.logger.debug(f"State transition: {old_state.value} -> {new_state.value}")

    def validate_context(self, context: AgentContext) -> bool:
        """Validate that the context contains required information.

        Args:
            context: Context to validate

        Returns:
            True if context is valid
        """
        if not context.research_question:
            self.logger.error("Missing research question in context")
            return False
        return True

    async def _execute_with_state_management(
        self, context: AgentContext
    ) -> AgentResult:
        """Execute agent with proper state management.

        Args:
            context: Shared context

        Returns:
            AgentResult with execution outcome
        """
        start_time = datetime.now()
        self.context = context

        try:
            self.update_state(AgentState.EXECUTING)

            if not self.validate_context(context):
                raise ValueError("Invalid context provided")

            result = await self.execute(context)

            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time

            if result.status == AgentState.ERROR:
                self.update_state(AgentState.ERROR)
            else:
                self.update_state(AgentState.COMPLETED)

            return result

        except Exception as e:
            self.update_state(AgentState.ERROR)
            execution_time = (datetime.now() - start_time).total_seconds()

            error_result = AgentResult(
                agent_name=self.name,
                status=AgentState.ERROR,
                data=None,
                error=str(e),
                execution_time=execution_time,
            )

            self.logger.error(f"Agent execution failed: {e}")
            return error_result

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status information.

        Returns:
            Dictionary with status information
        """
        return {
            "name": self.name,
            "state": self.state.value,
            "has_context": self.context is not None,
        }


class MultiAgentCoordinator:
    """Utility class for coordinating multiple agents."""

    def __init__(self):
        """Initialize the coordinator."""
        self.logger = logging.getLogger("coordinator")
        self.agents: Dict[str, BaseAgent] = {}
        self.execution_order: List[str] = []

    def register_agent(self, agent: BaseAgent):
        """Register an agent with the coordinator.

        Args:
            agent: Agent to register
        """
        self.agents[agent.name] = agent
        self.logger.info(f"Registered agent: {agent.name}")

    def set_execution_order(self, agent_names: List[str]):
        """Set the order in which agents should be executed.

        Args:
            agent_names: List of agent names in execution order
        """
        # Validate that all agents are registered
        for name in agent_names:
            if name not in self.agents:
                raise ValueError(f"Agent '{name}' not registered")

        self.execution_order = agent_names
        self.logger.info(f"Set execution order: {' -> '.join(agent_names)}")

    async def execute_sequential(self, context: AgentContext) -> List[AgentResult]:
        """Execute agents sequentially in the defined order.

        Args:
            context: Shared context for all agents

        Returns:
            List of AgentResult objects
        """
        results = []

        for agent_name in self.execution_order:
            agent = self.agents[agent_name]
            self.logger.info(f"🤖 Executing agent: {agent_name}")

            result = await agent._execute_with_state_management(context)
            results.append(result)

            # Store result in shared context
            context.agent_results[agent_name] = result

            # Stop execution if agent failed and it's critical
            if result.status == AgentState.ERROR and self._is_critical_agent(
                agent_name
            ):
                self.logger.error(
                    f"Critical agent {agent_name} failed, stopping execution"
                )
                break

        return results

    async def execute_parallel(
        self, context: AgentContext, agent_names: List[str] = None
    ) -> List[AgentResult]:
        """Execute multiple agents in parallel.

        Args:
            context: Shared context for all agents
            agent_names: Optional list of specific agents to run in parallel

        Returns:
            List of AgentResult objects
        """
        if agent_names is None:
            agent_names = self.execution_order

        tasks = []
        for agent_name in agent_names:
            if agent_name in self.agents:
                agent = self.agents[agent_name]
                task = agent._execute_with_state_management(context)
                tasks.append((agent_name, task))

        self.logger.info(f"🚀 Executing {len(tasks)} agents in parallel")

        results = []
        completed_tasks = await asyncio.gather(
            *[task for _, task in tasks], return_exceptions=True
        )

        for (agent_name, _), result in zip(tasks, completed_tasks):
            if isinstance(result, Exception):
                error_result = AgentResult(
                    agent_name=agent_name,
                    status=AgentState.ERROR,
                    data=None,
                    error=str(result),
                )
                results.append(error_result)
            else:
                results.append(result)
                context.agent_results[agent_name] = result

        return results

    def _is_critical_agent(self, agent_name: str) -> bool:
        """Determine if an agent is critical for the workflow.

        Args:
            agent_name: Name of the agent to check

        Returns:
            True if agent is critical
        """
        # Define critical agents that should stop execution if they fail
        critical_agents = ["research_planner", "orchestrator"]
        return agent_name in critical_agents

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all registered agents.

        Returns:
            Dictionary mapping agent names to their status
        """
        return {name: agent.get_status() for name, agent in self.agents.items()}
