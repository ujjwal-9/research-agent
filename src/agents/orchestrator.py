"""
Research orchestrator agent for coordinating the multi-agent research workflow.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

from .base_agent import (
    BaseAgent,
    AgentContext,
    AgentResult,
    AgentState,
    MultiAgentCoordinator,
)
from .research_planner import ResearchPlannerAgent
from .document_analyst import DocumentAnalystAgent
from .web_researcher import WebResearcherAgent
from .synthesis_agent import SynthesisAgent


@dataclass
class UserInteraction:
    """Represents a user interaction during the workflow."""

    interaction_type: str  # "clarification", "plan_approval", "progress_update"
    message: str
    options: List[str] = None
    required: bool = True


@dataclass
class WorkflowState:
    """Represents the current state of the research workflow."""

    stage: str
    progress: float  # 0.0 to 1.0
    current_agent: Optional[str]
    completed_agents: List[str]
    total_execution_time: float
    error_count: int
    last_update: datetime


@dataclass
class ResearchWorkflowResult:
    """Complete result from the research workflow."""

    research_report: Any  # Research report from synthesis
    workflow_metadata: Dict[str, Any]
    execution_timeline: List[Dict[str, Any]]
    user_interactions: List[UserInteraction]
    final_state: WorkflowState
    comprehensive_answer: str = ""  # Comprehensive answer to the research question


class ResearchOrchestrator(BaseAgent):
    """Main orchestrator for the multi-agent research workflow."""

    def __init__(self, collection_name: str = None):
        """Initialize the research orchestrator.

        Args:
            collection_name: Qdrant collection name for document retrieval
        """
        super().__init__("orchestrator")

        # Initialize agents
        self.research_planner = ResearchPlannerAgent()
        self.document_analyst = DocumentAnalystAgent(collection_name)
        self.web_researcher = WebResearcherAgent()
        self.synthesis_agent = SynthesisAgent()

        # Initialize coordinator
        self.coordinator = MultiAgentCoordinator()
        self._setup_coordinator()

        # Workflow state
        self.workflow_state = None
        self.user_interactions = []
        self.execution_timeline = []

    def _setup_coordinator(self):
        """Setup the multi-agent coordinator with all agents."""
        agents = [
            self.research_planner,
            self.document_analyst,
            self.web_researcher,
            self.synthesis_agent,
        ]

        for agent in agents:
            self.coordinator.register_agent(agent)

        # Set execution order
        self.coordinator.set_execution_order(
            [
                "research_planner",
                "document_analyst",
                "web_researcher",
                "synthesis_agent",
            ]
        )

    async def execute(self, context: AgentContext) -> AgentResult:
        """Execute the complete research workflow.

        Args:
            context: Initial context with research question

        Returns:
            AgentResult containing complete workflow results
        """
        start_time = datetime.now()

        try:
            self.logger.info(
                f"🚀 Starting research workflow for: {context.research_question}"
            )

            # Initialize workflow state
            self.workflow_state = WorkflowState(
                stage="initialization",
                progress=0.0,
                current_agent=None,
                completed_agents=[],
                total_execution_time=0.0,
                error_count=0,
                last_update=start_time,
            )

            # Stage 1: Research Planning and Clarification
            self.logger.info("📋 Stage 1: Research Planning")
            await self._update_workflow_state("planning", 0.1, "research_planner")

            planner_result = await self._execute_planning_stage(context)
            if planner_result.status == AgentState.ERROR:
                raise Exception(f"Planning failed: {planner_result.error}")

            # Store the planner's result in the shared context for other agents
            context.agent_results["research_planner"] = planner_result

            # Check for clarifying questions
            research_plan = planner_result.data
            if research_plan.clarifying_questions:
                await self._handle_clarifying_questions(
                    research_plan.clarifying_questions, context
                )

            # Stage 2: Plan Approval
            self.logger.info("✅ Stage 2: Plan Approval")
            await self._update_workflow_state("plan_approval", 0.2)

            plan_approved = await self._request_plan_approval(research_plan, context)
            if not plan_approved:
                raise Exception("Research plan was not approved by user")

            # Stage 3: Parallel Research Execution
            self.logger.info("🔄 Stage 3: Research Execution")
            await self._update_workflow_state("research_execution", 0.3)

            research_results = await self._execute_research_stage(context)

            # Stage 4: Synthesis and Report Generation
            self.logger.info("📊 Stage 4: Synthesis and Report Generation")
            await self._update_workflow_state("synthesis", 0.8)

            synthesis_result = await self._execute_synthesis_stage(context)
            if synthesis_result.status == AgentState.ERROR:
                raise Exception(f"Synthesis failed: {synthesis_result.error}")

            # Stage 5: Final Report Delivery
            self.logger.info("🎯 Stage 5: Report Delivery")
            await self._update_workflow_state("completed", 1.0)

            total_time = (datetime.now() - start_time).total_seconds()

            # Create final workflow result
            workflow_result = ResearchWorkflowResult(
                research_report=synthesis_result.data.research_report,
                comprehensive_answer=synthesis_result.data.comprehensive_answer,
                workflow_metadata={
                    "total_execution_time": total_time,
                    "agents_executed": len(self.workflow_state.completed_agents),
                    "user_interactions": len(self.user_interactions),
                    "error_count": self.workflow_state.error_count,
                    "research_question": context.research_question,
                },
                execution_timeline=self.execution_timeline,
                user_interactions=self.user_interactions,
                final_state=self.workflow_state,
            )

            self.logger.info(
                f"✅ Research workflow completed successfully in {total_time:.1f}s"
            )

            return AgentResult(
                agent_name=self.name,
                status=AgentState.COMPLETED,
                data=workflow_result,
                execution_time=total_time,
                metadata={
                    "workflow_stages_completed": 5,
                    "total_agents_executed": len(self.workflow_state.completed_agents),
                    "final_report_sections": len(
                        synthesis_result.data.research_report.sections
                    ),
                },
            )

        except Exception as e:
            self.logger.error(f"❌ Research workflow failed: {e}")

            if self.workflow_state:
                self.workflow_state.stage = "error"
                self.workflow_state.error_count += 1

            return AgentResult(
                agent_name=self.name,
                status=AgentState.ERROR,
                data=None,
                error=str(e),
                execution_time=(datetime.now() - start_time).total_seconds(),
            )

    async def _execute_planning_stage(self, context: AgentContext) -> AgentResult:
        """Execute the research planning stage.

        Args:
            context: Shared context

        Returns:
            Result from research planner
        """
        try:
            result = await self.research_planner._execute_with_state_management(context)

            # Record in timeline
            self.execution_timeline.append(
                {
                    "stage": "planning",
                    "agent": "research_planner",
                    "timestamp": datetime.now().isoformat(),
                    "status": result.status.value,
                    "execution_time": result.execution_time,
                }
            )

            if result.status == AgentState.COMPLETED:
                self.workflow_state.completed_agents.append("research_planner")

            return result

        except Exception as e:
            self.logger.error(f"Planning stage failed: {e}")
            raise

    async def _handle_clarifying_questions(
        self, questions: List[str], context: AgentContext
    ):
        """Handle clarifying questions from the research planner.

        Args:
            questions: List of clarifying questions
            context: Shared context to update
        """
        if not questions:
            return

        self.logger.info(f"❓ {len(questions)} clarifying questions identified")

        # Create user interaction for clarifying questions
        interaction = UserInteraction(
            interaction_type="clarification",
            message=f"The research planner has identified {len(questions)} clarifying questions to help refine the research scope:",
            options=questions,
            required=False,
        )

        self.user_interactions.append(interaction)

        # For now, we'll proceed without user input
        # In a full implementation, you would pause here for user input
        self.logger.info("📝 Proceeding with research based on current understanding")

    async def _request_plan_approval(
        self, research_plan, context: AgentContext
    ) -> bool:
        """Request user approval for the research plan.

        Args:
            research_plan: Generated research plan
            context: Shared context

        Returns:
            True if plan is approved
        """
        # Create plan summary for user
        plan_summary = f"""
**Research Plan Summary:**

**Question:** {research_plan.research_question}

**Methodology:** {research_plan.research_methodology}

**Internal Search Strategy:**
- {len(research_plan.internal_search_queries)} targeted queries across internal documents
- Expected to search {len(research_plan.expected_sources)} available sources

**External Research Strategy:**
- {len(research_plan.external_search_topics)} external topics to research
- Focus on current trends and authoritative sources

**Estimated Timeline:** {research_plan.estimated_timeline.get('total_estimated', 'Unknown')}

**Success Criteria:**
{chr(10).join(f"- {criterion}" for criterion in research_plan.success_criteria)}
        """.strip()

        interaction = UserInteraction(
            interaction_type="plan_approval",
            message=plan_summary,
            options=["Approve and proceed", "Request modifications", "Cancel research"],
            required=True,
        )

        self.user_interactions.append(interaction)

        # For this implementation, auto-approve
        # In a full implementation, you would wait for user input
        self.logger.info("📋 Research plan generated and auto-approved for execution")
        return True

    async def _execute_research_stage(self, context: AgentContext) -> List[AgentResult]:
        """Execute the parallel research stage.

        Args:
            context: Shared context

        Returns:
            List of agent results
        """
        try:
            # Execute document analysis and web research in parallel
            research_agents = ["document_analyst", "web_researcher"]

            self.logger.info(
                "🔄 Executing parallel research (document analysis + web research)"
            )

            # Run both agents in parallel
            results = await self.coordinator.execute_parallel(context, research_agents)

            # Record results in timeline
            for result in results:
                self.execution_timeline.append(
                    {
                        "stage": "research",
                        "agent": result.agent_name,
                        "timestamp": datetime.now().isoformat(),
                        "status": result.status.value,
                        "execution_time": result.execution_time,
                    }
                )

                if result.status == AgentState.COMPLETED:
                    self.workflow_state.completed_agents.append(result.agent_name)
                elif result.status == AgentState.ERROR:
                    self.workflow_state.error_count += 1
                    self.logger.error(
                        f"Agent {result.agent_name} failed: {result.error}"
                    )

            return results

        except Exception as e:
            self.logger.error(f"Research stage failed: {e}")
            raise

    async def _execute_synthesis_stage(self, context: AgentContext) -> AgentResult:
        """Execute the synthesis and report generation stage.

        Args:
            context: Shared context with all research results

        Returns:
            Result from synthesis agent
        """
        try:
            result = await self.synthesis_agent._execute_with_state_management(context)

            # Record in timeline
            self.execution_timeline.append(
                {
                    "stage": "synthesis",
                    "agent": "synthesis_agent",
                    "timestamp": datetime.now().isoformat(),
                    "status": result.status.value,
                    "execution_time": result.execution_time,
                }
            )

            if result.status == AgentState.COMPLETED:
                self.workflow_state.completed_agents.append("synthesis_agent")
            else:
                self.workflow_state.error_count += 1

            return result

        except Exception as e:
            self.logger.error(f"Synthesis stage failed: {e}")
            raise

    async def _update_workflow_state(
        self, stage: str, progress: float, current_agent: str = None
    ):
        """Update the current workflow state.

        Args:
            stage: Current workflow stage
            progress: Progress percentage (0.0 to 1.0)
            current_agent: Currently executing agent
        """
        if self.workflow_state:
            self.workflow_state.stage = stage
            self.workflow_state.progress = progress
            self.workflow_state.current_agent = current_agent
            self.workflow_state.last_update = datetime.now()

        self.logger.info(f"📊 Workflow progress: {stage} ({progress*100:.0f}%)")

    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status.

        Returns:
            Dictionary with current workflow status
        """
        if not self.workflow_state:
            return {"status": "not_started"}

        return {
            "stage": self.workflow_state.stage,
            "progress": self.workflow_state.progress,
            "current_agent": self.workflow_state.current_agent,
            "completed_agents": self.workflow_state.completed_agents,
            "total_execution_time": self.workflow_state.total_execution_time,
            "error_count": self.workflow_state.error_count,
            "last_update": self.workflow_state.last_update.isoformat(),
            "user_interactions": len(self.user_interactions),
        }

    async def execute_research_workflow(
        self, research_question: str, user_requirements: Dict[str, Any] = None
    ) -> ResearchWorkflowResult:
        """High-level method to execute a complete research workflow.

        Args:
            research_question: Research question to investigate
            user_requirements: Optional user requirements and preferences

        Returns:
            Complete workflow result
        """
        # Create context
        context = AgentContext(
            research_question=research_question,
            user_requirements=user_requirements or {},
        )

        # Execute workflow
        result = await self.execute(context)

        if result.status == AgentState.COMPLETED:
            return result.data
        else:
            raise Exception(f"Workflow failed: {result.error}")


# Convenience function for direct workflow execution
async def execute_research_workflow(
    research_question: str,
    collection_name: str = None,
    user_requirements: Dict[str, Any] = None,
) -> ResearchWorkflowResult:
    """Execute a complete research workflow.

    Args:
        research_question: Research question to investigate
        collection_name: Qdrant collection name for document retrieval
        user_requirements: Optional user requirements

    Returns:
        Complete research workflow result
    """
    orchestrator = ResearchOrchestrator(collection_name)
    return await orchestrator.execute_research_workflow(
        research_question, user_requirements
    )
