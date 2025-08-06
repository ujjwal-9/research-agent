"""
Chat-based research planner that uses natural conversation to refine research plans.

This agent conducts a conversational interview with the user to understand their
research needs and collaboratively develop an optimal research plan.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import openai
import os

from src.agents.base_agent import BaseAgent, AgentContext, AgentResult, AgentState
from src.agents.research_planner import ResearchPlan
from src.tools.document_retriever import DocumentRetriever


@dataclass
class ChatMessage:
    """Represents a message in the chat conversation."""

    role: str  # "user", "assistant", "system"
    content: str
    timestamp: float


@dataclass
class ChatSession:
    """Represents the ongoing chat session."""

    messages: List[ChatMessage]
    research_question: str
    refined_question: Optional[str] = None
    user_requirements: Dict[str, Any] = None
    research_plan: Optional[ResearchPlan] = None
    plan_approved: bool = False


class ChatPlannerAgent(BaseAgent):
    """Agent that uses conversational AI to develop research plans."""

    def __init__(self):
        """Initialize the chat planner agent."""
        super().__init__("chat_planner")

        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Initialize document retriever to understand available sources
        self.document_retriever = DocumentRetriever()

        # Chat session state
        self.chat_session = None

    async def start_chat_session(self, initial_question: str) -> ChatSession:
        """Start a new chat session for research planning.

        Args:
            initial_question: The user's initial research question

        Returns:
            ChatSession object to track the conversation
        """
        import time

        self.chat_session = ChatSession(
            messages=[], research_question=initial_question, user_requirements={}
        )

        # Get available sources for context
        available_sources = self.document_retriever.get_all_sources()

        # Create system prompt
        system_prompt = self._create_system_prompt(available_sources)

        # Add system message
        self.chat_session.messages.append(
            ChatMessage(role="system", content=system_prompt, timestamp=time.time())
        )

        # Add initial user message
        self.chat_session.messages.append(
            ChatMessage(
                role="user",
                content=f"I want to research: {initial_question}",
                timestamp=time.time(),
            )
        )

        # Generate initial response
        await self._generate_chat_response()

        return self.chat_session

    async def continue_chat(self, user_message: str) -> str:
        """Continue the chat conversation.

        Args:
            user_message: The user's message

        Returns:
            The assistant's response
        """
        import time

        if not self.chat_session:
            raise ValueError("No active chat session. Call start_chat_session first.")

        # Add user message
        self.chat_session.messages.append(
            ChatMessage(role="user", content=user_message, timestamp=time.time())
        )

        # Generate response
        response = await self._generate_chat_response()

        # Check if the user wants to finalize the plan
        if self._should_generate_plan(user_message, response):
            await self._generate_research_plan()

        return response

    async def _generate_chat_response(self) -> str:
        """Generate a chat response using OpenAI."""
        import time

        try:
            # Convert chat messages to OpenAI format
            openai_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in self.chat_session.messages
            ]

            response = self.openai_client.chat.completions.create(
                model="gpt-4", messages=openai_messages, temperature=0.7, max_tokens=800
            )

            assistant_message = response.choices[0].message.content

            # Add assistant message to chat history
            self.chat_session.messages.append(
                ChatMessage(
                    role="assistant", content=assistant_message, timestamp=time.time()
                )
            )

            return assistant_message

        except Exception as e:
            self.logger.error(f"Error generating chat response: {e}")
            fallback_response = "I apologize, but I'm having trouble generating a response. Could you please rephrase your question or requirement?"

            self.chat_session.messages.append(
                ChatMessage(
                    role="assistant", content=fallback_response, timestamp=time.time()
                )
            )

            return fallback_response

    def _create_system_prompt(self, available_sources: List[str]) -> str:
        """Create the system prompt for the chat agent."""

        source_context = ""
        if available_sources:
            source_count = len(available_sources)
            sample_sources = available_sources[:5]  # Show first 5 as examples
            source_context = f"""

**Available Internal Sources ({source_count} total):**
I have access to {source_count} internal documents that may contain relevant information. Here are some examples:
{chr(10).join(f"• {source}" for source in sample_sources)}
{"• ..." if len(available_sources) > 5 else ""}

I can search through these documents to find relevant information for your research.
"""
        else:
            source_context = "\n**Note:** I don't currently have access to internal documents, so I'll focus on external web research."

        return f"""You are an expert research planning assistant. Your role is to have a natural, conversational dialogue with the user to understand their research needs and collaboratively develop the best possible research plan.

**Your Capabilities:**
- I can conduct comprehensive research using both internal documents and external web sources
- I can analyze data, extract insights, and synthesize information from multiple sources  
- I can generate detailed reports with evidence and citations
- I have Excel analysis capabilities for data-driven research{source_context}

**Your Conversation Style:**
- Be conversational, helpful, and engaging
- Ask clarifying questions to understand their specific needs
- Suggest research approaches and methodologies
- Help them refine their research question if needed
- Be curious about what they're trying to accomplish

**Your Goal:**
Through natural conversation, help the user:
1. Clarify and refine their research question
2. Understand what type of analysis they need (overview, detailed, comparative, etc.)
3. Identify specific aspects they want to focus on
4. Determine the depth and scope of research needed
5. Decide on the format and style of the final report

**Important Guidelines:**
- Don't immediately jump to creating a research plan
- Have a genuine conversation to understand their needs
- Ask follow-up questions based on their responses
- Be flexible and adaptive to their requirements
- When they seem ready, offer to create a detailed research plan
- Always confirm the plan before proceeding

Start by greeting them and exploring their research question in a conversational way."""

    def _should_generate_plan(self, user_message: str, assistant_response: str) -> bool:
        """Determine if it's time to generate a research plan."""

        # Only generate plan if we have enough conversation
        if len(self.chat_session.messages) < 6:
            return False

        user_lower = user_message.lower()

        # Look for explicit plan generation requests
        explicit_plan_requests = [
            "generate the plan",
            "create the plan",
            "make the plan",
            "let's generate",
            "let's create",
            "ready to generate",
            "ready to create",
        ]

        # Look for completion/approval phrases combined with affirmative context
        completion_phrases = [
            "sounds good",
            "looks good",
            "let's do it",
            "go ahead",
            "that works",
            "perfect",
        ]

        # Look for simple approval words when assistant has discussed a plan
        simple_approvals = ["start", "yes", "ok", "okay", "proceed"]

        # Look for finalization words
        finalization_words = ["finalize", "approve", "confirm", "agree", "accept"]

        # Check if user explicitly requested plan generation
        explicit_request = any(
            phrase in user_lower for phrase in explicit_plan_requests
        )

        # Check if user indicated approval/completion AND assistant offered to create plan
        assistant_lower = assistant_response.lower()
        assistant_offered_plan = any(
            phrase in assistant_lower
            for phrase in [
                "create a plan",
                "generate a plan",
                "develop a research plan",
                "ready to create",
                "shall i create",
                "would you like me to",
            ]
        )

        user_approved = any(phrase in user_lower for phrase in completion_phrases)
        user_finalized = any(word in user_lower for word in finalization_words)
        user_simple_approval = any(word in user_lower for word in simple_approvals)

        # Check if assistant has discussed plan structure or asked for approval
        assistant_discussed_plan = any(
            phrase in assistant_lower
            for phrase in [
                "research plan",
                "research steps",
                "does this",
                "meet your expectations",
                "research topic",
                "final report",
                "does this plan",
            ]
        )

        # Only trigger plan generation if:
        # 1. User explicitly requested it, OR
        # 2. Assistant offered and user approved/finalized, OR
        # 3. Assistant discussed plan structure and user gave simple approval
        return (
            explicit_request
            or (assistant_offered_plan and (user_approved or user_finalized))
            or (assistant_discussed_plan and user_simple_approval)
        )

    async def _generate_research_plan(self):
        """Generate a detailed research plan based on the conversation."""

        # Extract insights from the conversation
        self._summarize_conversation()

        try:
            # Use the research planner's logic to generate the plan
            from src.agents.research_planner import ResearchPlannerAgent

            planner = ResearchPlannerAgent()

            # Create context for the planner
            context = AgentContext(
                research_question=self.chat_session.research_question,
                user_requirements=self.chat_session.user_requirements,
            )

            # Generate the plan
            plan_result = await planner.execute(context)

            if plan_result.status == AgentState.COMPLETED:
                self.chat_session.research_plan = plan_result.data
                self.logger.info("Research plan generated successfully")
            else:
                self.logger.error(
                    f"Failed to generate research plan: {plan_result.error}"
                )

        except Exception as e:
            self.logger.error(f"Error generating research plan: {e}")

    def _summarize_conversation(self) -> str:
        """Summarize the key points from the conversation."""

        user_messages = [
            msg.content
            for msg in self.chat_session.messages
            if msg.role == "user" and not msg.content.startswith("I want to research:")
        ]

        if not user_messages:
            return "User wants to research the original question without additional requirements."

        # Join user messages to understand their requirements
        user_input = " ".join(user_messages)

        # Extract key requirements
        requirements = {}

        # Simple extraction of common requirements
        if any(
            word in user_input.lower()
            for word in ["detailed", "comprehensive", "thorough", "deep"]
        ):
            requirements["depth_preference"] = "detailed"
        elif any(
            word in user_input.lower()
            for word in ["overview", "summary", "brief", "high-level"]
        ):
            requirements["depth_preference"] = "overview"

        if any(
            word in user_input.lower()
            for word in ["compare", "comparison", "versus", "vs"]
        ):
            requirements["analysis_type"] = "comparative"

        if any(
            word in user_input.lower()
            for word in ["trend", "trends", "over time", "historical"]
        ):
            requirements["include_trends"] = True

        if any(
            word in user_input.lower()
            for word in ["data", "numbers", "statistics", "metrics"]
        ):
            requirements["data_focused"] = True

        self.chat_session.user_requirements = requirements

        return f"User requirements: {user_input}"

    def get_research_plan(self) -> Optional[ResearchPlan]:
        """Get the generated research plan."""
        return self.chat_session.research_plan if self.chat_session else None

    def approve_plan(self) -> bool:
        """Mark the research plan as approved."""
        if self.chat_session and self.chat_session.research_plan:
            self.chat_session.plan_approved = True
            return True
        return False

    def get_chat_history(self) -> List[ChatMessage]:
        """Get the chat history."""
        return self.chat_session.messages if self.chat_session else []

    async def execute(self, context: AgentContext) -> AgentResult:
        """Execute the chat planner agent (required by BaseAgent).

        Note: This method is required by the BaseAgent interface but the ChatPlannerAgent
        is designed for interactive use via start_chat_session and continue_chat methods.

        Args:
            context: Shared context containing research question

        Returns:
            AgentResult with the chat session or research plan
        """
        try:
            # If we already have a research plan, return it
            if self.chat_session and self.chat_session.research_plan:
                return AgentResult(
                    agent_name=self.name,
                    status=AgentState.COMPLETED,
                    data=self.chat_session.research_plan,
                    metadata={
                        "chat_messages": len(self.chat_session.messages),
                        "plan_approved": self.chat_session.plan_approved,
                    },
                )

            # Otherwise, start a basic chat session with the context research question
            if context.research_question:
                await self.start_chat_session(context.research_question)

                return AgentResult(
                    agent_name=self.name,
                    status=AgentState.COMPLETED,
                    data=self.chat_session,
                    metadata={
                        "chat_session_started": True,
                        "initial_question": context.research_question,
                    },
                )
            else:
                return AgentResult(
                    agent_name=self.name,
                    status=AgentState.ERROR,
                    error="No research question provided in context",
                )

        except Exception as e:
            self.logger.error(f"Chat planner execution failed: {e}")
            return AgentResult(
                agent_name=self.name, status=AgentState.ERROR, error=str(e)
            )
