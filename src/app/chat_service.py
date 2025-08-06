"""
Chat service that integrates with the existing ChatPlannerAgent.
"""

import asyncio
import uuid
import logging
from typing import Dict, Optional
from datetime import datetime

from src.agents.chat_planner import ChatPlannerAgent, ChatSession, ChatMessage
from .models import ChatMessageResponse, ChatSessionResponse, MessageRole


logger = logging.getLogger(__name__)


class ChatService:
    """Service for managing chat sessions with the research planner agent."""

    def __init__(self):
        """Initialize the chat service."""
        self.chat_agent = ChatPlannerAgent()
        self.active_sessions: Dict[str, ChatSession] = {}

    async def create_session(self, initial_question: str) -> ChatSessionResponse:
        """Create a new chat session.

        Args:
            initial_question: The user's initial research question

        Returns:
            ChatSessionResponse with session details
        """
        session_id = str(uuid.uuid4())

        try:
            # Start a new chat session with the agent
            chat_session = await self.chat_agent.start_chat_session(initial_question)

            # Store the session
            self.active_sessions[session_id] = chat_session

            # Convert messages to response format
            messages = []
            for msg in chat_session.messages:
                if msg.role != "system":  # Don't include system messages in response
                    messages.append(
                        ChatMessageResponse(
                            message=msg.content,
                            role=MessageRole(msg.role),
                            timestamp=datetime.fromtimestamp(msg.timestamp),
                            session_id=session_id,
                        )
                    )

            return ChatSessionResponse(
                session_id=session_id,
                research_question=chat_session.research_question,
                messages=messages,
                plan_approved=chat_session.plan_approved,
            )

        except Exception as e:
            logger.error(f"Error creating chat session: {e}")
            raise

    async def send_message(self, session_id: str, message: str) -> ChatMessageResponse:
        """Send a message to an existing chat session.

        Args:
            session_id: ID of the chat session
            message: User's message

        Returns:
            ChatMessageResponse with the agent's response
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")

        try:
            # Set the chat session in the agent
            self.chat_agent.chat_session = self.active_sessions[session_id]

            # Send the message and get response
            response = await self.chat_agent.continue_chat(message)

            # Update the stored session
            self.active_sessions[session_id] = self.chat_agent.chat_session

            # Return the latest assistant message
            latest_message = self.chat_agent.chat_session.messages[-1]

            return ChatMessageResponse(
                message=latest_message.content,
                role=MessageRole(latest_message.role),
                timestamp=datetime.fromtimestamp(latest_message.timestamp),
                session_id=session_id,
            )

        except Exception as e:
            logger.error(f"Error sending message to session {session_id}: {e}")
            raise

    async def get_session(self, session_id: str) -> Optional[ChatSessionResponse]:
        """Get details of an existing chat session.

        Args:
            session_id: ID of the chat session

        Returns:
            ChatSessionResponse if session exists, None otherwise
        """
        if session_id not in self.active_sessions:
            return None

        chat_session = self.active_sessions[session_id]

        # Convert messages to response format
        messages = []
        for msg in chat_session.messages:
            if msg.role != "system":  # Don't include system messages in response
                messages.append(
                    ChatMessageResponse(
                        message=msg.content,
                        role=MessageRole(msg.role),
                        timestamp=datetime.fromtimestamp(msg.timestamp),
                        session_id=session_id,
                    )
                )

        return ChatSessionResponse(
            session_id=session_id,
            research_question=chat_session.research_question,
            messages=messages,
            plan_approved=chat_session.plan_approved,
        )

    def end_session(self, session_id: str) -> bool:
        """End a chat session.

        Args:
            session_id: ID of the chat session to end

        Returns:
            True if session was ended, False if session not found
        """
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Ended chat session: {session_id}")
            return True
        return False

    def get_active_sessions_count(self) -> int:
        """Get the number of active sessions."""
        return len(self.active_sessions)
