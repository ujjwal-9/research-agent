"""
Pydantic models for the chat API.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from enum import Enum


class MessageRole(str, Enum):
    """Message role enumeration."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """Chat message model."""

    role: MessageRole
    content: str
    timestamp: Optional[float] = None


class ChatSessionRequest(BaseModel):
    """Request to start a new chat session."""

    initial_question: str


class ChatMessageRequest(BaseModel):
    """Request to send a message in an existing chat session."""

    message: str
    session_id: str


class ChatResponse(BaseModel):
    """Chat response model."""

    message: str
    session_id: str
    plan_available: bool = False
    plan_approved: bool = False


class ResearchPlanResponse(BaseModel):
    """Research plan response model."""

    research_question: str
    research_methodology: str
    internal_search_queries: List[str]
    external_search_topics: List[str]
    estimated_timeline: str
    success_criteria: List[str]


class StartResearchRequest(BaseModel):
    """Request to start research workflow."""

    session_id: str
    use_code_interpreter: bool = True


class WebSocketMessage(BaseModel):
    """WebSocket message model."""

    type: str  # "start_session", "send_message", "get_plan", "start_research"
    data: Dict[str, Any]


class WebSocketResponse(BaseModel):
    """WebSocket response model."""

    type: str  # "chat_response", "research_plan", "research_progress", "error"
    data: Dict[str, Any]
    success: bool = True
    error: Optional[str] = None
