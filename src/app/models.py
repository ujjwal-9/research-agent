"""
Data models for the chat API.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from datetime import datetime
from enum import Enum


class MessageRole(str, Enum):
    """Role of a message in the chat."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessageRequest(BaseModel):
    """Request model for chat messages."""

    message: str
    session_id: Optional[str] = None


class ChatMessageResponse(BaseModel):
    """Response model for chat messages."""

    message: str
    role: MessageRole
    timestamp: datetime
    session_id: str


class ChatSessionCreate(BaseModel):
    """Request model for creating a new chat session."""

    initial_question: str


class ChatSessionResponse(BaseModel):
    """Response model for chat session."""

    session_id: str
    research_question: str
    messages: List[ChatMessageResponse]
    plan_approved: bool = False


class WebSocketMessage(BaseModel):
    """WebSocket message model."""

    type: str  # "message", "session_start", "session_end", "error"
    data: Dict[str, Any]
    session_id: Optional[str] = None
    timestamp: datetime = None

    def __init__(self, **data):
        if data.get("timestamp") is None:
            data["timestamp"] = datetime.now()
        super().__init__(**data)


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    message: str
    status_code: int
