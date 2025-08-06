"""
WebSocket connection manager for real-time chat.
"""

import json
import logging
from typing import Dict, List
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime

from .models import WebSocketMessage, MessageRole
from .chat_service import ChatService


logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for chat sessions."""

    def __init__(self):
        """Initialize the connection manager."""
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_connections: Dict[str, str] = {}  # session_id -> connection_id
        self.chat_service = ChatService()

    async def connect(self, websocket: WebSocket, connection_id: str):
        """Accept a new WebSocket connection.

        Args:
            websocket: The WebSocket connection
            connection_id: Unique identifier for this connection
        """
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        logger.info(f"WebSocket connection established: {connection_id}")

    def disconnect(self, connection_id: str):
        """Remove a WebSocket connection.

        Args:
            connection_id: Unique identifier for the connection
        """
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]

        # Clean up session mapping
        session_to_remove = None
        for session_id, conn_id in self.session_connections.items():
            if conn_id == connection_id:
                session_to_remove = session_id
                break

        if session_to_remove:
            del self.session_connections[session_to_remove]
            # Optionally end the chat session
            self.chat_service.end_session(session_to_remove)

        logger.info(f"WebSocket connection closed: {connection_id}")

    async def send_message(self, connection_id: str, message: WebSocketMessage):
        """Send a message to a specific WebSocket connection.

        Args:
            connection_id: Unique identifier for the connection
            message: WebSocket message to send
        """
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            try:
                await websocket.send_text(message.model_dump_json())
            except Exception as e:
                logger.error(f"Error sending message to {connection_id}: {e}")
                self.disconnect(connection_id)

    async def broadcast_to_session(self, session_id: str, message: WebSocketMessage):
        """Broadcast a message to all connections in a session.

        Args:
            session_id: Session ID to broadcast to
            message: WebSocket message to send
        """
        if session_id in self.session_connections:
            connection_id = self.session_connections[session_id]
            await self.send_message(connection_id, message)

    async def handle_message(self, connection_id: str, data: dict):
        """Handle incoming WebSocket messages.

        Args:
            connection_id: Unique identifier for the connection
            data: Parsed JSON data from the WebSocket
        """
        try:
            message_type = data.get("type")

            if message_type == "session_start":
                await self._handle_session_start(connection_id, data)
            elif message_type == "message":
                await self._handle_chat_message(connection_id, data)
            elif message_type == "session_end":
                await self._handle_session_end(connection_id, data)
            else:
                await self._send_error(
                    connection_id, f"Unknown message type: {message_type}"
                )

        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            await self._send_error(connection_id, "Internal server error")

    async def _handle_session_start(self, connection_id: str, data: dict):
        """Handle session start requests."""
        initial_question = data.get("data", {}).get("initial_question")
        if not initial_question:
            await self._send_error(connection_id, "Missing initial_question")
            return

        try:
            session_response = await self.chat_service.create_session(initial_question)
            self.session_connections[session_response.session_id] = connection_id

            # Send session created confirmation
            response = WebSocketMessage(
                type="session_created",
                data={
                    "session_id": session_response.session_id,
                    "research_question": session_response.research_question,
                    "messages": [msg.model_dump() for msg in session_response.messages],
                },
                session_id=session_response.session_id,
            )
            await self.send_message(connection_id, response)

        except Exception as e:
            logger.error(f"Error creating session: {e}")
            await self._send_error(connection_id, "Failed to create session")

    async def _handle_chat_message(self, connection_id: str, data: dict):
        """Handle chat messages."""
        session_id = data.get("session_id")
        message = data.get("data", {}).get("message")

        if not session_id or not message:
            await self._send_error(connection_id, "Missing session_id or message")
            return

        try:
            response = await self.chat_service.send_message(session_id, message)

            # Send the response back
            ws_response = WebSocketMessage(
                type="message_response",
                data={
                    "message": response.message,
                    "role": response.role.value,
                    "timestamp": response.timestamp.isoformat(),
                },
                session_id=session_id,
            )
            await self.send_message(connection_id, ws_response)

        except ValueError as e:
            await self._send_error(connection_id, str(e))
        except Exception as e:
            logger.error(f"Error processing chat message: {e}")
            await self._send_error(connection_id, "Failed to process message")

    async def _handle_session_end(self, connection_id: str, data: dict):
        """Handle session end requests."""
        session_id = data.get("session_id")
        if not session_id:
            await self._send_error(connection_id, "Missing session_id")
            return

        success = self.chat_service.end_session(session_id)
        if session_id in self.session_connections:
            del self.session_connections[session_id]

        response = WebSocketMessage(
            type="session_ended", data={"success": success}, session_id=session_id
        )
        await self.send_message(connection_id, response)

    async def _send_error(self, connection_id: str, error_message: str):
        """Send an error message to a WebSocket connection."""
        error_response = WebSocketMessage(type="error", data={"error": error_message})
        await self.send_message(connection_id, error_response)
