"""
WebSocket service for real-time chat communication.
"""

import json
import asyncio
import logging
from typing import Dict, Set
from fastapi import WebSocket, WebSocketDisconnect

from .chat_service import ChatService
from .models import WebSocketMessage, WebSocketResponse


class WebSocketManager:
    """Manager for WebSocket connections and chat sessions."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_connections: Dict[str, str] = {}  # session_id -> connection_id
        self.chat_service = ChatService()

        # Setup logging
        self.logger = logging.getLogger(__name__)

    async def connect(self, websocket: WebSocket, connection_id: str):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        self.logger.info(f"WebSocket connection established: {connection_id}")

    def disconnect(self, connection_id: str):
        """Remove a WebSocket connection."""
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
            self.chat_service.cleanup_session(session_to_remove)

        self.logger.info(f"WebSocket connection closed: {connection_id}")

    async def send_message(self, connection_id: str, message: WebSocketResponse):
        """Send a message to a specific connection."""
        if connection_id in self.active_connections:
            try:
                websocket = self.active_connections[connection_id]
                await websocket.send_text(message.model_dump_json())
            except Exception as e:
                self.logger.error(f"Error sending message to {connection_id}: {e}")
                self.disconnect(connection_id)

    async def handle_message(self, connection_id: str, raw_message: str):
        """Handle incoming WebSocket message."""
        try:
            # Parse message
            message_data = json.loads(raw_message)
            message = WebSocketMessage(**message_data)

            response = await self._process_message(connection_id, message)
            await self.send_message(connection_id, response)

        except json.JSONDecodeError:
            response = WebSocketResponse(
                type="error", data={}, success=False, error="Invalid JSON format"
            )
            await self.send_message(connection_id, response)

        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
            response = WebSocketResponse(
                type="error", data={}, success=False, error=str(e)
            )
            await self.send_message(connection_id, response)

    async def _process_message(
        self, connection_id: str, message: WebSocketMessage
    ) -> WebSocketResponse:
        """Process a WebSocket message and return response."""

        if message.type == "start_session":
            return await self._handle_start_session(connection_id, message.data)

        elif message.type == "send_message":
            return await self._handle_send_message(connection_id, message.data)

        elif message.type == "get_plan":
            return await self._handle_get_plan(connection_id, message.data)

        elif message.type == "start_research":
            return await self._handle_start_research(connection_id, message.data)

        elif message.type == "get_session_status":
            return await self._handle_get_session_status(connection_id, message.data)

        else:
            return WebSocketResponse(
                type="error",
                data={},
                success=False,
                error=f"Unknown message type: {message.type}",
            )

    async def _handle_start_session(
        self, connection_id: str, data: Dict
    ) -> WebSocketResponse:
        """Handle start session request."""
        try:
            initial_question = data.get("initial_question")
            if not initial_question:
                return WebSocketResponse(
                    type="error",
                    data={},
                    success=False,
                    error="initial_question is required",
                )

            session_id, chat_response = await self.chat_service.start_chat_session(
                initial_question
            )

            # Map session to connection
            self.session_connections[session_id] = connection_id

            return WebSocketResponse(
                type="session_started",
                data={
                    "session_id": session_id,
                    "message": chat_response.message,
                    "plan_available": chat_response.plan_available,
                    "plan_approved": chat_response.plan_approved,
                },
            )

        except Exception as e:
            return WebSocketResponse(type="error", data={}, success=False, error=str(e))

    async def _handle_send_message(
        self, connection_id: str, data: Dict
    ) -> WebSocketResponse:
        """Handle send message request."""
        try:
            session_id = data.get("session_id")
            user_message = data.get("message")

            if not session_id or not user_message:
                return WebSocketResponse(
                    type="error",
                    data={},
                    success=False,
                    error="session_id and message are required",
                )

            chat_response = await self.chat_service.send_message(
                session_id, user_message
            )

            return WebSocketResponse(
                type="chat_response",
                data={
                    "message": chat_response.message,
                    "session_id": chat_response.session_id,
                    "plan_available": chat_response.plan_available,
                    "plan_approved": chat_response.plan_approved,
                },
            )

        except Exception as e:
            return WebSocketResponse(type="error", data={}, success=False, error=str(e))

    async def _handle_get_plan(
        self, connection_id: str, data: Dict
    ) -> WebSocketResponse:
        """Handle get research plan request."""
        try:
            session_id = data.get("session_id")
            if not session_id:
                return WebSocketResponse(
                    type="error", data={}, success=False, error="session_id is required"
                )

            plan = self.chat_service.get_research_plan(session_id)

            if not plan:
                return WebSocketResponse(type="research_plan", data={"plan": None})

            return WebSocketResponse(
                type="research_plan",
                data={
                    "plan": {
                        "research_question": plan.research_question,
                        "research_methodology": plan.research_methodology,
                        "internal_search_queries": plan.internal_search_queries,
                        "external_search_topics": plan.external_search_topics,
                        "estimated_timeline": plan.estimated_timeline,
                        "success_criteria": plan.success_criteria,
                    }
                },
            )

        except Exception as e:
            return WebSocketResponse(type="error", data={}, success=False, error=str(e))

    async def _handle_start_research(
        self, connection_id: str, data: Dict
    ) -> WebSocketResponse:
        """Handle start research request."""
        try:
            session_id = data.get("session_id")
            use_code_interpreter = data.get("use_code_interpreter", True)

            if not session_id:
                return WebSocketResponse(
                    type="error", data={}, success=False, error="session_id is required"
                )

            # Send initial progress update
            await self.send_message(
                connection_id,
                WebSocketResponse(
                    type="research_progress",
                    data={
                        "status": "starting",
                        "message": "Starting research workflow...",
                    },
                ),
            )

            # Start research in background and send progress updates
            asyncio.create_task(
                self._run_research_with_progress(
                    connection_id, session_id, use_code_interpreter
                )
            )

            return WebSocketResponse(
                type="research_started", data={"session_id": session_id}
            )

        except Exception as e:
            return WebSocketResponse(type="error", data={}, success=False, error=str(e))

    async def _run_research_with_progress(
        self, connection_id: str, session_id: str, use_code_interpreter: bool
    ):
        """Run research and send progress updates."""
        try:
            # Send progress updates
            await self.send_message(
                connection_id,
                WebSocketResponse(
                    type="research_progress",
                    data={
                        "status": "running",
                        "message": "Research workflow is running...",
                    },
                ),
            )

            # Execute research
            results = await self.chat_service.start_research(
                session_id, use_code_interpreter
            )

            # Send completion
            await self.send_message(
                connection_id,
                WebSocketResponse(
                    type="research_completed",
                    data={"results": results, "status": "completed"},
                ),
            )

        except Exception as e:
            self.logger.error(f"Error in research workflow: {e}")
            await self.send_message(
                connection_id,
                WebSocketResponse(
                    type="research_error", data={}, success=False, error=str(e)
                ),
            )

    async def _handle_get_session_status(
        self, connection_id: str, data: Dict
    ) -> WebSocketResponse:
        """Handle get session status request."""
        try:
            session_id = data.get("session_id")
            if not session_id:
                return WebSocketResponse(
                    type="error", data={}, success=False, error="session_id is required"
                )

            status = self.chat_service.get_session_status(session_id)

            return WebSocketResponse(type="session_status", data=status)

        except Exception as e:
            return WebSocketResponse(type="error", data={}, success=False, error=str(e))
