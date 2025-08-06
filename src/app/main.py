"""
FastAPI application with WebSocket support for chat agents.
"""

import uuid
import logging
import json
from typing import Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .models import (
    ChatMessageRequest,
    ChatMessageResponse,
    ChatSessionCreate,
    ChatSessionResponse,
    ErrorResponse,
)
from .chat_service import ChatService
from .websocket_manager import ConnectionManager


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Research Chat API",
    description="API for conversational research planning with multi-agent workflow",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
chat_service = ChatService()
connection_manager = ConnectionManager()


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Research Chat API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_sessions": chat_service.get_active_sessions_count(),
        "active_connections": len(connection_manager.active_connections),
    }


# REST API Endpoints


@app.post("/api/sessions", response_model=ChatSessionResponse)
async def create_session(request: ChatSessionCreate):
    """Create a new chat session."""
    try:
        session = await chat_service.create_session(request.initial_question)
        return session
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail="Failed to create session")


@app.get("/api/sessions/{session_id}", response_model=ChatSessionResponse)
async def get_session(session_id: str):
    """Get details of an existing chat session."""
    session = await chat_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.post("/api/sessions/{session_id}/messages", response_model=ChatMessageResponse)
async def send_message(session_id: str, request: ChatMessageRequest):
    """Send a message to a chat session."""
    try:
        response = await chat_service.send_message(session_id, request.message)
        return response
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error sending message: {e}")
        raise HTTPException(status_code=500, detail="Failed to send message")


@app.delete("/api/sessions/{session_id}")
async def end_session(session_id: str):
    """End a chat session."""
    success = chat_service.end_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session ended successfully"}


# WebSocket Endpoint


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat."""
    connection_id = str(uuid.uuid4())
    await connection_manager.connect(websocket, connection_id)

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()

            try:
                # Parse JSON message
                message_data = json.loads(data)
                await connection_manager.handle_message(connection_id, message_data)

            except json.JSONDecodeError:
                await connection_manager._send_error(
                    connection_id, "Invalid JSON format"
                )
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                await connection_manager._send_error(
                    connection_id, "Internal server error"
                )

    except WebSocketDisconnect:
        connection_manager.disconnect(connection_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        connection_manager.disconnect(connection_id)


# Exception handlers


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Error",
            "message": exc.detail,
            "status_code": exc.status_code,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "status_code": 500,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
