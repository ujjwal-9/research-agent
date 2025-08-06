"""
FastAPI application with WebSocket endpoints for the chat agent.
"""

import uuid
import os
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .websocket_service import WebSocketManager
from .chat_service import ChatService
from .models import (
    ChatSessionRequest,
    ChatMessageRequest,
    ChatResponse,
    ResearchPlanResponse,
    StartResearchRequest,
)


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/Users/ujjwal/code/redesign/logs/api.log"),
        logging.StreamHandler(),
    ],
)

app = FastAPI(
    title="Research Chat API",
    description="WebSocket API for interactive research chat agent",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize WebSocket manager
websocket_manager = WebSocketManager()

# Initialize chat service for REST endpoints
chat_service = ChatService()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Research Chat API",
        "version": "1.0.0",
        "websocket_endpoint": "/ws/{connection_id}",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


# WebSocket endpoint
@app.websocket("/ws/{connection_id}")
async def websocket_endpoint(websocket: WebSocket, connection_id: str):
    """
    WebSocket endpoint for real-time chat communication.

    Message types:
    - start_session: {"type": "start_session", "data": {"initial_question": "..."}}
    - send_message: {"type": "send_message", "data": {"session_id": "...", "message": "..."}}
    - get_plan: {"type": "get_plan", "data": {"session_id": "..."}}
    - start_research: {"type": "start_research", "data": {"session_id": "...", "use_code_interpreter": true}}
    - get_session_status: {"type": "get_session_status", "data": {"session_id": "..."}}
    """
    await websocket_manager.connect(websocket, connection_id)

    try:
        while True:
            # Receive message
            data = await websocket.receive_text()

            # Handle message
            await websocket_manager.handle_message(connection_id, data)

    except WebSocketDisconnect:
        websocket_manager.disconnect(connection_id)
    except Exception as e:
        logging.error(f"WebSocket error for {connection_id}: {e}")
        websocket_manager.disconnect(connection_id)


# REST API endpoints (alternative to WebSocket)


@app.post("/api/chat/session", response_model=ChatResponse)
async def start_chat_session(request: ChatSessionRequest):
    """Start a new chat session."""
    try:
        session_id, response = await chat_service.start_chat_session(
            request.initial_question
        )
        return response
    except Exception as e:
        logging.error(f"Error starting chat session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/message", response_model=ChatResponse)
async def send_chat_message(request: ChatMessageRequest):
    """Send a message to an existing chat session."""
    try:
        response = await chat_service.send_message(request.session_id, request.message)
        return response
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logging.error(f"Error sending message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chat/plan/{session_id}", response_model=ResearchPlanResponse)
async def get_research_plan(session_id: str):
    """Get the research plan for a session."""
    try:
        plan = chat_service.get_research_plan(session_id)
        if not plan:
            raise HTTPException(status_code=404, detail="No research plan found")
        return plan
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logging.error(f"Error getting research plan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/research/start")
async def start_research(request: StartResearchRequest):
    """Start the research workflow."""
    try:
        results = await chat_service.start_research(
            request.session_id, request.use_code_interpreter
        )
        return {"status": "completed", "results": results}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logging.error(f"Error starting research: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chat/status/{session_id}")
async def get_session_status(session_id: str):
    """Get the status of a chat session."""
    try:
        status = chat_service.get_session_status(session_id)
        return status
    except Exception as e:
        logging.error(f"Error getting session status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/chat/session/{session_id}")
async def cleanup_session(session_id: str):
    """Clean up a chat session."""
    try:
        chat_service.cleanup_session(session_id)
        return {"status": "cleaned"}
    except Exception as e:
        logging.error(f"Error cleaning up session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        exit(1)

    # Run the application
    uvicorn.run(
        "src.app.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
