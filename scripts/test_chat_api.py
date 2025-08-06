#!/usr/bin/env python3
"""
Test script for the chat API endpoints.
"""

import asyncio
import json
import logging
import requests
import websockets
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API base URL
BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws"


def test_health_check():
    """Test the health check endpoint."""
    logger.info("Testing health check endpoint...")

    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()

        data = response.json()
        logger.info(f"Health check successful: {data}")
        return True
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False


def test_rest_api():
    """Test the REST API endpoints."""
    logger.info("Testing REST API endpoints...")

    try:
        # Create a new session
        logger.info("Creating new session...")
        session_data = {
            "initial_question": "How does artificial intelligence impact healthcare?"
        }

        response = requests.post(f"{BASE_URL}/api/sessions", json=session_data)
        response.raise_for_status()

        session = response.json()
        session_id = session["session_id"]
        logger.info(f"Session created: {session_id}")
        logger.info(f"Initial messages: {len(session['messages'])}")

        # Send a message
        logger.info("Sending message...")
        message_data = {"message": "Can you focus on diagnostic applications?"}

        response = requests.post(
            f"{BASE_URL}/api/sessions/{session_id}/messages", json=message_data
        )
        response.raise_for_status()

        message_response = response.json()
        logger.info(f"Response received: {message_response['message'][:100]}...")

        # Get session details
        logger.info("Getting session details...")
        response = requests.get(f"{BASE_URL}/api/sessions/{session_id}")
        response.raise_for_status()

        session_details = response.json()
        logger.info(f"Total messages in session: {len(session_details['messages'])}")

        # End session
        logger.info("Ending session...")
        response = requests.delete(f"{BASE_URL}/api/sessions/{session_id}")
        response.raise_for_status()

        logger.info("REST API test completed successfully!")
        return True

    except Exception as e:
        logger.error(f"REST API test failed: {e}")
        return False


async def test_websocket():
    """Test the WebSocket endpoint."""
    logger.info("Testing WebSocket endpoint...")

    try:
        async with websockets.connect(WS_URL) as websocket:
            # Start a new session
            logger.info("Starting WebSocket session...")
            start_message = {
                "type": "session_start",
                "data": {
                    "initial_question": "What are the latest developments in quantum computing?"
                },
            }

            await websocket.send(json.dumps(start_message))

            # Wait for session created response
            response = await websocket.recv()
            data = json.loads(response)

            if data["type"] == "session_created":
                session_id = data["session_id"]
                logger.info(f"WebSocket session created: {session_id}")
                logger.info(f"Initial messages: {len(data['data']['messages'])}")

                # Send a message
                logger.info("Sending WebSocket message...")
                chat_message = {
                    "type": "message",
                    "session_id": session_id,
                    "data": {"message": "Tell me about quantum algorithms"},
                }

                await websocket.send(json.dumps(chat_message))

                # Wait for response
                response = await websocket.recv()
                response_data = json.loads(response)

                if response_data["type"] == "message_response":
                    logger.info(
                        f"WebSocket response: {response_data['data']['message'][:100]}..."
                    )

                    # End session
                    logger.info("Ending WebSocket session...")
                    end_message = {"type": "session_end", "session_id": session_id}

                    await websocket.send(json.dumps(end_message))

                    # Wait for end confirmation
                    response = await websocket.recv()
                    end_data = json.loads(response)

                    if end_data["type"] == "session_ended":
                        logger.info("WebSocket session ended successfully!")
                        return True
                    else:
                        logger.error(f"Unexpected end response: {end_data}")
                        return False
                else:
                    logger.error(f"Unexpected message response: {response_data}")
                    return False
            else:
                logger.error(f"Unexpected session start response: {data}")
                return False

    except Exception as e:
        logger.error(f"WebSocket test failed: {e}")
        return False


async def main():
    """Run all tests."""
    logger.info("Starting chat API tests...")

    # Test health check
    health_ok = test_health_check()

    # Test REST API
    rest_ok = test_rest_api()

    # Test WebSocket
    ws_ok = await test_websocket()

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Health Check: {'PASS' if health_ok else 'FAIL'}")
    logger.info(f"REST API: {'PASS' if rest_ok else 'FAIL'}")
    logger.info(f"WebSocket: {'PASS' if ws_ok else 'FAIL'}")
    logger.info("=" * 50)

    if all([health_ok, rest_ok, ws_ok]):
        logger.info("All tests PASSED!")
        return 0
    else:
        logger.error("Some tests FAILED!")
        return 1


if __name__ == "__main__":
    import sys

    print("Chat API Test Script")
    print("Make sure the API server is running: python src/app/main.py")
    print("Or: uvicorn src.app.main:app --reload")
    print()

    exit_code = asyncio.run(main())
    sys.exit(exit_code)
