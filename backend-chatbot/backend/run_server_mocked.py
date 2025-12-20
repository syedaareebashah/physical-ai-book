import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import logging
import uuid
from datetime import datetime

# Mock services to avoid SQLAlchemy dependency issues
class MockRAGService:
    def get_response(self, query: str, context: str = ""):
        # Mock response - in a real implementation, this would use the RAG system
        response_text = f"Mock response to: '{query}'. This is a simulated response since the backend dependencies aren't fully functional. In a working setup, this would contain information from the book related to your query."
        return MockRAGResponse(response_text, [])

class MockRAGResponse:
    def __init__(self, response, sources):
        self.response = response
        self.sources = sources

class MockIngestionService:
    def save_uploaded_file(self, contents, filename):
        return f"/tmp/{filename}"
    
    def process_file(self, file_path, filename):
        return {"chunks_processed": 0, "status": "mocked"}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Integrated RAG Chatbot API (Mocked Version)",
    description="API for RAG-based chatbot with book content (mocked for compatibility)",
    version="1.0.0"
)

# ADD CORS MIDDLEWARE PROPERLY
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize mock services
ingestion_service = MockIngestionService()
rag_service = MockRAGService()

# In-memory storage for session data
sessions: Dict[str, List[Dict[str, Any]]] = {}

# Request models
class ChatRequest(BaseModel):
    query: str
    selected_text: Optional[str] = None

class ChatKitRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatWithSelectedTextRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

# Response models
class ChatResponse(BaseModel):
    response: str

class ChatKitResponse(BaseModel):
    message: str
    session_id: str
    sources: List[Dict[str, Any]] = []

# Endpoints
@app.post("/chat")
async def chat_with_bot(request: ChatRequest):
    logger.info(f"Chat query: {request.query[:50]}... selected_text: {bool(request.selected_text)}")

    try:
        rag_response = rag_service.get_response(request.query, request.selected_text or "")
        return {"response": rag_response.response}
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing query")

# New chat endpoint for ChatKit frontend compatibility
@app.post("/api/chat")
async def chat_with_chatkit(request: ChatKitRequest):
    logger.info(f"ChatKit chat message: {request.message[:50]}...")

    # Create a new session if one doesn't exist
    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = []

    try:
        # Add user message to session
        user_message = {
            "id": str(uuid.uuid4()),
            "role": "user",
            "content": request.message,
            "timestamp": datetime.now().isoformat()
        }
        sessions[session_id].append(user_message)

        # Get response from mock RAG service
        rag_response = rag_service.get_response(request.message, "")

        # Add assistant message to session
        assistant_message = {
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": rag_response.response,
            "timestamp": datetime.now().isoformat(),
            "sources": rag_response.sources
        }
        sessions[session_id].append(assistant_message)

        return ChatKitResponse(
            message=rag_response.response,
            session_id=session_id,
            sources=rag_response.sources
        )
    except Exception as e:
        logger.error(f"ChatKit chat error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing query")

# Endpoint for chat with selected text for ChatKit frontend
@app.post("/api/chat/with-selected-text")
async def chat_with_selected_text(
    request: ChatWithSelectedTextRequest,
    selected_text: str = None
):
    logger.info(f"ChatKit chat with selected text: {request.message[:50]}...")

    # Create a new session if one doesn't exist
    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = []

    try:
        # Add user message to session
        user_message = {
            "id": str(uuid.uuid4()),
            "role": "user",
            "content": request.message,
            "timestamp": datetime.now().isoformat()
        }
        sessions[session_id].append(user_message)

        # Get response from mock RAG service using selected text as context
        rag_response = rag_service.get_response(request.message, selected_text or "")

        # Add assistant message to session
        assistant_message = {
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": rag_response.response,
            "timestamp": datetime.now().isoformat(),
            "sources": rag_response.sources
        }
        sessions[session_id].append(assistant_message)

        return ChatKitResponse(
            message=rag_response.response,
            session_id=session_id,
            sources=rag_response.sources
        )
    except Exception as e:
        logger.error(f"ChatKit with selected text error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing query")

# Endpoint to get conversation history for ChatKit frontend
@app.get("/api/chat/history/{session_id}")
async def get_conversation_history(session_id: str):
    logger.info(f"Retrieving conversation history for session: {session_id}")

    if session_id not in sessions:
        return {"messages": []}

    # Map our internal format to the format expected by ChatKit
    messages = []
    for msg in sessions[session_id]:
        messages.append({
            "id": msg["id"],
            "role": msg["role"],
            "content": msg["content"],
            "timestamp": msg["timestamp"],
            "sources": msg.get("sources", [])
        })

    return {"messages": messages}

# Endpoint for knowledge upload for ChatKit frontend (placeholder)
@app.post("/api/knowledge/upload")
async def upload_knowledge():
    return {"status": "success", "chunks_processed": 0}

# Endpoint for knowledge status for ChatKit frontend (placeholder)
@app.get("/api/knowledge/status")
async def get_knowledge_status():
    return {"status": "ready", "documents": 0, "indexed_chunks": 0}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Mock RAG Chatbot API running!"}

if __name__ == "__main__":
    logger.info("Starting server with mocked dependencies...")
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info"
    )