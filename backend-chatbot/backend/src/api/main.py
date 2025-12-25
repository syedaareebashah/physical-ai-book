from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import logging
import uuid
from datetime import datetime
from ..services.ingestion import IngestionService
from ..services.rag import RAGService
from ..config.config import Base, engine
from contextlib import asynccontextmanager
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lifespan for DB creation
@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")
    yield
    logger.info("Application shutting down")

# Initialize FastAPI app
app = FastAPI(
    title="Integrated RAG Chatbot API",
    description="API for RAG-based chatbot with book content",
    version="1.0.0",
    lifespan=lifespan
)

# ADD CORS MIDDLEWARE PROPERLY - YE BAAD MEIN ADD KARNA HAI!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Sab allow for demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
ingestion_service = IngestionService()
rag_service = RAGService()

# In-memory storage for session data (in a real application, use a proper database)
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
@app.post("/ingest")
async def ingest_book(book_file: UploadFile = File(...)):
    logger.info(f"Received ingestion request for file: {book_file.filename}")

    allowed_extensions = ['.pdf', '.txt', '.text', '.md', '.markdown']
    file_extension = os.path.splitext(book_file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")

    try:
        contents = await book_file.read()
        temp_file_path = ingestion_service.save_uploaded_file(contents, book_file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

    try:
        result = ingestion_service.process_file(temp_file_path, book_file.filename)
        logger.info(f"Processed {book_file.filename}: {result.get('chunks_processed', 0)} chunks")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass

# Original chat endpoint for widget.html
@app.post("/chat")
async def chat_with_bot(request: ChatRequest):
    logger.info(f"Chat query: {request.query[:50]}... selected_text: {bool(request.selected_text)}")

    try:
        rag_response = rag_service.get_response(request.query, request.selected_text or "")
        # Extract just the response text from the RAGResponse object
        # YE RETURN KAR – widget isko expect kar raha hai!
        return {"response": rag_response.response}
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        logger.error(traceback.format_exc())
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

        # Get response from RAG service
        rag_response = rag_service.get_response(request.message, "")

        # Add assistant message to session
        assistant_message = {
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": rag_response.response,
            "timestamp": datetime.now().isoformat(),
            "sources": [source.dict() for source in rag_response.sources]
        }
        sessions[session_id].append(assistant_message)

        return ChatKitResponse(
            message=rag_response.response,
            session_id=session_id,
            sources=[source.dict() for source in rag_response.sources]
        )
    except Exception as e:
        logger.error(f"ChatKit chat error: {str(e)}")
        logger.error(traceback.format_exc())
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

        # Get response from RAG service using selected text as context
        rag_response = rag_service.get_response(request.message, selected_text or "")

        # Add assistant message to session
        assistant_message = {
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": rag_response.response,
            "timestamp": datetime.now().isoformat(),
            "sources": [source.dict() for source in rag_response.sources]
        }
        sessions[session_id].append(assistant_message)

        return ChatKitResponse(
            message=rag_response.response,
            session_id=session_id,
            sources=[source.dict() for source in rag_response.sources]
        )
    except Exception as e:
        logger.error(f"ChatKit with selected text error: {str(e)}")
        logger.error(traceback.format_exc())
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
async def upload_knowledge(book_file: UploadFile = File(...)):
    logger.info(f"Received knowledge upload request for file: {book_file.filename}")

    allowed_extensions = ['.pdf', '.txt', '.text', '.md', '.markdown']
    file_extension = os.path.splitext(book_file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")

    try:
        contents = await book_file.read()
        temp_file_path = ingestion_service.save_uploaded_file(contents, book_file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

    try:
        result = ingestion_service.process_file(temp_file_path, book_file.filename)
        logger.info(f"Processed {book_file.filename}: {result.get('chunks_processed', 0)} chunks")
        return {"status": "success", "chunks_processed": result.get('chunks_processed', 0)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass

# Endpoint for knowledge status for ChatKit frontend (placeholder)
@app.get("/api/knowledge/status")
async def get_knowledge_status():
    return {"status": "ready", "documents": 0, "indexed_chunks": 0}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "RAG Chatbot API running!"}