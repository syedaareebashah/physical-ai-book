from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import logging
from services.ingestion import IngestionService
from services.rag import RAGService
from config.config import Base, engine
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

# Request model
class ChatRequest(BaseModel):
    query: str
    selected_text: Optional[str] = None

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

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "RAG Chatbot API running!"}