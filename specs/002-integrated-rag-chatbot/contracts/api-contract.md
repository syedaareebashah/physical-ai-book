# FastAPI Endpoints for Integrated RAG Chatbot

## POST /ingest
Upload and process a book file for RAG queries.

### Request
- **Path**: `/ingest`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`

#### Request Body
```python
from typing import Optional
from pydantic import BaseModel
from fastapi import File, UploadFile

class IngestionRequest(BaseModel):
    book_file: UploadFile  # PDF, text, or Markdown file
```

### Response
- **Success Response**: `200 OK`
```json
{
  "status": "success",
  "message": "Book ingested successfully",
  "chunks_processed": 120,
  "source": "example-book.pdf"
}
```

- **Error Response**: `400 Bad Request`
```json
{
  "status": "error",
  "message": "File format not supported"
}
```

- **Error Response**: `500 Internal Server Error`
```json
{
  "status": "error",
  "message": "Ingestion failed due to malformed or corrupted content, suggesting re-upload"
}
```

---

## POST /chat
Query the chatbot with an optional selected text context.

### Request
- **Path**: `/chat`
- **Method**: `POST`

#### Request Body
```python
from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    query: str
    selected_text: Optional[str] = None
```

### Response
- **Success Response**: `200 OK`
```json
{
  "response": "The answer to your question based on the book content...",
  "sources": [
    {
      "text": "Relevant text snippet from the book",
      "page_number": 42
    }
  ]
}
```

- **Error Response**: `400 Bad Request`
```json
{
  "status": "error",
  "message": "Query has ambiguous or unclear intent, please clarify"
}
```

---

## GET /health
Basic health check to verify the service is running.

### Request
- **Path**: `/health`
- **Method**: `GET`

### Response
- **Success Response**: `200 OK`
```json
{
  "status": "healthy",
  "timestamp": "2025-01-08T10:30:00Z"
}
```

---

## API Implementation with FastAPI

```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Optional
import cohere
from qdrant_client import QdrantClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

app = FastAPI()

# Request models
class ChatRequest(BaseModel):
    query: str
    selected_text: Optional[str] = None

# API endpoints
@app.post("/ingest")
async def ingest_book(book_file: UploadFile = File(...)):
    """
    Upload and process a book file for RAG queries.
    Supports PDF, text, and Markdown formats.
    """
    # Implementation details
    pass

@app.post("/chat")
async def chat_with_bot(request: ChatRequest):
    """
    Query the chatbot with an optional selected text context.
    If selected_text is provided, answers will be based only on that text.
    Otherwise, uses RAG to retrieve relevant information from the book.
    """
    # Implementation details
    pass

@app.get("/health")
async def health_check():
    """
    Basic health check to verify the service is running.
    """
    return {"status": "healthy"}

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "status": "error",
        "message": str(exc.detail)
    }
```