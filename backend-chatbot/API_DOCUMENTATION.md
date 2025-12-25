# API Endpoints Documentation

## Overview
This document describes the API endpoints available in the Integrated RAG Chatbot application.

## Base URL
All endpoints are relative to the base URL where the application is hosted (e.g., `http://localhost:8000`).

## Endpoints

### 1. Ingest Book Content
Upload and process a book file for RAG queries.

- **Method**: `POST`
- **Endpoint**: `/ingest`
- **Content-Type**: `multipart/form-data`
- **Description**: Upload and process a book file (PDF, text, or Markdown) for RAG queries.
- **Request Body**:
  - `book_file`: The file to be uploaded (supports PDF, text, Markdown)

- **Success Response** (200 OK):
```json
{
  "source": "example-book.pdf",
  "chunks_processed": 120,
  "status": "success"
}
```

- **Error Responses**:
  - `400 Bad Request`: File format not supported or ingestion error
  - `500 Internal Server Error`: Unexpected server error during ingestion

### 2. Chat with Bot
Query the chatbot with an optional selected text context.

- **Method**: `POST`
- **Endpoint**: `/chat`
- **Content-Type**: `application/json`
- **Description**: Query the chatbot with an optional selected text context.
  - If `selected_text` is provided, answers will be based only on that text.
  - Otherwise, uses RAG to retrieve relevant information from the book.

- **Request Body**:
```json
{
  "query": "string (required)",
  "selected_text": "string (optional)"
}
```

- **Success Response** (200 OK):
```json
{
  "response": "The answer to your question...",
  "sources": [
    {
      "text": "Relevant text snippet...",
      "page_number": 42
    }
  ]
}
```

- **Error Responses**:
  - `400 Bad Request`: Invalid request parameters
  - `500 Internal Server Error`: Unexpected server error during processing

### 3. Health Check
Basic health check to verify the service is running.

- **Method**: `GET`
- **Endpoint**: `/health`
- **Description**: Check if the service is running and responding.

- **Success Response** (200 OK):
```json
{
  "status": "healthy",
  "message": "RAG Chatbot API is running"
}
```

## Example Requests

### Ingesting a Book
```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "accept: application/json" \
  -F "book_file=@path/to/your/book.pdf"
```

### Querying the Chatbot (with selected text)
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain this concept",
    "selected_text": "The concept of RAG involves retrieval-augmented generation where..."
  }'
```

### Querying the Chatbot (without selected text)
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main theme of this book?"
  }'
```

### Health Check
```bash
curl -X GET "http://localhost:8000/health"
```

## Error Handling
The API uses standard HTTP status codes:
- `200`: Success
- `400`: Client error (bad request, validation error)
- `413`: Payload too large (if file size validation is implemented)
- `500`: Server error (internal server error)

Error responses follow the format:
```json
{
  "status": "error",
  "status_code": 400,
  "message": "Error description"
}
```