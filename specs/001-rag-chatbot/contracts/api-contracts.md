# API Contracts: RAG Chatbot

**Feature**: RAG Chatbot
**Created**: 2025-12-07

## Chat API

### POST /api/chat

**Description**: Send a message and receive a response from the RAG system

#### Request
```json
{
  "message": "string, user's question",
  "session_id": "string, optional session identifier (will be created if not provided)"
}
```

#### Response (200 OK)
```json
{
  "message": "string, assistant's response",
  "session_id": "string, session identifier",
  "sources": [
    {
      "filename": "string, source document name",
      "page": "number, page number if applicable",
      "content": "string, relevant excerpt",
      "score": "number, relevance score 0-1"
    }
  ]
}
```

#### Error Responses
- `400 Bad Request`: Invalid request format
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Processing error

### GET /api/chat/history/{session_id}

**Description**: Retrieve conversation history for a session

#### Response (200 OK)
```json
{
  "session_id": "string",
  "messages": [
    {
      "id": "string, message UUID",
      "role": "string, user|assistant",
      "content": "string, message content",
      "timestamp": "string, ISO 8601 timestamp",
      "sources": "array, source documents for assistant messages"
    }
  ]
}
```

#### Error Responses
- `404 Not Found`: Session does not exist
- `500 Internal Server Error`: Database error

## Knowledge Base API

### POST /api/knowledge/upload

**Description**: Upload documents to knowledge base for indexing

#### Request (multipart/form-data)
- Files: One or more documents (PDF, DOCX, MD)

#### Response (200 OK)
```json
{
  "status": "success",
  "processed_files": ["filename1", "filename2"],
  "message": "string, processing status message"
}
```

#### Error Responses
- `400 Bad Request`: Invalid file format
- `413 Payload Too Large`: File too large
- `500 Internal Server Error`: Processing error

### GET /api/knowledge/status

**Description**: Get status of knowledge base indexing process

#### Response (200 OK)
```json
{
  "total_documents": "number",
  "indexed_chunks": "number",
  "last_update": "string, ISO 8601 timestamp",
  "status": "string, ready|processing|error"
}
```

## Health Check API

### GET /health

**Description**: Check health status of the service

#### Response (200 OK)
```json
{
  "status": "healthy",
  "timestamp": "string, ISO 8601 timestamp",
  "services": {
    "database": "string, status",
    "vector_store": "string, status",
    "openai_api": "string, status"
  }
}
```