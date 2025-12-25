# Quickstart Guide: RAG Chatbot Development

**Feature**: RAG Chatbot
**Created**: 2025-12-07

## Prerequisites

- Python 3.11+
- Node.js 18+ (for Docusaurus frontend)
- Docker (for local development of vector database)
- Gemini API key
- Qdrant vector database (local or cloud)

## Environment Setup

### 1. Clone and Navigate to Project

```bash
git clone <repository-url>
cd physical-ai-book
```

### 2. Backend Setup

```bash
# Navigate to backend directory
cd chatbot-backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and service URLs
```

### 3. Frontend Setup

```bash
# Navigate to frontend directory
cd physical-ai-book  # or wherever Docusaurus is located

# Install dependencies
npm install
```

## Local Development

### 1. Start Qdrant Vector Database

Option 1: Using Docker
```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v ./qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

Option 2: Using Qdrant Cloud
- Sign up at [qdrant.tech](https://qdrant.tech)
- Create a collection named `physical_ai_book`
- Update your `.env` file with the URL and API key

### 2. Initialize Vector Store

```bash
# Run the ingestion script to process documents
python scripts/ingest_book_content.py
```

### 3. Start Backend Service

```bash
cd chatbot-backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Start Frontend

```bash
cd physical-ai-book
npm run start
```

## Configuration

### Environment Variables

Create a `.env` file in the `chatbot-backend` directory:

```env
# GEMINI Configuration
GEMINI_API_KEY="AIzaSyAz4Ol6y2y74HBxqOz6sG8ys5YJq8m8Pik"
OPENAI_MODEL=gemini-2.0-flash


# Qdrant Configuration
QDRANT_URL=http://localhost:6333  # or your Qdrant cloud URL
QDRANT_API_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.CKytWoXzr_m9lQuHqcL8Va-ee_cvHSZ7lrvEMWZ-wKE

# Database Configuration
DATABASE_URL=postgresql+asyncpg://user:password@localhost/dbname

# Application Settings
MAX_TOKENS=1000
TEMPERATURE=0.7
VECTOR_DIMENSION=1536  # For OpenAI embeddings
SIMILARITY_THRESHOLD=0.5
MAX_KNOWLEDGE_RESULTS=5
```

## Running Tests

```bash
# Backend tests
cd chatbot-backend
python -m pytest tests/

# Frontend tests (if applicable)
cd physical-ai-book
npm run test
```

## API Endpoints

### Chat API
- `POST /api/chat` - Send message and get RAG response
- `GET /api/chat/history/{session_id}` - Get conversation history

### Knowledge Base API
- `POST /api/knowledge/upload` - Upload documents
- `GET /api/knowledge/status` - Get indexing status

### Health Check
- `GET /health` - Service health status

## Example Usage

### Using the API

```bash
# Send a message
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is ROS 2?",
    "session_id": "session-123"
  }'
```

### Frontend Integration

The chat component can be integrated into Docusaurus using the ChatKit React component:

```jsx
import ChatKit from './components/ChatKit';

function MyPage() {
  return (
    <div>
      <ChatKit
        endpoint="http://localhost:8000/api/chat"
        sessionId="unique-session-id"
      />
    </div>
  );
}
```

## Troubleshooting

### Common Issues

1. **Embedding API errors**: Verify your OpenAI API key is correct and has sufficient quota
2. **Vector store connection**: Ensure Qdrant is running and accessible
3. **Document parsing**: Check that uploaded documents are in supported formats
4. **CORS errors**: Configure CORS settings in the FastAPI backend

### Logs

Backend logs are available via the console when running with `--reload` flag. For production deployments, configure logging to file as needed.