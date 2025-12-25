# Physical AI & Humanoid Robotics RAG Chatbot

This repository contains a complete RAG (Retrieval-Augmented Generation) chatbot system for the Physical AI & Humanoid Robotics educational platform. The system combines a FastAPI backend with a Docusaurus-integrated React frontend to provide an interactive learning experience.

## Architecture Overview

The system consists of three main components:

1. **Backend Service** (`chatbot-backend/`): FastAPI application handling RAG logic, vector storage, and API endpoints
2. **Frontend Integration** (`physical-ai-book/`): Docusaurus documentation site with integrated ChatKit component
3. **Content Pipeline** (`scripts/`): Document ingestion and processing scripts

## Features

- **RAG-powered Q&A**: Answers questions based on ingested knowledge base documents from the Physical AI curriculum
- **Document Ingestion**: Supports PDF, DOCX, Markdown, and MDX files
- **Conversation Context**: Maintains conversation history and context
- **Source Citations**: Provides citations for answers from knowledge base
- **Responsive UI**: Mobile-friendly chat interface that matches the site's aesthetic
- **Secure API**: Implements proper validation, sanitization, and rate limiting

## Tech Stack

### Backend
- **Framework**: FastAPI
- **Language**: Python 3.11+
- **Vector Store**: Qdrant
- **Database**: PostgreSQL (asyncpg)
- **AI Services**: Google Gemini API
- **Document Processing**: PyPDF2, python-docx, langchain-text-splitters

### Frontend
- **Framework**: Docusaurus v3
- **UI Library**: React 18+
- **Styling**: Tailwind CSS, custom CSS modules
- **API Client**: Custom fetch-based client

## Project Structure

```
├── chatbot-backend/          # FastAPI backend service
│   ├── app/                  # Application code
│   │   ├── routers/          # API endpoints
│   │   ├── services/         # Business logic (RAG, vector store, etc.)
│   │   ├── models/           # Data models and Pydantic schemas
│   │   ├── utils/            # Utility functions
│   │   └── main.py           # Application entry point
│   ├── tests/                # Test suite
│   ├── requirements.txt      # Python dependencies
│   ├── Dockerfile            # Container configuration
│   └── README.md             # Backend documentation
├── physical-ai-book/         # Docusaurus documentation site
│   ├── src/
│   │   └── components/
│   │       └── ChatKit/      # Chat interface components
│   ├── src/pages/chat.tsx    # Chat page integration
│   └── docusaurus.config.ts  # Site configuration
├── scripts/
│   └── ingest_book_content.py # Content ingestion script
└── .github/workflows/
    └── deploy-backend.yml    # CI/CD pipeline
```

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd chatbot-backend
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and service URLs
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd physical-ai-book
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

## Running the System

### Development Mode

1. Start Qdrant vector database (using Docker):
   ```bash
   docker run -p 6333:6333 -p 6334:6334 \
       -v ./qdrant_storage:/qdrant/storage:z \
       qdrant/qdrant
   ```

2. Run the backend:
   ```bash
   cd chatbot-backend
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. Run the frontend:
   ```bash
   cd physical-ai-book
   npm run start
   ```

### Ingest Content

To populate the knowledge base with content from the Physical AI curriculum:

```bash
cd chatbot-backend
python scripts/ingest_book_content.py
```

## API Endpoints

### Chat API
- `POST /api/chat` - Send message and get RAG response
- `GET /api/chat/history/{session_id}` - Get conversation history

### Knowledge Base API
- `POST /api/knowledge/upload` - Upload documents
- `GET /api/knowledge/status` - Get indexing status

### Health Check
- `GET /api/health` - Service health status

## Frontend Integration

The chat interface is available at `/chat` on the Docusaurus site. The ChatKit component can also be embedded in any page:

```jsx
import ChatKit from '@site/src/components/ChatKit';

function MyPage() {
  return (
    <div>
      <ChatKit
        endpoint="http://localhost:8000/api"
        sessionId={null} // Will be generated automatically
      />
    </div>
  );
}
```

## Deployment

The backend is configured for deployment with GitHub Actions to container registries. The workflow builds and pushes a Docker image when changes are pushed to the main branch.

## Testing

Run backend tests with pytest:
```bash
cd chatbot-backend
python -m pytest tests/
```

## Environment Variables

Required environment variables are documented in `.env.example`. Key variables include:

- `GEMINI_API_KEY` - Google Gemini API key
- `QDRANT_URL` - Qdrant vector database URL
- `QDRANT_API_KEY` - Qdrant API key (if using cloud)
- `DATABASE_URL` - PostgreSQL database connection string

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Specify your license here]