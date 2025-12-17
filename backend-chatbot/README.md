# Integrated RAG Chatbot for Published Interactive Book

This project implements a fully functional Retrieval-Augmented Generation (RAG) chatbot that can be embedded in a published web-based book. The system will answer questions about the entire book content using RAG and, when users select specific text in the book, will answer based ONLY on that selected text (ignoring retrieval).

## Features

- **RAG-based Q&A**: Ask questions about the book content and receive accurate answers based on the book's information
- **Selected Text Context**: When users select specific text in the book, the system responds exclusively based on that text without referencing other parts of the book
- **Multi-Format Support**: Supports ingestion of PDF, text, and Markdown formats
- **Embeddable Widget**: Provides an easy-to-integrate widget that can be embedded in any web-based book interface
- **API Endpoints**: Provides REST API endpoints for ingestion and chat functionality

## Architecture

- **Backend**: FastAPI application
- **LLM Provider**: Cohere API (embed-english-v3.0 for embeddings, command-r-plus for generation)
- **Vector Database**: Qdrant Cloud
- **Metadata Database**: Neon Serverless Postgres
- **Frontend**: Embeddable HTML/JS widget

## Setup

### Prerequisites

- Python 3.11+
- pip package manager

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd integrated-rag-chatbot
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   - Copy `.env.example` to `.env`
   - Update `.env` with your actual credentials

## Configuration

Create a `.env` file based on `.env.example` with the following variables:

- `COHERE_API_KEY`: Your Cohere API key
- `QDRANT_URL`: Your Qdrant cluster URL
- `QDRANT_API_KEY`: Your Qdrant API key
- `NEON_DATABASE_URL`: Your Neon Postgres connection string

## Usage

### Running the Backend

```bash
cd backend
uvicorn src.api.main:app --reload --port=8000
```

The API will be available at `http://localhost:8000`.

### API Endpoints

- `POST /ingest`: Upload and process a book file (PDF, text, or Markdown)
- `POST /chat`: Query the chatbot with optional selected text context
- `GET /health`: Health check endpoint

### Embedding the Widget

Include the `frontend/widget.html` file in your book's HTML:

```html
<iframe src="path/to/widget.html" width="100%" height="500px"></iframe>
```

## Development

For development, run the application with hot reloading:

```bash
uvicorn src.api.main:app --reload
```

Run tests:
```bash
pytest tests/
```

## Project Structure

```
backend/
├── src/
│   ├── config/           # Environment variables and client initialization
│   ├── models/           # Database models (SQLAlchemy)
│   ├── services/         # Business logic (ingestion, vector store, RAG)
│   ├── api/              # API endpoints (FastAPI)
│   └── utils/            # Utility functions (text splitting)
├── frontend/
│   └── widget.html       # Embeddable chat widget
├── tests/
│   ├── unit/
│   ├── integration/
│   └── contract/
├── requirements.txt
├── README.md
├── .env.example
└── .gitignore
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request