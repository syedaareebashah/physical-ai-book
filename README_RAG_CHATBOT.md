# Physical AI Book - Integrated RAG Chatbot

This project includes an integrated Retrieval-Augmented Generation (RAG) chatbot that allows users to ask questions about the Physical AI book content. The chatbot is built using FastAPI, Neon Serverless Postgres, Qdrant Cloud, and the OpenAI API.

## Architecture

The system consists of:

1. **Backend API**: FastAPI application with RAG functionality
2. **Vector Database**: Qdrant Cloud for storing book content embeddings
3. **Frontend**: Docusaurus-based book with integrated chatbot component
4. **RAG Service**: Handles document processing, embedding generation, and similarity search

## Features

- **Question Answering**: Ask questions about the Physical AI book content
- **Selected Text Context**: Select text in the book and ask context-aware questions
- **Conversation History**: Persistent conversation sessions
- **Source Attribution**: Responses include source references to book sections

## Setup

### Prerequisites

- Python 3.8+
- Node.js and npm/yarn for the frontend
- Qdrant Cloud account (or local instance)
- Google Gemini API key (for embeddings and generation)
- Neon Serverless Postgres database

### Backend Setup

1. Navigate to the backend-chatbot directory:
   ```bash
   cd backend-chatbot
   ```

2. **Important**: Due to a compatibility issue with Python 3.13, please use Python 3.11 or 3.12 for the backend. You can use the setup script:
   ```bash
   setup_backend_env.bat
   ```
   Or manually create a virtual environment with Python 3.11 or 3.12:
   ```bash
   python3.11 -m venv venv_backend  # or python3.12
   venv_backend\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Create a `.env` file with the following configuration:
   ```env
   GEMINI_API_KEY=your_gemini_api_key
   QDRANT_URL=your_qdrant_cluster_url
   QDRANT_API_KEY=your_qdrant_api_key  # if required
   DATABASE_URL=postgresql+asyncpg://username:password@ep-xxx.us-east-1.aws.neon.tech/dbname
   ```

4. Start the backend server:
   ```bash
   cd backend  # Navigate to the backend subdirectory
   python run_server.py
   ```

### Populate Book Content

Before using the chatbot, populate the vector database with book content:

```bash
python populate_book_content.py
```

This script will:
- Extract all markdown content from the `physical-ai-book/docs` directory
- Process each document into chunks
- Generate embeddings using the configured model
- Store the embeddings in the Qdrant vector database

### Frontend Setup

1. Navigate to the physical-ai-book directory:
   ```bash
   cd physical-ai-book
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run start
   ```

## Usage

1. **Basic Q&A**: Type questions in the chatbot interface on the homepage
2. **Context-Aware Questions**: Select text in the book content, then ask questions in the chatbot - the selected text will be used as additional context
3. **Session Management**: Conversations persist through session IDs stored in the database

## API Endpoints

- `POST /api/chat` - Basic chat functionality
- `POST /api/chat/with-selected-text` - Chat with selected text context (accepts `selected_text` as query parameter)
- `GET /api/chat/history/{session_id}` - Retrieve conversation history

## Testing

Run the integration test to verify all components:

```bash
python test_integration.py
```

## Components

### Backend Services

- **RAGService**: Core RAG functionality with enhanced book content handling
- **VectorStoreService**: Qdrant integration for vector storage and retrieval
- **EmbeddingService**: Google Gemini-powered embedding generation
- **DocumentLoader**: Handles various document formats (markdown, PDF, etc.)

### Frontend Components

- **ChatKit**: Main chatbot component with selected text functionality
- **Styling**: Responsive design with dark/light mode support
- **Text Selection Integration**: Automatically captures selected book text

## Customization

To customize the chatbot for your specific book content:

1. Update the book content in `physical-ai-book/docs`
2. Modify the RAG service in `chatbot-backend/app/services/rag.py`
3. Adjust the frontend styling in `physical-ai-book/src/components/ChatKit/ChatKit.css`

## Troubleshooting

- If the chatbot returns "No relevant information found", ensure the book content was successfully populated to the vector database
- Check that your API keys are properly configured in the `.env` file
- Verify that the Qdrant connection is working
- Review the backend logs for any error messages

## Security Considerations

- API keys should be stored securely and never committed to version control
- CORS settings are configured in the backend
- Input validation is implemented for all endpoints
- Rate limiting is configured to prevent abuse