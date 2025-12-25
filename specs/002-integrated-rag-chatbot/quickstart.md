# Quickstart Guide: Integrated RAG Chatbot for Published Interactive Book

## Prerequisites
- Python 3.11+
- pip package manager
- Git (optional, for cloning the repository)

## Setup

### 1. Clone the Repository
```bash
git clone <repository-url>  # If using a repository
# OR download and extract the source files
```

### 2. Create a Virtual Environment
```bash
cd integrated-rag-chatbot  # Navigate to the project directory
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
Create a `.env` file in the project root based on the `.env.example`:

```bash
cp .env.example .env
```

Then update the `.env` file with your actual credentials:

```env
COHERE_API_KEY=your_cohere_api_key_here
QDRANT_URL=your_qdrant_cluster_url_here
QDRANT_API_KEY=your_qdrant_api_key_here
NEON_DATABASE_URL=your_neon_database_connection_string_here
```

## Running the Application

### 1. Start the Backend Server
```bash
cd backend
uvicorn src.api.main:app --reload --port=8000
```

The API will be available at `http://localhost:8000`.

### 2. Ingest a Book
Use the `/ingest` endpoint to upload and process your book:

```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "accept: application/json" \
  -F "book_file=@path/to/your/book.pdf"
```

Supported formats: PDF, text (.txt), Markdown (.md).

### 3. Query the Chatbot
Send queries to the chat endpoint:

```bash
# General query about the book
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main theme of this book?"
  }'

# Query with selected text context (response will be based only on provided text)
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain this concept",
    "selected_text": "The concept of RAG involves retrieval-augmented generation which..."
  }'
```

## Embedding the Widget

### 1. Copy the Widget HTML
Include the `frontend/widget.html` file in your book's HTML:

```html
<!-- Include this where you want the chat widget to appear -->
<iframe src="path/to/widget.html" width="100%" height="500px"></iframe>
```

Or embed the JavaScript directly if you prefer inline integration:

```html
<div id="rag-chatbot-widget">
  <!-- Load the widget content -->
</div>
<script src="path/to/widget.js"></script>
```

### 2. Configure API Endpoint
Update the widget's backend API endpoint if your server is hosted elsewhere:

In `frontend/widget.js`, change the `API_BASE_URL`:
```javascript
const API_BASE_URL = 'http://localhost:8000';  // Update to your server's URL
```

## API Endpoints

### POST /ingest
Upload and process a book file for RAG queries.
- Request: `multipart/form-data` with file field
- Response: JSON with ingestion status

### POST /chat
Query the chatbot with an optional selected text context.
- Request: JSON with `query` and optional `selected_text`
- Response: JSON with `response` and optional `sources`

### GET /health
Basic health check.
- Response: JSON with health status

## Verification

After setup, verify everything works:

1. Check the health endpoint: `GET http://localhost:8000/health`
2. Ingest a sample book file
3. Query about content in the book to ensure RAG works
4. Test with selected text to ensure it responds only based on provided text

## Troubleshooting

- **API Key Issues**: Verify all API keys in `.env` are correct and have appropriate permissions
- **Database Connection**: Ensure the Neon Postgres connection string is correct
- **Qdrant Connection**: Confirm Qdrant URL and API key are valid
- **File Upload Issues**: Ensure uploaded files are in supported formats (PDF, text, Markdown)
- **Cohere API Issues**: Check your Cohere API key and account limits

## Development

For development, run the application with hot reloading:

```bash
uvicorn src.api.main:app --reload
```

Run tests:
```bash
pytest tests/
```

## Next Steps

- Customize the chat widget's appearance to match your book's design
- Add more sophisticated text selection handling
- Implement session management for conversation history (if needed)
- Add logging and monitoring for production use