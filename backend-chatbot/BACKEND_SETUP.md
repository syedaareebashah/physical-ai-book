# Backend Setup Instructions

The backend has a compatibility issue with Python 3.13. To run the backend server, please follow these steps:

## Option 1: Using an older Python version (Recommended)

1. Install Python 3.11 or 3.12 if not already installed
2. Run the setup script:
   ```
   setup_backend_env.bat
   ```
3. Follow the instructions from the script to activate the environment and start the server

## Option 2: Using Docker (if available)

If you have Docker installed:

1. Navigate to the backend directory:
   ```
   cd backend-chatbot/backend
   ```
2. Build the Docker image:
   ```
   docker build -t rag-chatbot .
   ```
3. Run the container:
   ```
   docker run -p 8000:8000 rag-chatbot
   ```

## Starting the Backend Server

Once the environment is set up:

1. Activate the virtual environment:
   ```
   venv_backend\Scripts\activate
   ```
2. Navigate to the backend directory:
   ```
   cd backend
   ```
3. Start the server:
   ```
   python run_server.py
   ```

The server will be available at http://localhost:8000

## API Endpoints

- `/api/chat` - Main chat endpoint for the frontend
- `/api/chat/with-selected-text` - Chat endpoint with selected text context
- `/api/chat/history/{session_id}` - Get chat history for a session
- `/api/knowledge/upload` - Upload knowledge documents
- `/health` - Health check endpoint

## Frontend Integration

The frontend is configured to connect to the backend at:
- Development: http://localhost:8000
- Production: [your-production-url]

The frontend chatbot widget will automatically connect to the backend when available.