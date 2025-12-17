#!/bin/bash
# Script to start the RAG chatbot backend server

echo "Starting RAG Chatbot Backend Server..."

# Navigate to the chatbot-backend directory
cd "$(dirname "$0")/chatbot-backend"

# Start the FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

echo "Server stopped."