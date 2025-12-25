#!/bin/bash
# Script to start the RAG chatbot backend server

echo "Starting RAG Chatbot Backend Server..."

# Navigate to the backend-chatbot directory and then to its backend subdirectory
cd "$(dirname "$0")/backend-chatbot/backend"

# Start the FastAPI server
python run_server.py

echo "Server stopped."