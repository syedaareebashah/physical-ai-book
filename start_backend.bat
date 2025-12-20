@echo off
echo Starting RAG Chatbot Backend Server...

REM Navigate to the backend-chatbot directory and then to its backend subdirectory
cd /d "%~dp0backend-chatbot\backend"

REM Start the FastAPI server
python run_server.py

echo Server stopped.
pause