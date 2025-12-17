@echo off
echo Starting RAG Chatbot Backend Server...

REM Navigate to the chatbot-backend directory
cd /d "%~dp0chatbot-backend"

REM Start the FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

echo Server stopped.
pause