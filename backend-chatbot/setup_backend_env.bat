@echo off
echo Setting up Python virtual environment for backend...

REM Check if Python 3.11 or 3.12 is available
python3.11 --version >nul 2>&1
if %errorlevel% == 0 (
    echo Found Python 3.11
    python3.11 -m venv venv_backend
    goto :activate
)

python3.12 --version >nul 2>&1
if %errorlevel% == 0 (
    echo Found Python 3.12
    python3.12 -m venv venv_backend
    goto :activate
)

echo Python 3.11 or 3.12 not found. Please install Python 3.11 or 3.12 to run the backend.
pause
exit /b 1

:activate
echo Activating virtual environment...
call venv_backend\Scripts\activate.bat

echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Virtual environment setup complete!
echo To start the backend server:
echo 1. Run: venv_backend\Scripts\activate.bat
echo 2. Navigate to the backend directory: cd backend
echo 3. Run: python run_server.py
echo.
echo The server will be available at http://localhost:8000
pause