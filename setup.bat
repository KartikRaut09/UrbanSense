@echo off
REM UrbanSense Quick Start Script for Windows

setlocal enabledelayedexpansion

echo ================================
echo UrbanSense Initialization Script
echo ================================
echo.

REM Check Python
echo Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    exit /b 1
)
echo ✓ Python found: 
python --version

REM Check Node
echo.
echo Checking Node.js...
node --version >nul 2>&1
if errorlevel 1 (
    echo Error: Node.js is not installed or not in PATH
    exit /b 1
)
echo ✓ Node.js found: 
node --version

REM Backend Setup
echo.
echo Setting up Backend...
cd backend

if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

if not exist ".env" (
    echo Creating .env file from template...
    copy .env.example .env
    echo Note: Update .env file with your configuration
)

echo ✓ Backend setup complete

REM Frontend Setup
echo.
echo Setting up Frontend...
cd ..\frontend

echo Installing dependencies...
call npm install

if not exist ".env" (
    echo Creating .env file...
    (
        echo VITE_API_URL=http://localhost:8000/api/v1
    ) > .env
)

echo ✓ Frontend setup complete

REM Summary
echo.
echo ================================
echo Setup Complete!
echo ================================
echo.
echo Next steps:
echo 1. Update backend\.env with your configuration
echo 2. Start backend:   cd backend ^& venv\Scripts\activate ^& python main.py
echo 3. Start frontend:  cd frontend ^& npm run dev
echo.
echo Services will be available at:
echo   - Frontend:  http://localhost:5173
echo   - Backend:   http://localhost:8000
echo   - API Docs:  http://localhost:8000/docs
echo.

pause
