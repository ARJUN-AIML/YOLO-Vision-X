@echo off
echo ===================================================
echo Starting the Object Tracking Service
echo ===================================================
echo.

cd /d "%~dp0backend"

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found in the backend folder!
    pause
    exit /b
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Starting the FastAPI server...
python main.py

pause
