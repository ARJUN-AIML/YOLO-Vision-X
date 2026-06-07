@echo off
setlocal EnableDelayedExpansion

echo ===================================================
echo   YOLO Vision X  --  Real-Time Object Tracking
echo ===================================================
echo.

REM ── Locate the backend directory relative to this script ────────────────────
set "ROOT=%~dp0"
set "BACKEND=%ROOT%backend"

if not exist "%BACKEND%" (
    echo [ERROR] "backend" folder not found next to run.bat.
    echo         Expected location: %BACKEND%
    echo.
    pause
    exit /b 1
)

REM ── Resolve the virtual-environment Python executable ───────────────────────
REM    Check .venv first, then venv, both inside backend\.
set "PYTHON_EXE="

if exist "%BACKEND%\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%BACKEND%\.venv\Scripts\python.exe"
    set "ACTIVATE=%BACKEND%\.venv\Scripts\activate.bat"
    echo [OK] Found virtual environment: backend\.venv
    goto :found_venv
)

if exist "%BACKEND%\venv\Scripts\python.exe" (
    set "PYTHON_EXE=%BACKEND%\venv\Scripts\python.exe"
    set "ACTIVATE=%BACKEND%\venv\Scripts\activate.bat"
    echo [OK] Found virtual environment: backend\venv
    goto :found_venv
)

REM ── No venv found — print a clear, actionable error and exit ────────────────
echo [ERROR] No virtual environment found in the backend folder.
echo.
echo  To set one up, open a terminal in the project root and run:
echo.
echo    python -m venv backend\.venv
echo    backend\.venv\Scripts\activate
echo    pip install -r backend\requirements.txt
echo.
echo  Then double-click run.bat again.
echo.
pause
exit /b 1

:found_venv
REM ── Activate the environment (updates PATH within this shell session) ────────
echo Activating virtual environment...
call "%ACTIVATE%"

REM ── Sanity-check: make sure the resolved Python actually runs ────────────────
"%PYTHON_EXE%" --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] The virtual environment Python is not working correctly.
    echo         Path checked: %PYTHON_EXE%
    echo.
    pause
    exit /b 1
)

REM ── Verify key dependency (fastapi) is installed before launching ────────────
"%PYTHON_EXE%" -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Required Python packages are not installed.
    echo.
    echo  To install them, run:
    echo.
    echo    backend\.venv\Scripts\activate
    echo    pip install -r backend\requirements.txt
    echo.
    pause
    exit /b 1
)

REM ── Launch the FastAPI server using the venv Python directly ─────────────────
echo Starting YOLO Vision X backend...
echo Access the dashboard at:  http://127.0.0.1:8000
echo.
echo (Press Ctrl+C to stop the server)
echo.

REM Open the browser after a short delay (runs in background, does not block)
start "" /b cmd /c "timeout /t 3 /nobreak >nul && start http://127.0.0.1:8000"

REM cd into backend so relative model paths in main.py resolve correctly
cd /d "%BACKEND%"
"%PYTHON_EXE%" main.py

echo.
echo Server stopped.
pause
endlocal
