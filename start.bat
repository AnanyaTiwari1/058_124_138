@echo off
title Multimodal Document Intelligence Platform - Launcher
cls
color 0B
echo =====================================================================
echo    MULTIMODAL DOCUMENT INTELLIGENCE PLATFORM (Phase 1 - 25)
echo =====================================================================
echo.
echo [*] Checking Python environment...
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in system PATH!
    pause
    exit /b 1
)

echo [OK] Python detected.
echo.
echo [*] Starting Services...
echo.

REM 1. Start FastAPI REST Backend (Port 8000)
echo [+] Launching FastAPI Backend on http://localhost:8000 ...
start "FastAPI Backend (8000)" cmd /k "cd /d %~dp0 && python -m uvicorn app.main:app --port 8000"

REM Wait 2 seconds
timeout /t 2 /nobreak >nul

REM 2. Start Streamlit Dashboard (Port 8501)
echo [+] Launching Streamlit Dashboard on http://localhost:8501 ...
start "Streamlit Dashboard (8501)" cmd /k "cd /d %~dp0 && streamlit run frontend/app.py --server.port 8501"

REM 3. Start Persistent Public Tunnel Manager
echo [+] Launching Public Tunnel Daemon...
start "Public Tunnel Manager" cmd /k "cd /d %~dp0 && python scripts/tunnel_manager.py"

REM Wait 4 seconds
timeout /t 4 /nobreak >nul

REM Open Browser to Streamlit UI
start http://localhost:8501

echo.
echo =====================================================================
echo   [SUCCESS] All Services are running!
echo =====================================================================
echo   * Streamlit Dashboard:  http://localhost:8501
echo   * FastAPI REST Docs:    http://localhost:8000/docs
echo   * Public URL File:      data\public_url.txt
echo =====================================================================
echo   (To stop all services, run stop.bat)
echo.
pause
