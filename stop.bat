@echo off
title Multimodal Document Intelligence Platform - Shutdown
cls
color 0C
echo =====================================================================
echo    MULTIMODAL DOCUMENT INTELLIGENCE PLATFORM - SHUTDOWN
echo =====================================================================
echo.
echo [*] Terminating services running on Port 8000 (FastAPI) and Port 8501 (Streamlit)...

powershell -NoProfile -Command ^
    "$ports = @(8000, 8501); " ^
    "$conns = Get-NetTCPConnection -LocalPort $ports -ErrorAction SilentlyContinue; " ^
    "if ($conns) { " ^
    "    $pids = $conns | Select-Object -ExpandProperty OwningProcess -Unique; " ^
    "    foreach ($p in $pids) { " ^
    "        try { Stop-Process -Id $p -Force -ErrorAction SilentlyContinue; Write-Host \"[+] Stopped process ID: $p\" } catch {} " ^
    "    } " ^
    "} else { " ^
    "    Write-Host \"[i] No active processes listening on ports 8000 or 8501.\"; " ^
    "}"

echo.
echo [*] Closing console windows and background tunnel daemons...

taskkill /F /FI "WINDOWTITLE eq FastAPI Backend (8000)*" /T >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq Streamlit Dashboard (8501)*" /T >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq Public Tunnel Manager*" /T >nul 2>&1

powershell -NoProfile -Command ^
    "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*tunnel_manager.py*' -or $_.CommandLine -like '*uvicorn app.main:app*' -or $_.CommandLine -like '*streamlit run frontend/app.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; Write-Host \"[+] Terminated task: $($_.Name) (PID: $($_.ProcessId))\" }"

echo.
echo =====================================================================
echo   [SUCCESS] All platform services have been cleanly stopped!
echo =====================================================================
echo.
pause
