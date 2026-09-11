@echo off
title QuantaRoute ENTERPRISE Launcher
echo =======================================================
echo   QuantaRoute ENTERPRISE - Starting Platform
echo =======================================================

cd /d "%~dp0"

echo [1/3] Clearing stale ports 8000 and 5500 to prevent WinError 10013...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do (
  taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5500 ^| findstr LISTENING') do (
  taskkill /F /PID %%a >nul 2>&1
)

echo [2/3] Starting Backend API on http://127.0.0.1:8000...
start "QuantaRoute Backend API" cmd /k "call .\venv\Scripts\activate.bat && python -m uvicorn app.main:app --reload --port 8000"

echo [3/3] Starting Frontend Web Server on http://127.0.0.1:5500...
start "QuantaRoute Frontend" cmd /k "call .\venv\Scripts\activate.bat && python -m http.server 5500 --directory frontend"

timeout /t 2 >nul
echo Launching Dashboard in default browser...
start http://127.0.0.1:5500/dashboard.html

echo.
echo =======================================================
echo   QuantaRoute is now RUNNING smoothly!
echo   - Backend API:  http://127.0.0.1:8000
echo   - Frontend:     http://127.0.0.1:5500/dashboard.html
echo =======================================================
