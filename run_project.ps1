# QuantaRoute ENTERPRISE PowerShell Launcher

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "  QuantaRoute ENTERPRISE - Starting Platform" -ForegroundColor White
Write-Host "=======================================================" -ForegroundColor Cyan

Set-Location $PSScriptRoot

# 1. Clean up ports 8000 and 5500 if occupied
Write-Host "[1/3] Checking for any processes occupying ports 8000 & 5500..." -ForegroundColor Yellow
$pids = Get-NetTCPConnection -LocalPort 8000, 5500 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
if ($pids) {
    foreach ($p in $pids) {
        try {
            Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
            Write-Host "  Cleaned process PID $p" -ForegroundColor DarkGray
        } catch {}
    }
}

# 2. Start Uvicorn Backend
Write-Host "[2/3] Starting Backend API on http://127.0.0.1:8000..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot'; .\venv\Scripts\Activate.ps1; uvicorn app.main:app --reload --port 8000"

# 3. Start Frontend
Write-Host "[3/3] Starting Frontend Web Server on http://127.0.0.1:5500..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot'; .\venv\Scripts\Activate.ps1; python -m http.server 5500 --directory frontend"

Start-Sleep -Seconds 2
Write-Host "Opening Dashboard in default browser..." -ForegroundColor Cyan
Start-Process "http://127.0.0.1:5500/dashboard.html"

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "  QuantaRoute is now RUNNING smoothly!" -ForegroundColor Green
Write-Host "  - Backend:  http://127.0.0.1:8000" -ForegroundColor White
Write-Host "  - Frontend: http://127.0.0.1:5500/dashboard.html" -ForegroundColor White
Write-Host "=======================================================" -ForegroundColor Cyan
