$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$backend = Join-Path $root "backend"
$frontend = Join-Path $root "frontend"

Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$backend'; python -m uvicorn main:app --host 127.0.0.1 --port 8000"
)
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$frontend'; npm run dev -- --host 127.0.0.1 --port 5173"
)

Write-Host "Arayuz: http://127.0.0.1:5173"
Write-Host "API:    http://127.0.0.1:8000/api/health"
