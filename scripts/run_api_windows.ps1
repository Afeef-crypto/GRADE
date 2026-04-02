# Start the GRADE API on Windows (loads repo-root `.env` via python-dotenv in autograder.api).
# Usage: .\scripts\run_api_windows.ps1
# From repo root: install once with  pip install -e ".[api,ocr_google]"

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

if (-not (Test-Path ".env")) {
    Write-Host "No .env in repo root. Copy .env.example to .env and set GOOGLE_APPLICATION_CREDENTIALS." -ForegroundColor Yellow
}

$env:PYTHONPATH = "src"
python -m uvicorn autograder.api:app --host 127.0.0.1 --port 8000
