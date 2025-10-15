# PowerShell Script to download required HuggingFace models
# Usage: .\download_models.ps1

# Enable strict error handling
$ErrorActionPreference = "Stop"

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "       Downloading HuggingFace Models for FastAPI App" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if huggingface-cli is installed
try {
    $null = Get-Command huggingface-cli -ErrorAction Stop
    Write-Host "✓ huggingface-cli found" -ForegroundColor Green
} catch {
    Write-Host "⚠️  huggingface-cli not found!" -ForegroundColor Yellow
    Write-Host "Installing huggingface_hub..." -ForegroundColor Yellow
    pip install huggingface_hub
}

# Create model directory
$MODEL_DIR = ".\app\asset\model"
if (-not (Test-Path $MODEL_DIR)) {
    New-Item -ItemType Directory -Path $MODEL_DIR -Force | Out-Null
}

Write-Host ""
Write-Host "----------------------------------------------------------------" -ForegroundColor Gray
Write-Host "📥 Downloading KhanomTanLLM-1B (~2.6 GB)" -ForegroundColor Yellow
Write-Host "----------------------------------------------------------------" -ForegroundColor Gray

# Download KhanomTanLLM-1B
$khanomPath = Join-Path $MODEL_DIR "KhanomTanLLM-1B"
if (Test-Path $khanomPath) {
    Write-Host "✅ KhanomTanLLM-1B already exists, skipping..." -ForegroundColor Green
} else {
    huggingface-cli download `
        --local-dir "$khanomPath" `
        --local-dir-use-symlinks False `
        KhanomTan/KhanomTanLLM-1B
    Write-Host "✅ KhanomTanLLM-1B downloaded successfully!" -ForegroundColor Green
}

Write-Host ""
Write-Host "----------------------------------------------------------------" -ForegroundColor Gray
Write-Host "📥 Downloading thonburain-whisper (~1.5 GB)" -ForegroundColor Yellow
Write-Host "----------------------------------------------------------------" -ForegroundColor Gray

# Download thonburain-whisper
$whisperPath = Join-Path $MODEL_DIR "thonburain-whisper"
if (Test-Path $whisperPath) {
    Write-Host "✅ thonburain-whisper already exists, skipping..." -ForegroundColor Green
} else {
    huggingface-cli download `
        --local-dir "$whisperPath" `
        --local-dir-use-symlinks False `
        biodatlab/thonburian-whisper-th-en-large-v3
    Write-Host "✅ thonburian-whisper downloaded successfully!" -ForegroundColor Green
}

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "✅ All models downloaded successfully!" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Total size: ~4.1 GB"
Write-Host ""
Write-Host "Models location:"
Write-Host "  - $khanomPath"
Write-Host "  - $whisperPath"
Write-Host ""
Write-Host "You can now run: docker-compose up"
Write-Host ""

# Pause if running interactively
if ($Host.UI.RawUI.KeyAvailable -eq $false) {
    Read-Host "Press Enter to exit"
}
