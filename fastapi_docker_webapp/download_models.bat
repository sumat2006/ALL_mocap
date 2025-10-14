@echo off
REM Script to download required HuggingFace models for Windows
REM Usage: download_models.bat

echo ================================================================
echo       Downloading HuggingFace Models for FastAPI App
echo ================================================================
echo.

REM Check if huggingface-cli is installed
where huggingface-cli >nul 2>nul
if %errorlevel% neq 0 (
    echo WARNING: huggingface-cli not found!
    echo Installing huggingface_hub...
    pip install huggingface_hub
)

REM Create model directory
set MODEL_DIR=.\app\asset\model
if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"

echo.
echo ----------------------------------------------------------------
echo Downloading KhanomTanLLM-1B (~2.6 GB)
echo ----------------------------------------------------------------

REM Download KhanomTanLLM-1B
if exist "%MODEL_DIR%\KhanomTanLLM-1B" (
    echo KhanomTanLLM-1B already exists, skipping...
) else (
    huggingface-cli download --local-dir "%MODEL_DIR%\KhanomTanLLM-1B" --local-dir-use-symlinks False KhanomTan/KhanomTanLLM-1B
    echo KhanomTanLLM-1B downloaded successfully!
)

echo.
echo ----------------------------------------------------------------
echo Downloading thonburain-whisper (~1.5 GB)
echo ----------------------------------------------------------------

REM Download thonburain-whisper
if exist "%MODEL_DIR%\thonburain-whisper" (
    echo thonburain-whisper already exists, skipping...
) else (
    huggingface-cli download --local-dir "%MODEL_DIR%\thonburain-whisper" --local-dir-use-symlinks False biodatlab/thonburian-whisper-th-en-large-v3
    echo thonburian-whisper downloaded successfully!
)

echo.
echo ================================================================
echo All models downloaded successfully!
echo ================================================================
echo.
echo Total size: ~4.1 GB
echo.
echo Models location:
echo   - %MODEL_DIR%\KhanomTanLLM-1B\
echo   - %MODEL_DIR%\thonburain-whisper\
echo.
echo You can now run: docker-compose up
echo.

pause
