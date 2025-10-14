#!/bin/bash
# Script to download required HuggingFace models
# Usage: bash download_models.sh

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║       Downloading HuggingFace Models for FastAPI App          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "⚠️  huggingface-cli not found!"
    echo "Installing huggingface_hub..."
    pip install huggingface_hub
fi

# Create model directory
MODEL_DIR="./app/asset/model"
mkdir -p "$MODEL_DIR"

echo ""
echo "────────────────────────────────────────────────────────────────"
echo "📥 Downloading KhanomTanLLM-1B (~2.6 GB)"
echo "────────────────────────────────────────────────────────────────"

# Download KhanomTanLLM-1B
if [ -d "$MODEL_DIR/KhanomTanLLM-1B" ]; then
    echo "✅ KhanomTanLLM-1B already exists, skipping..."
else
    huggingface-cli download \
        --local-dir "$MODEL_DIR/KhanomTanLLM-1B" \
        --local-dir-use-symlinks False \
        KhanomTan/KhanomTanLLM-1B
    echo "✅ KhanomTanLLM-1B downloaded successfully!"
fi

echo ""
echo "────────────────────────────────────────────────────────────────"
echo "📥 Downloading thonburain-whisper (~1.5 GB)"
echo "────────────────────────────────────────────────────────────────"

# Download thonburain-whisper
if [ -d "$MODEL_DIR/thonburain-whisper" ]; then
    echo "✅ thonburain-whisper already exists, skipping..."
else
    huggingface-cli download \
        --local-dir "$MODEL_DIR/thonburain-whisper" \
        --local-dir-use-symlinks False \
        biodatlab/thonburian-whisper-th-en-large-v3
    echo "✅ thonburian-whisper downloaded successfully!"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✅ All models downloaded successfully!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Total size: ~4.1 GB"
echo ""
echo "Models location:"
echo "  - $MODEL_DIR/KhanomTanLLM-1B/"
echo "  - $MODEL_DIR/thonburain-whisper/"
echo ""
echo "You can now run: docker-compose up"
echo ""
