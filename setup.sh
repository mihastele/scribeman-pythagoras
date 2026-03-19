#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# setup.sh  —  Linux / macOS dependency installer
# Run once: bash setup.sh
#
# For Windows: run setup.bat instead
# ──────────────────────────────────────────────────────────────
set -euo pipefail

OS="$(uname -s)"

if [[ "$OS" == "Linux" ]]; then
    echo "==> Installing system audio libraries (Linux)…"
    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends \
        portaudio19-dev \
        libsndfile1 \
        ffmpeg \
        pulseaudio-utils
elif [[ "$OS" == "Darwin" ]]; then
    echo "==> Installing system audio libraries (macOS)…"
    if ! command -v brew &>/dev/null; then
        echo "Homebrew not found. Install it from https://brew.sh first."
        exit 1
    fi
    brew install portaudio ffmpeg
    echo ""
    echo "NOTE: For system audio capture on macOS you also need BlackHole:"
    echo "  https://existential.audio/blackhole/"
    echo "  After installing, create a Multi-Output Device in Audio MIDI Setup."
fi

echo ""
echo "==> Installing Python dependencies…"
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "==> Done."
echo "    Run: python transcribe.py --list-devices"
