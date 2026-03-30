#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$(dirname "$SCRIPT_DIR")"
ROOT_DIR="$(dirname "$APP_DIR")"

echo "=== CatLLM Desktop Build ==="
echo "App dir:  $APP_DIR"
echo "Root dir: $ROOT_DIR"

# Step 1: Install dependencies
echo ""
echo "--- Installing dependencies ---"
pip install pyinstaller
pip install -e "$ROOT_DIR[pdf]"
pip install -r "$APP_DIR/requirements.txt"

# Step 2: Build with PyInstaller
echo ""
echo "--- Building .app bundle ---"
cd "$SCRIPT_DIR"
pyinstaller catllm.spec --noconfirm

echo ""
echo "--- Build complete ---"
echo "App bundle: $SCRIPT_DIR/dist/CatLLM.app"

# Step 3: Create DMG (if create-dmg is installed)
if command -v create-dmg &>/dev/null; then
    echo ""
    echo "--- Creating DMG installer ---"

    # Remove old DMG if it exists
    rm -f "$SCRIPT_DIR/dist/CatLLM-Installer.dmg"

    create-dmg \
        --volname "CatLLM" \
        --window-pos 200 120 \
        --window-size 600 400 \
        --icon-size 100 \
        --icon "CatLLM.app" 175 120 \
        --hide-extension "CatLLM.app" \
        --app-drop-link 425 120 \
        "$SCRIPT_DIR/dist/CatLLM-Installer.dmg" \
        "$SCRIPT_DIR/dist/CatLLM.app"

    echo "DMG: $SCRIPT_DIR/dist/CatLLM-Installer.dmg"
else
    echo ""
    echo "Skipping DMG creation (install create-dmg: brew install create-dmg)"
fi
