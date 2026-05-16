#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$(dirname "$SCRIPT_DIR")"
ROOT_DIR="$(dirname "$APP_DIR")"
VERSION="$(grep -oE '__version__\s*=\s*"[^"]+"' "$ROOT_DIR/src/catllm/__about__.py" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')"
ARCH="$(uname -m)"   # arm64 (Apple Silicon) or x86_64 (Intel)
DMG_NAME="CatLLM-${VERSION}-${ARCH}.dmg"

echo "=== CatLLM Desktop Build ==="
echo "Version:  $VERSION"
echo "Arch:     $ARCH"
echo "App dir:  $APP_DIR"
echo "Root dir: $ROOT_DIR"

# Step 1: Install dependencies
echo ""
echo "--- Installing dependencies ---"
pip install pyinstaller
pip install -e "$ROOT_DIR[pdf]"
pip install -r "$APP_DIR/requirements.txt"

# Step 2: Clean previous output and build
echo ""
echo "--- Cleaning previous build artifacts ---"
rm -rf "$SCRIPT_DIR/build" "$SCRIPT_DIR/dist"

echo ""
echo "--- Building .app bundle ---"
cd "$SCRIPT_DIR"
pyinstaller catllm.spec --noconfirm

APP_BUNDLE="$SCRIPT_DIR/dist/CatLLM.app"
echo ""
echo "--- Build complete ---"
echo "App bundle: $APP_BUNDLE"

# Step 3: Ad-hoc codesign — won't satisfy Apple notarization, but stops the
# "app is damaged and can't be opened" error that fully-unsigned bundles hit
# on Apple Silicon.
echo ""
echo "--- Ad-hoc codesigning ---"
codesign --force --deep --sign - "$APP_BUNDLE"
codesign --verify --verbose "$APP_BUNDLE" || true

# Step 4: Create DMG (if create-dmg is installed)
if command -v create-dmg &>/dev/null; then
    echo ""
    echo "--- Creating DMG installer ---"

    rm -f "$SCRIPT_DIR/dist/$DMG_NAME"

    create-dmg \
        --volname "CatLLM ${VERSION}" \
        --window-pos 200 120 \
        --window-size 600 400 \
        --icon-size 100 \
        --icon "CatLLM.app" 175 120 \
        --hide-extension "CatLLM.app" \
        --app-drop-link 425 120 \
        "$SCRIPT_DIR/dist/$DMG_NAME" \
        "$APP_BUNDLE"

    echo "DMG: $SCRIPT_DIR/dist/$DMG_NAME"
    du -sh "$SCRIPT_DIR/dist/$DMG_NAME"

    # Step 5: Generate SHA-256 checksum so users can verify their download.
    echo ""
    echo "--- Generating SHA-256 checksum ---"
    cd "$SCRIPT_DIR/dist"
    shasum -a 256 "$DMG_NAME" > "${DMG_NAME}.sha256"
    cat "${DMG_NAME}.sha256"
    cd - >/dev/null
else
    echo ""
    echo "Skipping DMG creation (install create-dmg: brew install create-dmg)"
fi
