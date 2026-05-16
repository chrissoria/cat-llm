#!/usr/bin/env bash
# Upload the latest DMG + checksum + README to the HuggingFace model repo.
# Usage: bash app/desktop/upload_to_hf.sh
#
# Requires `huggingface-cli login` (or HF_TOKEN env) one-time, and write
# access to chrissoria/catllm-desktop.

set -euo pipefail

REPO="chrissoria/catllm-desktop"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIST_DIR="$SCRIPT_DIR/dist"

if ! command -v hf &>/dev/null && ! command -v huggingface-cli &>/dev/null; then
    echo "ERROR: huggingface_hub CLI not found. Install with: pip install huggingface_hub" >&2
    exit 1
fi

# Prefer modern `hf` CLI, fall back to deprecated `huggingface-cli`.
if command -v hf &>/dev/null; then
    HF="hf"
else
    HF="huggingface-cli"
fi

# Make sure the repo exists (idempotent — succeeds if already there).
$HF repo create "$REPO" --type model --exist-ok 2>/dev/null \
    || $HF repo create "$REPO" --type model 2>/dev/null \
    || true

# Upload every DMG + sha256 in dist/.
shopt -s nullglob
for f in "$DIST_DIR"/CatLLM-*.dmg "$DIST_DIR"/CatLLM-*.dmg.sha256; do
    name="$(basename "$f")"
    echo "Uploading $name -> $REPO ..."
    $HF upload "$REPO" "$f" "$name"
done

# Upload the README (renamed to README.md on the hub).
if [ -f "$SCRIPT_DIR/HF_README.md" ]; then
    echo "Uploading README.md -> $REPO ..."
    $HF upload "$REPO" "$SCRIPT_DIR/HF_README.md" "README.md"
fi

echo ""
echo "Done. View at: https://huggingface.co/$REPO"
