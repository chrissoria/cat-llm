"""
Persistent API key storage using ~/Library/Application Support/CatLLM/ on macOS,
with fallback to ~/.catllm/ on other platforms.

Keys are stored in a JSON file with base64 obfuscation (not encryption — this
deters casual snooping but is not a security boundary; the keys are on disk).
"""

import base64
import json
import os
import platform


def _storage_dir():
    """Return the directory for storing CatLLM config files."""
    if platform.system() == "Darwin":
        return os.path.join(os.path.expanduser("~"), "Library", "Application Support", "CatLLM")
    return os.path.join(os.path.expanduser("~"), ".catllm")


def _keys_path():
    return os.path.join(_storage_dir(), "api_keys.json")


def _encode(value):
    return base64.b64encode(value.encode()).decode()


def _decode(value):
    try:
        return base64.b64decode(value.encode()).decode()
    except Exception:
        return value


def load_keys():
    """Load saved API keys from disk. Returns dict of provider -> key."""
    path = _keys_path()
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return {k: _decode(v) for k, v in data.items() if v}
    except Exception:
        return {}


def save_keys(keys_dict):
    """Save API keys dict to disk. Empty values are omitted."""
    d = _storage_dir()
    os.makedirs(d, exist_ok=True)
    filtered = {k: _encode(v) for k, v in keys_dict.items() if v}
    path = _keys_path()
    with open(path, "w") as f:
        json.dump(filtered, f, indent=2)
    # Restrict permissions to owner only
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def clear_keys():
    """Delete saved API keys from disk."""
    path = _keys_path()
    if os.path.isfile(path):
        os.remove(path)
