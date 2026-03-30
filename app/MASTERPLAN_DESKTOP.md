# CatLLM Desktop App — Mac Packaging Master Plan

## Goal

Package the unified Streamlit app (`app/`) into a downloadable `.dmg` installer for macOS that users can double-click to install and run without needing Python, pip, or any terminal commands.

---

## Architecture

```
CatLLM.app (macOS .app bundle)
  └── launches a local Streamlit server
      └── opens the user's default browser to http://localhost:8501
```

The app is a **thin native wrapper** around the Streamlit server. When the user opens CatLLM.app, it:
1. Starts a bundled Python + Streamlit process in the background
2. Opens `http://localhost:8501` in the default browser
3. Shows a menu bar icon / dock icon indicating the server is running
4. Stops the server when the user quits the app

---

## Approach: PyInstaller + Streamlit

### Why PyInstaller
- Most mature Python packaging tool for macOS
- Can bundle Python interpreter + all dependencies into a single `.app`
- Handles C extensions (PyMuPDF, numpy, etc.)
- Produces a standalone `.app` that doesn't require system Python

### Alternatives Considered
| Tool | Pros | Cons |
|------|------|------|
| **PyInstaller** | Mature, well-documented, handles C extensions | Large bundle size (~500MB+) |
| **PyApp** | Lighter, Rust-based | Less mature, tricky with Streamlit |
| **Briefcase (BeeWare)** | Native app packaging | Poor Streamlit support |
| **Nuitka** | Compiles to C, fast | Complex setup, Streamlit compatibility unknown |
| **Electron + Python** | Native-feeling app | Heavy, adds complexity |
| **Docker** | Reliable | Requires Docker Desktop installed |

**Recommendation**: PyInstaller is the proven path for Streamlit apps.

---

## Implementation Phases

### Phase 1: PyInstaller Bundle (Core)

**Goal**: Create a working `.app` bundle that runs the Streamlit app.

1. **Create launcher script** (`app/desktop/launcher.py`)
   ```python
   """Launch CatLLM Streamlit app and open browser."""
   import subprocess
   import sys
   import os
   import time
   import webbrowser

   def main():
       app_dir = os.path.dirname(os.path.abspath(__file__))
       main_py = os.path.join(app_dir, "main.py")

       # Start Streamlit server
       proc = subprocess.Popen([
           sys.executable, "-m", "streamlit", "run", main_py,
           "--server.headless", "true",
           "--server.port", "8501",
           "--browser.gatherUsageStats", "false",
       ])

       # Wait for server to start, then open browser
       time.sleep(3)
       webbrowser.open("http://localhost:8501")

       # Keep running until process exits
       proc.wait()

   if __name__ == "__main__":
       main()
   ```

2. **Create PyInstaller spec file** (`app/desktop/catllm.spec`)
   - Bundle Python interpreter
   - Include all `app/` source files as data
   - Include `assets/` directory
   - Include `.streamlit/config.toml`
   - Hidden imports for all catllm sub-packages

3. **Build command**:
   ```bash
   pyinstaller catllm.spec --noconfirm
   ```

4. **Test the `.app` bundle**

### Phase 2: API Key Persistence

**Goal**: Store API keys on disk so users don't re-enter them each session.

1. **Use macOS Keychain** via `keyring` package
   - Or simpler: encrypted `.env` file in `~/Library/Application Support/CatLLM/`
2. **First-run setup wizard**: on first launch, show a page that asks for API keys
3. **Settings page**: accessible from sidebar, allows updating keys

### Phase 3: DMG Installer

**Goal**: Create a polished `.dmg` with drag-to-Applications install.

1. **Use `create-dmg`** (npm package or shell tool):
   ```bash
   create-dmg \
     --volname "CatLLM" \
     --volicon "app/assets/logo.icns" \
     --window-pos 200 120 \
     --window-size 600 400 \
     --icon-size 100 \
     --icon "CatLLM.app" 175 120 \
     --hide-extension "CatLLM.app" \
     --app-drop-link 425 120 \
     "CatLLM-Installer.dmg" \
     "dist/CatLLM.app"
   ```

2. **App icon**: Convert `logo.png` to `.icns` format for the macOS dock icon

### Phase 4: Auto-Update (Optional)

1. **Sparkle framework** or simple version-check against GitHub releases
2. On launch, check `https://api.github.com/repos/chrissoria/cat-llm/releases/latest`
3. If newer version available, prompt user to download

### Phase 5: Code Signing & Notarization (Distribution)

**Required for users to open the app without Gatekeeper warnings.**

1. **Apple Developer Account** ($99/year)
2. **Code sign** the `.app` bundle:
   ```bash
   codesign --deep --force --sign "Developer ID Application: ..." CatLLM.app
   ```
3. **Notarize** with Apple:
   ```bash
   xcrun notarytool submit CatLLM.dmg --apple-id ... --team-id ...
   ```
4. **Staple** the notarization ticket:
   ```bash
   xcrun stapler staple CatLLM.dmg
   ```

Without notarization, users must right-click → Open → "Open Anyway" on first launch.

---

## Key Technical Challenges

### 1. Bundle Size
- PyInstaller + Streamlit + catllm + PyMuPDF = ~500-800MB
- **Mitigation**: Use `--exclude-module` to strip unused packages, use UPX compression

### 2. C Extensions
- PyMuPDF (fitz), numpy, matplotlib all have C extensions
- **Mitigation**: PyInstaller handles these on macOS; test on both Intel and Apple Silicon

### 3. Apple Silicon vs Intel
- Need to build **two bundles** or a **universal binary**
- **Mitigation**: Build on Apple Silicon (supports Rosetta for Intel), or use GitHub Actions with matrix builds

### 4. Streamlit Server Lifecycle
- Need to cleanly start/stop the Streamlit process
- **Mitigation**: Launcher script with signal handling; menu bar icon with "Quit" option

### 5. Port Conflicts
- Port 8501 might be in use
- **Mitigation**: Try ports 8501-8510, use first available

---

## File Structure

```
app/
  desktop/
    launcher.py          # Main entry point for desktop app
    catllm.spec          # PyInstaller spec file
    build.sh             # Build script (pyinstaller + create-dmg)
    Info.plist           # macOS app metadata
    icon.icns            # App icon (converted from logo.png)
  ... (existing app files)
```

---

## Build Pipeline (CI)

```yaml
# .github/workflows/build-desktop.yml
name: Build Desktop App
on:
  release:
    types: [published]
jobs:
  build-macos:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -e ".[pdf]" pyinstaller
      - run: cd app/desktop && pyinstaller catllm.spec --noconfirm
      - run: cd app/desktop && ./build-dmg.sh
      - uses: actions/upload-artifact@v4
        with:
          name: CatLLM-macOS
          path: app/desktop/dist/CatLLM-Installer.dmg
```

---

## Timeline Estimate

| Phase | Effort | Description |
|-------|--------|-------------|
| Phase 1 | 1-2 days | PyInstaller bundle, launcher script, basic .app |
| Phase 2 | 0.5 day | API key persistence |
| Phase 3 | 0.5 day | DMG installer with drag-to-install |
| Phase 4 | 1 day | Auto-update (optional) |
| Phase 5 | 1 day | Code signing & notarization (requires Apple Developer account) |

**Minimum viable desktop app (Phases 1-3): ~2-3 days**

---

## Prerequisites

- [ ] PyInstaller: `pip install pyinstaller`
- [ ] create-dmg: `brew install create-dmg` or `npm install -g create-dmg`
- [ ] Apple Developer account (for Phase 5 only)
- [ ] Test machines: Apple Silicon Mac + Intel Mac (or Rosetta)
