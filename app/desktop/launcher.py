"""Launch CatLLM Streamlit app in a native window."""

import sys
import os
import socket
import subprocess
import threading
import time


def find_free_port(start=8501, end=8510):
    """Find first available port in range."""
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start


def _wait_for_server(port, timeout=30):
    """Block until the Streamlit server is accepting connections."""
    for _ in range(timeout * 2):
        time.sleep(0.5)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
    return False


def _start_streamlit_frozen(main_py, port, streamlit_config, app_dir):
    """Start Streamlit in-process (for PyInstaller bundles where subprocess won't work)."""
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_SERVER_PORT"] = str(port)
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"
    os.environ["STREAMLIT_BROWSER_SERVER_ADDRESS"] = "localhost"
    if os.path.isdir(streamlit_config):
        os.environ["STREAMLIT_CONFIG_DIR"] = streamlit_config
    sys.path.insert(0, app_dir)

    import streamlit.config as _cfg
    _cfg.set_option("global.developmentMode", False)

    from streamlit.web import bootstrap

    # Monkey-patch signal setup to no-op (can't set signals from a thread)
    bootstrap._set_up_signal_handler = lambda server: None

    bootstrap.run(main_py, False, [], {})


def _start_streamlit_subprocess(main_py, port, streamlit_config):
    """Start Streamlit as a subprocess (for running from source)."""
    env = os.environ.copy()
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    env["STREAMLIT_SERVER_PORT"] = str(port)
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    env["STREAMLIT_BROWSER_SERVER_ADDRESS"] = "localhost"
    if os.path.isdir(streamlit_config):
        env["STREAMLIT_CONFIG_DIR"] = streamlit_config

    return subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", main_py,
         "--server.headless", "true",
         "--server.port", str(port),
         "--browser.gatherUsageStats", "false"],
        env=env,
    )


def main():
    frozen = getattr(sys, "frozen", False)

    if frozen:
        bundle_dir = os.path.dirname(sys.executable)
        frameworks_dir = os.path.join(os.path.dirname(bundle_dir), "Frameworks")
        app_dir = frameworks_dir
    else:
        desktop_dir = os.path.dirname(os.path.abspath(__file__))
        app_dir = os.path.dirname(desktop_dir)

    main_py = os.path.join(app_dir, "main.py")
    streamlit_config = os.path.join(app_dir, ".streamlit")

    port = find_free_port()
    url = f"http://localhost:{port}"

    proc = None
    if frozen:
        # In-process Streamlit in a daemon thread
        t = threading.Thread(
            target=_start_streamlit_frozen,
            args=(main_py, port, streamlit_config, app_dir),
            daemon=True,
        )
        t.start()
    else:
        proc = _start_streamlit_subprocess(main_py, port, streamlit_config)

    if not _wait_for_server(port):
        print("ERROR: Streamlit server did not start", file=sys.stderr)
        if proc:
            proc.terminate()
        sys.exit(1)

    import webview

    def on_closed():
        if proc:
            proc.terminate()
            proc.wait(timeout=5)

    window = webview.create_window(
        "CatLLM",
        url,
        width=1280,
        height=850,
        min_size=(900, 600),
    )
    window.events.closed += on_closed
    webview.start()


if __name__ == "__main__":
    main()
