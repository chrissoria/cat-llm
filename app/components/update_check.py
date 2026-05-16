"""
Update check.

On first render of a session, query PyPI for the latest cat-llm release and
compare to the bundled version. If a newer release exists, set a flag in
session state so the UI can show a discreet banner.

Fully best-effort: any network error, parse error, or version-comparison
issue is silently ignored — we never want this to block startup.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

import streamlit as st


PYPI_URL = "https://pypi.org/pypi/cat-llm/json"
DOWNLOAD_URL = "https://huggingface.co/chrissoria/catllm-desktop"


def _installed_version() -> Optional[str]:
    try:
        from catllm.__about__ import __version__  # type: ignore
        return __version__
    except Exception:
        try:
            from importlib.metadata import version
            return version("cat-llm")
        except Exception:
            return None


def _parse(v: str) -> Tuple[int, ...]:
    """Tolerant semver-ish parse — only the leading numeric components."""
    nums = re.findall(r"\d+", v)
    return tuple(int(n) for n in nums[:3])


def check_for_updates() -> None:
    """Populate st.session_state.update_available if a newer release exists."""
    if "update_checked" in st.session_state:
        return
    st.session_state.update_checked = True
    st.session_state.update_available = None

    installed = _installed_version()
    if not installed:
        return

    try:
        import requests
        resp = requests.get(PYPI_URL, timeout=2.5)
        if resp.status_code != 200:
            return
        latest = resp.json().get("info", {}).get("version")
        if not latest:
            return
        if _parse(latest) > _parse(installed):
            st.session_state.update_available = {
                "installed": installed,
                "latest": latest,
                "url": DOWNLOAD_URL,
            }
    except Exception:
        pass


def render_update_banner() -> None:
    """Render a small banner if an update is available. No-op otherwise."""
    info = st.session_state.get("update_available")
    if not info:
        return
    st.markdown(
        f"""
        <div style="
            background: rgba(255,140,60,0.08);
            border: 1px solid #D87024;
            border-radius: 8px;
            padding: 0.6rem 0.9rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            font-family: 'DM Sans', sans-serif;
            font-size: 0.9rem;
            color: #E8EDF5;
        ">
          <span>
            CatLLM <b>{info['latest']}</b> is available
            <span style="color:#8A95A8;">(you have {info['installed']})</span>.
          </span>
          <a href="{info['url']}" target="_blank" style="
              color: #00DCC3;
              text-decoration: none;
              font-weight: 500;
              border: 1px solid #00A89A;
              padding: 0.35rem 0.8rem;
              border-radius: 6px;
              white-space: nowrap;
          ">Download →</a>
        </div>
        """,
        unsafe_allow_html=True,
    )
