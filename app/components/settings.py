"""
Settings dialog — opened from the gear button at the bottom of the sidebar.
Persists user preferences to ~/Library/Application Support/CatLLM/settings.json.
"""

import json
import os
import platform
import requests
import streamlit as st
from components.key_store import load_keys, save_keys, clear_keys


def get_ollama_models(endpoint="http://localhost:11434"):
    """Query Ollama for downloaded models. Returns list of dicts with name and size."""
    try:
        resp = requests.get(f"{endpoint}/api/tags", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            return [
                {"name": m["name"], "size_gb": round(m["size"] / 1e9, 1)}
                for m in data.get("models", [])
            ]
    except Exception:
        pass
    return []


def _settings_dir():
    if platform.system() == "Darwin":
        return os.path.join(os.path.expanduser("~"), "Library", "Application Support", "CatLLM")
    return os.path.join(os.path.expanduser("~"), ".catllm")


def _settings_path():
    return os.path.join(_settings_dir(), "settings.json")


def load_settings():
    """Load saved settings from disk."""
    path = _settings_path()
    if not os.path.isfile(path):
        return _default_settings()
    try:
        with open(path, "r") as f:
            saved = json.load(f)
        defaults = _default_settings()
        defaults.update(saved)
        return defaults
    except Exception:
        return _default_settings()


def save_settings(settings):
    """Save settings dict to disk."""
    d = _settings_dir()
    os.makedirs(d, exist_ok=True)
    path = _settings_path()
    with open(path, "w") as f:
        json.dump(settings, f, indent=2)


def _default_settings():
    return {
        # Default model
        "default_model_tier": "Bring Your Own Key",
        "default_model": "gpt-4o",
        # Default parameters
        "default_creativity": 0.0,
        "default_batch_mode": False,
        "default_cot": False,
        "default_cove": False,
        # Output preferences
        "auto_save_results": True,
        "output_format": "CSV",
        "output_directory": os.path.expanduser("~/Downloads"),
        # Ollama
        "ollama_endpoint": "http://localhost:11434",
        # Theme
        "theme": "Default",
    }


def init_settings():
    """Load settings into session state on first run."""
    if "settings" not in st.session_state:
        st.session_state.settings = load_settings()


def get_setting(key, default=None):
    """Get a setting value."""
    settings = st.session_state.get("settings", {})
    return settings.get(key, default)


@st.dialog("Settings", width="large")
def show_settings_dialog():
    """Render the settings dialog with tabs."""
    settings = st.session_state.get("settings", _default_settings())

    tab_keys, tab_creds, tab_params, tab_output, tab_ollama, tab_about = st.tabs([
        "API Keys", "Platform Credentials", "Default Parameters", "Output", "Local Models", "About"
    ])

    # ---- API Keys ----
    with tab_keys:
        st.markdown("#### Cloud API Keys")
        st.caption("Keys are saved locally and never sent anywhere except to the provider.")

        api_keys = st.session_state.get("api_keys", {})
        providers = [
            ("openai", "OpenAI"),
            ("anthropic", "Anthropic"),
            ("google", "Google"),
            ("mistral", "Mistral"),
            ("xai", "xAI"),
            ("huggingface", "HuggingFace"),
            ("perplexity", "Perplexity"),
        ]

        for provider_id, display_name in providers:
            key = st.text_input(
                display_name,
                value=api_keys.get(provider_id, ""),
                type="password",
                key=f"settings_apikey_{provider_id}",
            )
            if key:
                api_keys[provider_id] = key
            elif provider_id in api_keys:
                del api_keys[provider_id]

        save_to_disk = st.checkbox(
            "Remember keys across sessions",
            value=True,
            key="settings_save_keys",
        )

    # ---- Platform Credentials ----
    with tab_creds:
        st.markdown("#### Social Media Platform Credentials")
        st.caption("Tokens and API keys for fetching data from social media platforms. Saved locally alongside your API keys.")

        st.markdown("##### Threads")
        threads_token = st.text_input(
            "Threads Access Token",
            value=api_keys.get("threads_token", ""),
            type="password",
            key="settings_threads_token",
        )
        if threads_token:
            api_keys["threads_token"] = threads_token
        elif "threads_token" in api_keys:
            del api_keys["threads_token"]

        st.markdown("##### Reddit")
        col1, col2 = st.columns(2)
        with col1:
            reddit_id = st.text_input(
                "Reddit Client ID",
                value=api_keys.get("reddit_client_id", ""),
                key="settings_reddit_id",
            )
            if reddit_id:
                api_keys["reddit_client_id"] = reddit_id
            elif "reddit_client_id" in api_keys:
                del api_keys["reddit_client_id"]
        with col2:
            reddit_secret = st.text_input(
                "Reddit Client Secret",
                value=api_keys.get("reddit_client_secret", ""),
                type="password",
                key="settings_reddit_secret",
            )
            if reddit_secret:
                api_keys["reddit_client_secret"] = reddit_secret
            elif "reddit_client_secret" in api_keys:
                del api_keys["reddit_client_secret"]

        st.markdown("##### Bluesky")
        bluesky_pw = st.text_input(
            "Bluesky App Password",
            value=api_keys.get("bluesky_app_password", ""),
            type="password",
            key="settings_bluesky_pw",
        )
        if bluesky_pw:
            api_keys["bluesky_app_password"] = bluesky_pw
        elif "bluesky_app_password" in api_keys:
            del api_keys["bluesky_app_password"]

        st.markdown("##### Mastodon")
        mastodon_token = st.text_input(
            "Mastodon Access Token",
            value=api_keys.get("mastodon_token", ""),
            type="password",
            key="settings_mastodon_token",
        )
        if mastodon_token:
            api_keys["mastodon_token"] = mastodon_token
        elif "mastodon_token" in api_keys:
            del api_keys["mastodon_token"]

        st.markdown("##### YouTube")
        youtube_key = st.text_input(
            "YouTube Data API Key",
            value=api_keys.get("youtube_api_key", ""),
            type="password",
            key="settings_youtube_key",
        )
        if youtube_key:
            api_keys["youtube_api_key"] = youtube_key
        elif "youtube_api_key" in api_keys:
            del api_keys["youtube_api_key"]

        st.markdown("---")
        st.markdown("#### Academic")
        openalex_email = st.text_input(
            "OpenAlex Polite Email",
            value=api_keys.get("openalex_email", ""),
            key="settings_openalex_email",
            help="Providing an email gives you access to OpenAlex's polite pool (faster rate limits).",
        )
        if openalex_email:
            api_keys["openalex_email"] = openalex_email
        elif "openalex_email" in api_keys:
            del api_keys["openalex_email"]

    # ---- Default Parameters ----
    with tab_params:
        st.markdown("#### Default Classification Parameters")

        from config import PAID_MODEL_CHOICES, FREE_MODEL_DISPLAY_NAMES

        settings["default_model_tier"] = st.radio(
            "Default Model Tier",
            options=["Free Models", "Bring Your Own Key"],
            index=0 if settings.get("default_model_tier") == "Free Models" else 1,
            horizontal=True,
            key="settings_default_tier",
        )

        if settings["default_model_tier"] == "Bring Your Own Key":
            model_options = PAID_MODEL_CHOICES
        else:
            model_options = FREE_MODEL_DISPLAY_NAMES

        current_model = settings.get("default_model", model_options[0])
        model_idx = model_options.index(current_model) if current_model in model_options else 0
        settings["default_model"] = st.selectbox(
            "Default Model",
            options=model_options,
            index=model_idx,
            key="settings_default_model",
        )

        settings["default_creativity"] = st.slider(
            "Default Creativity (Temperature)",
            min_value=0.0,
            max_value=2.0,
            value=float(settings.get("default_creativity", 0.0)),
            step=0.25,
            key="settings_default_creativity",
        )

        col1, col2 = st.columns(2)
        with col1:
            settings["default_cot"] = st.checkbox(
                "Chain-of-Thought by default",
                value=settings.get("default_cot", False),
                key="settings_default_cot",
            )
        with col2:
            settings["default_cove"] = st.checkbox(
                "CoVe by default",
                value=settings.get("default_cove", False),
                key="settings_default_cove",
            )

        settings["default_batch_mode"] = st.checkbox(
            "Batch mode by default",
            value=settings.get("default_batch_mode", False),
            key="settings_default_batch",
            help="Use async batch APIs for large datasets (cheaper, slower).",
        )

    # ---- Output ----
    with tab_output:
        st.markdown("#### Output Preferences")

        settings["auto_save_results"] = st.checkbox(
            "Auto-save results",
            value=settings.get("auto_save_results", True),
            key="settings_auto_save",
            help="Automatically save every run's results to disk. Browse past runs from the History section in the sidebar.",
        )

        settings["output_format"] = st.selectbox(
            "Default Export Format",
            options=["CSV", "Excel", "JSON"],
            index=["CSV", "Excel", "JSON"].index(settings.get("output_format", "CSV")),
            key="settings_output_format",
        )

        settings["output_directory"] = st.text_input(
            "Default Save Directory",
            value=settings.get("output_directory", os.path.expanduser("~/Downloads")),
            key="settings_output_dir",
        )

    # ---- Local Models ----
    with tab_ollama:
        st.markdown("#### Ollama Configuration")

        settings["ollama_endpoint"] = st.text_input(
            "Ollama Endpoint",
            value=settings.get("ollama_endpoint", "http://localhost:11434"),
            key="settings_ollama_endpoint",
        )

        # Show downloaded models
        endpoint = settings.get("ollama_endpoint", "http://localhost:11434")
        downloaded = get_ollama_models(endpoint)

        if downloaded:
            st.markdown("##### Downloaded Models")
            downloaded_names = {m["name"] for m in downloaded}
            for m in downloaded:
                st.markdown(f"- **{m['name']}** ({m['size_gb']} GB)")
        else:
            downloaded_names = set()
            st.info("No Ollama models found. Make sure Ollama is running, or check the endpoint above.")

        st.markdown("##### Suggested Models")

        # Build table with download status
        suggested = [
            ("Lower", "llama3.2:1b", "1.3 GB", "Fastest, for quick tests"),
            ("Lower", "llama3.2", "2.0 GB", "Good speed/accuracy tradeoff"),
            ("Lower", "phi3:mini", "2.2 GB", "Compact, good reasoning"),
            ("Middle", "llama3.1:8b", "4.7 GB", "General purpose"),
            ("Middle", "mistral", "4.1 GB", "Strong reasoning"),
            ("Middle", "qwen2.5:7b", "4.7 GB", "Good multilingual"),
            ("Upper", "gemma2:27b", "16 GB", "High accuracy"),
            ("Upper", "llama3.1:70b", "40 GB", "Best accuracy"),
        ]

        table_rows = []
        for tier, name, size, notes in suggested:
            status = "Downloaded" if name in downloaded_names else "—"
            table_rows.append(f"| {tier} | `{name}` | {size} | {notes} | {status} |")

        table = "| Tier | Model | Size | Notes | Status |\n|------|-------|------|-------|--------|\n"
        table += "\n".join(table_rows)
        st.markdown(table)

        st.caption("Install models with: `ollama pull <model-name>`")

    # ---- About ----
    with tab_about:
        st.markdown("#### About CatLLM")
        st.markdown("""
**CatLLM** — LLM-powered text classification for social science.

- **Version:** 3.0.0
- **License:** GPL-3.0
- **GitHub:** github.com/chrissoria/cat-llm
- **Docs:** catllm.org

Part of the CatLLM ecosystem:
`cat-stack` · `cat-survey` · `cat-vader` · `cat-ademic` · `cat-cog` · `cat-search`
""")

    # ---- Save button ----
    st.markdown("---")
    col_save, col_cancel = st.columns(2)
    with col_save:
        if st.button("Save", type="primary", use_container_width=True):
            st.session_state.settings = settings
            save_settings(settings)
            st.session_state.api_keys = api_keys
            if save_to_disk:
                save_keys(api_keys)
            else:
                clear_keys()
            st.rerun()
    with col_cancel:
        if st.button("Cancel", use_container_width=True):
            st.rerun()
