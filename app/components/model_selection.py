"""
Model selection UI: free/paid tier, single/comparison/ensemble modes + API key input.
"""

import streamlit as st
from config import (
    MODEL_CHOICES, PAID_MODEL_CHOICES, FREE_MODELS_MAP, FREE_MODEL_DISPLAY_NAMES,
    HF_ROUTED_MODELS, resolve_api_key, get_model_source,
)
from components.key_store import load_keys, save_keys, clear_keys
from components.settings import get_ollama_models


def render_api_keys_sidebar():
    """Render API key input fields in the sidebar under collapsible sections."""
    st.sidebar.markdown("### Model Setup")

    # Load saved keys on first run
    if "api_keys" not in st.session_state:
        st.session_state.api_keys = load_keys()

    api_keys = st.session_state.get("api_keys", {})

    # --- Local Models (Ollama) ---
    with st.sidebar.expander("Local Models (Ollama)"):
        ollama_endpoint = st.text_input(
            "Ollama Endpoint",
            value=api_keys.get("ollama_endpoint", "http://localhost:11434"),
            key="ollama_endpoint_input",
            help="URL where your Ollama server is running.",
        )
        api_keys["ollama_endpoint"] = ollama_endpoint

        # Query Ollama for downloaded models
        downloaded_models = get_ollama_models(ollama_endpoint)
        downloaded_names = {m["name"] for m in downloaded_models}

        def _label(name, size, downloaded_names):
            check = " \u2713" if name in downloaded_names else ""
            return f"{name} ({size}){check}"

        # Suggested models organized by tier, with checkmarks for downloaded
        local_model_options = {
            "— Lower Tier (fast, smaller) —": None,
            _label("llama3.2:1b", "1.3 GB", downloaded_names): "llama3.2:1b",
            _label("llama3.2", "2.0 GB", downloaded_names): "llama3.2",
            _label("phi3:mini", "2.2 GB", downloaded_names): "phi3:mini",
            _label("qwen2.5:1.5b", "1.0 GB", downloaded_names): "qwen2.5:1.5b",
            _label("gemma2:2b", "1.6 GB", downloaded_names): "gemma2:2b",
            "— Middle Tier (balanced) —": None,
            _label("llama3.1:8b", "4.7 GB", downloaded_names): "llama3.1:8b",
            _label("mistral", "4.1 GB", downloaded_names): "mistral",
            _label("qwen2.5:7b", "4.7 GB", downloaded_names): "qwen2.5:7b",
            _label("gemma2:9b", "5.4 GB", downloaded_names): "gemma2:9b",
            _label("deepseek-r1", "4.7 GB", downloaded_names): "deepseek-r1",
            "— Upper Tier (best accuracy) —": None,
            _label("gemma2:27b", "16 GB", downloaded_names): "gemma2:27b",
            _label("mixtral", "26 GB", downloaded_names): "mixtral",
            _label("llama3.1:70b", "40 GB", downloaded_names): "llama3.1:70b",
            "— Other —": None,
            "Other (enter manually)": "other",
        }

        # Also add any downloaded models not in the suggested list
        suggested_names = {
            "llama3.2:1b", "llama3.2", "phi3:mini", "qwen2.5:1.5b", "gemma2:2b",
            "llama3.1:8b", "mistral", "qwen2.5:7b", "gemma2:9b", "deepseek-r1",
            "gemma2:27b", "mixtral", "llama3.1:70b",
        }
        extra_downloaded = [m for m in downloaded_models if m["name"] not in suggested_names]
        if extra_downloaded:
            extra_options = {"— Downloaded (other) —": None}
            for m in extra_downloaded:
                extra_options[f"{m['name']} ({m['size_gb']} GB) \u2713"] = m["name"]
            # Insert before "— Other —"
            items = list(local_model_options.items())
            other_idx = next(i for i, (k, _) in enumerate(items) if k == "— Other —")
            items = items[:other_idx] + list(extra_options.items()) + items[other_idx:]
            local_model_options = dict(items)

        selected_local = st.selectbox(
            "Model",
            options=list(local_model_options.keys()),
            key="ollama_model_select",
            help="Select a suggested Ollama model or choose 'Other' to enter your own.",
        )

        selected_value = local_model_options[selected_local]

        if selected_value == "other":
            ollama_model = st.text_input(
                "Custom Model Name",
                value=api_keys.get("ollama_model", ""),
                key="ollama_model_custom",
                placeholder="e.g. codellama, phi3, your-custom-model",
            )
        elif selected_value is not None:
            ollama_model = selected_value
        else:
            # Selected a tier header — no model chosen
            ollama_model = ""

        if ollama_model:
            api_keys["ollama_model"] = ollama_model

    # --- Cloud API Keys ---
    with st.sidebar.expander("Cloud API Keys"):
        save_to_disk = st.checkbox(
            "Remember keys",
            value=True,
            help="Save API keys to disk so you don't have to re-enter them next time.",
            key="save_keys_to_disk",
        )

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
                key=f"apikey_{provider_id}",
            )
            if key:
                api_keys[provider_id] = key
            elif provider_id in api_keys and provider_id not in ("ollama_endpoint", "ollama_model"):
                del api_keys[provider_id]

        # Persist to disk when checkbox is on
        if save_to_disk:
            save_keys(api_keys)
        else:
            clear_keys()

    st.session_state.api_keys = api_keys


def _resolve_model_and_key(model_name, model_tier, api_key_input, api_keys):
    """Resolve actual model name, source, and API key based on tier."""
    if model_tier == "Free Models":
        # Free models use HF_API_KEY from env
        import os
        hf_key = os.environ.get("HF_API_KEY", "")
        if not hf_key:
            # Fall back to sidebar-provided HF key
            hf_key = api_keys.get("huggingface", "")
        if not hf_key:
            return model_name, "", "HuggingFace"
        source = get_model_source(model_name)
        return model_name, hf_key, source
    else:
        # Paid models use user-provided keys
        key_val, provider = resolve_api_key(model_name, api_keys)
        source = get_model_source(model_name)
        return model_name, key_val, source


def render_model_selector(mode="single", key_prefix=""):
    """Render model selection UI with free/paid tier.

    Args:
        mode: "single", "classify" (includes comparison/ensemble), or "extract"/"summarize" (single only).
        key_prefix: Prefix for widget keys to avoid collisions.

    Returns:
        dict with keys:
            models_tuples: list of (model, source, api_key) or 4-tuples with creativity
            classify_mode: "Single Model", "Model Comparison", or "Ensemble"
            models_list: list of model name strings
            model_temperatures: dict of model -> temperature
            ensemble_runs: list of (model, temperature) for ensemble
            consensus_threshold: float (for ensemble)
            error: str or None
    """
    result = {
        "models_tuples": [],
        "classify_mode": "Single Model",
        "models_list": [],
        "model_temperatures": {},
        "ensemble_runs": [],
        "consensus_threshold": 0.5,
        "error": None,
    }

    api_keys = st.session_state.get("api_keys", {})

    # Classification mode selector (only for classify function)
    if mode == "classify":
        classify_mode = st.radio(
            "Classification Mode",
            options=["Single Model", "Model Comparison", "Ensemble"],
            horizontal=True,
            key=f"{key_prefix}classify_mode",
            help="Single: one model. Comparison: side-by-side. Ensemble: majority vote.",
        )
        result["classify_mode"] = classify_mode
    else:
        classify_mode = "Single Model"

    # Model tier selector
    model_tier = st.radio(
        "Model Tier",
        options=["Free Models", "Bring Your Own Key"],
        key=f"{key_prefix}model_tier",
        horizontal=True,
    )

    is_free = model_tier == "Free Models"

    if is_free:
        model_options = FREE_MODEL_DISPLAY_NAMES
        model_values = list(FREE_MODELS_MAP.values())
    else:
        model_options = PAID_MODEL_CHOICES
        model_values = PAID_MODEL_CHOICES

    model_temperatures = {}
    ensemble_runs = []

    if classify_mode == "Ensemble":
        # Dynamic run rows allowing same model multiple times
        if "ensemble_num_runs" not in st.session_state:
            st.session_state.ensemble_num_runs = 3

        st.markdown("**Model Runs** (select 3+ runs)")
        for i in range(st.session_state.ensemble_num_runs):
            cols = st.columns([3, 1, 0.5])
            with cols[0]:
                default_idx = i % len(model_options)
                selected = st.selectbox(
                    f"Run {i + 1}",
                    options=model_options,
                    index=default_idx,
                    key=f"{key_prefix}ensemble_model_{i}",
                    label_visibility="collapsed",
                )
            with cols[1]:
                temp = st.number_input(
                    "Temp",
                    min_value=0.0,
                    max_value=2.0,
                    value=round(i * 0.25, 2),
                    step=0.25,
                    key=f"{key_prefix}ensemble_temp_{i}",
                    label_visibility="collapsed",
                )
            with cols[2]:
                if st.session_state.ensemble_num_runs > 3:
                    if st.button("x", key=f"{key_prefix}ensemble_remove_{i}"):
                        st.session_state.ensemble_num_runs -= 1
                        st.rerun()

            # Resolve display name to actual model name for free tier
            model_name = FREE_MODELS_MAP[selected] if is_free else selected
            ensemble_runs.append((model_name, temp))

        if st.button("Add Run", key=f"{key_prefix}add_ensemble_run"):
            st.session_state.ensemble_num_runs += 1
            st.rerun()

        result["ensemble_runs"] = ensemble_runs
        result["models_list"] = [r[0] for r in ensemble_runs]

        # Consensus threshold
        consensus_options = {
            "Majority (50%+)": 0.5,
            "Two-Thirds (67%+)": 0.67,
            "Unanimous (100%)": 1.0,
        }
        consensus_choice = st.radio(
            "Consensus Rule",
            options=list(consensus_options.keys()),
            horizontal=True,
            key=f"{key_prefix}consensus_choice",
            help="How many models must agree for a category to be marked present",
        )
        result["consensus_threshold"] = consensus_options[consensus_choice]

        # Build model tuples
        for m, temp in ensemble_runs:
            m_name, key_val, m_source = _resolve_model_and_key(m, model_tier, None, api_keys)
            if not key_val:
                result["error"] = f"No API key for {m_source} (needed for {m})"
                return result
            result["models_tuples"].append((m, m_source, key_val, {"creativity": temp}))

    elif classify_mode == "Model Comparison":
        default_models = model_options[:2] if len(model_options) >= 2 else model_options
        selected_models = st.multiselect(
            "Models (select 2+)",
            options=model_options,
            default=default_models,
            key=f"{key_prefix}classify_models_multi",
        )
        # Resolve display names to actual model names
        models_list = [FREE_MODELS_MAP[d] for d in selected_models] if is_free else selected_models
        result["models_list"] = models_list

        if models_list:
            st.markdown("**Model Temperature**")
            temp_cols = st.columns(len(models_list))
            for idx, (col, m) in enumerate(zip(temp_cols, models_list)):
                short_name = m.split("/")[-1].split(":")[0][:20]
                model_temperatures[m] = col.number_input(
                    short_name,
                    min_value=0.0,
                    max_value=2.0,
                    value=0.0,
                    step=0.25,
                    key=f"{key_prefix}temp_{idx}",
                    help=f"Temperature for {m}",
                )
        result["model_temperatures"] = model_temperatures

        # Build model tuples
        for m in models_list:
            m_name, key_val, m_source = _resolve_model_and_key(m, model_tier, None, api_keys)
            if not key_val:
                result["error"] = f"No API key for {m_source} (needed for {m})"
                return result
            temp = model_temperatures.get(m)
            if temp is not None:
                result["models_tuples"].append((m, m_source, key_val, {"creativity": temp}))
            else:
                result["models_tuples"].append((m, m_source, key_val))

    else:
        # Single model
        model_display = st.selectbox("Model", options=model_options, key=f"{key_prefix}model")
        model = FREE_MODELS_MAP[model_display] if is_free else model_display
        result["models_list"] = [model]

        m_name, key_val, m_source = _resolve_model_and_key(model, model_tier, None, api_keys)
        if not key_val:
            result["error"] = f"No API key for {m_source} (needed for {model}). Add it in the sidebar or set HF_API_KEY env var."
            return result
        result["models_tuples"].append((model, m_source, key_val))

    return result
