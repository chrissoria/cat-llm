"""
Configuration: model lists, API key helpers, constants.
"""

import os

MAX_CATEGORIES = 10
INITIAL_CATEGORIES = 3
MAX_FILE_SIZE_MB = 100

# Free models — display name -> actual API model name
# These are routed through HuggingFace Inference API and use HF_API_KEY env var
FREE_MODELS_MAP = {
    "Qwen 2.5 72B": "Qwen/Qwen2.5-72B-Instruct",
}
FREE_MODEL_DISPLAY_NAMES = list(FREE_MODELS_MAP.keys())

# Paid models (user provides their own API key)
PAID_MODEL_CHOICES = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gpt-4.1",
    "gpt-4o",
    "gpt-4o-mini",
    "claude-sonnet-4-5-20250929",
    "claude-opus-4-20250514",
    "claude-3-5-haiku-20241022",
    "mistral-large-latest",
    "mistral-medium-2505",
    "grok-4-fast-non-reasoning",
]

# Combined list for components that need all models
MODEL_CHOICES = list(FREE_MODELS_MAP.values()) + PAID_MODEL_CHOICES

# HuggingFace-routed models (need HF_API_KEY)
HF_ROUTED_MODELS = list(FREE_MODELS_MAP.values())

# Provider -> env var name mapping (for users who set env vars)
PROVIDER_ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "huggingface": "HF_API_KEY",
    "perplexity": "PERPLEXITY_API_KEY",
    "xai": "XAI_API_KEY",
}


def get_model_source(model):
    """Auto-detect model provider from model name."""
    model_lower = model.lower()
    if "gpt" in model_lower:
        return "openai"
    elif "claude" in model_lower:
        return "anthropic"
    elif "gemini" in model_lower:
        return "google"
    elif "mistral" in model_lower and ":novita" not in model_lower:
        return "mistral"
    elif any(x in model_lower for x in [":novita", ":groq", "qwen", "llama", "deepseek"]):
        return "huggingface"
    elif "sonar" in model_lower:
        return "perplexity"
    elif "grok" in model_lower:
        return "xai"
    return "huggingface"


def resolve_api_key(model, api_keys_dict):
    """Resolve the API key for a model from the user-provided keys dict.

    Args:
        model: Model name string.
        api_keys_dict: Dict mapping provider names to API key strings,
                       e.g. {"openai": "sk-...", "anthropic": "sk-ant-..."}.
    Returns:
        (api_key, provider_name) tuple.
    """
    provider = get_model_source(model)
    # Check user-provided keys first
    key = api_keys_dict.get(provider, "")
    if key:
        return key, provider
    # Fall back to environment variables
    env_var = PROVIDER_ENV_KEYS.get(provider, "")
    if env_var:
        key = os.environ.get(env_var, "")
        if key:
            return key, provider
    return "", provider


def render_single_model_selector(key_prefix=""):
    """Render a simple free/paid model selector (no ensemble/comparison).

    Returns:
        (model_name, api_key, model_source, error) tuple.
    """
    import streamlit as st

    model_tier = st.radio(
        "Model Tier",
        options=["Free Models", "Bring Your Own Key"],
        key=f"{key_prefix}model_tier",
        horizontal=True,
    )

    api_keys = st.session_state.get("api_keys", {})

    if model_tier == "Free Models":
        model_display = st.selectbox("Model", options=FREE_MODEL_DISPLAY_NAMES, key=f"{key_prefix}model")
        model = FREE_MODELS_MAP[model_display]
        hf_key = os.environ.get("HF_API_KEY", "") or api_keys.get("huggingface", "")
        if not hf_key:
            return model, "", "huggingface", "No HuggingFace API key. Set HF_API_KEY env var or add it in the sidebar."
        return model, hf_key, get_model_source(model), None
    else:
        model = st.selectbox("Model", options=PAID_MODEL_CHOICES, key=f"{key_prefix}model")
        api_key, provider = resolve_api_key(model, api_keys)
        if not api_key:
            return model, "", provider, f"No API key for {provider}. Add it in the sidebar."
        return model, api_key, get_model_source(model), None
