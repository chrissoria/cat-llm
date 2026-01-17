# Image-aware Stepback prompting functions for various LLM providers
# These functions generate abstract insights about image categorization tasks

import requests


def get_image_stepback_insight_openai(
    stepback,
    api_key,
    user_model,
    model_source="openai",
    creativity=None
):
    """
    Get stepback insight for image categorization from OpenAI-compatible APIs.
    Supports OpenAI, Perplexity, Huggingface, and xAI.

    The stepback prompt asks for abstract thinking about image categorization
    before analyzing specific images.

    Uses direct HTTP requests instead of OpenAI SDK for lighter dependencies.
    """
    # Determine the base URL based on model source
    if model_source == "huggingface":
        from catllm._providers import _detect_huggingface_endpoint
        base_url = _detect_huggingface_endpoint(api_key, user_model)
    elif model_source == "huggingface-together":
        base_url = "https://router.huggingface.co/together/v1"
    elif model_source == "perplexity":
        base_url = "https://api.perplexity.ai"
    elif model_source == "xai":
        base_url = "https://api.x.ai/v1"
    else:
        base_url = "https://api.openai.com/v1"

    endpoint = f"{base_url}/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": user_model,
        "messages": [{"role": "user", "content": stepback}],
    }

    if creativity is not None:
        payload["temperature"] = creativity

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        stepback_insight = result["choices"][0]["message"]["content"]

        return stepback_insight, True

    except Exception as e:
        return None, False


def get_image_stepback_insight_anthropic(
    stepback,
    api_key,
    user_model,
    model_source="anthropic",
    creativity=None
):
    """
    Get stepback insight for image categorization from Anthropic Claude.

    Uses direct HTTP requests instead of Anthropic SDK for lighter dependencies.
    """
    import requests

    endpoint = "https://api.anthropic.com/v1/messages"

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }

    payload = {
        "model": user_model,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": stepback}],
    }

    if creativity is not None:
        payload["temperature"] = creativity

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()

        # Parse response - Anthropic returns content as a list
        content = result.get("content", [])
        if content and content[0].get("type") == "text":
            stepback_insight = content[0].get("text", "")
            return stepback_insight, True

        return None, False

    except Exception as e:
        return None, False


def get_image_stepback_insight_google(
    stepback,
    api_key,
    user_model,
    model_source="google",
    creativity=None
):
    """
    Get stepback insight for image categorization from Google Gemini.
    """
    import requests

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{user_model}:generateContent"

    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json"
    }

    payload = {
        "contents": [{
            "parts": [{"text": stepback}]
        }],
        **({"generationConfig": {"temperature": creativity}} if creativity is not None else {})
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()

        result = response.json()
        stepback_insight = result['candidates'][0]['content']['parts'][0]['text']

        return stepback_insight, True

    except Exception as e:
        return None, False


def get_image_stepback_insight_mistral(
    stepback,
    api_key,
    user_model,
    model_source="mistral",
    creativity=None
):
    """
    Get stepback insight for image categorization from Mistral AI.
    """
    import requests

    endpoint = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": user_model,
        "messages": [{'role': 'user', 'content': stepback}],
    }
    if creativity is not None:
        payload["temperature"] = creativity

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        stepback_insight = result["choices"][0]["message"]["content"]

        return stepback_insight, True

    except Exception as e:
        return None, False


def get_image_stepback_insight(model_source, stepback, api_key, user_model, creativity):
    """Get step-back insight using the appropriate provider for image tasks."""
    stepback_functions = {
        "openai": get_image_stepback_insight_openai,
        "perplexity": get_image_stepback_insight_openai,
        "huggingface": get_image_stepback_insight_openai,
        "huggingface-together": get_image_stepback_insight_openai,
        "xai": get_image_stepback_insight_openai,
        "anthropic": get_image_stepback_insight_anthropic,
        "google": get_image_stepback_insight_google,
        "mistral": get_image_stepback_insight_mistral,
    }

    func = stepback_functions.get(model_source)
    if func is None:
        return None, False

    return func(
        stepback=stepback,
        api_key=api_key,
        user_model=user_model,
        model_source=model_source,
        creativity=creativity
    )
