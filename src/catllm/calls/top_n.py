# Top N category extraction functions for various LLM providers

import requests


def get_openai_top_n(
    prompt,
    user_model,
    specificity,
    model_source,
    api_key,
    research_question,
    creativity
):
    """
    Get response from OpenAI API with system message.
    Supports OpenAI, Perplexity, Huggingface, and xAI.

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

    # Build system message
    if research_question:
        system_content = (
            f"You are a helpful assistant that extracts categories from survey responses. "
            f"The specific task is to identify {specificity} categories of responses to a survey question. "
            f"The research question is: {research_question}"
        )
    else:
        system_content = "You are a helpful assistant."

    payload = {
        "model": user_model,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ],
    }

    if creativity is not None:
        payload["temperature"] = creativity

    response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    result = response.json()

    return result["choices"][0]["message"]["content"]


def get_anthropic_top_n(
    prompt,
    user_model,
    model_source,
    specificity,
    api_key,
    research_question,
    creativity
):
    """
    Get response from Anthropic API with system prompt.

    Uses direct HTTP requests instead of Anthropic SDK for lighter dependencies.
    """
    import requests

    endpoint = "https://api.anthropic.com/v1/messages"

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }

    # Build system prompt
    if research_question:
        system_content = (f"You are a helpful assistant that extracts categories from survey responses. "
                        f"The specific task is to identify {specificity} categories of responses to a survey question. "
                        f"The research question is: {research_question}")
    else:
        system_content = "You are a helpful assistant."

    payload = {
        "model": user_model,
        "max_tokens": 4096,
        "system": system_content,
        "messages": [{"role": "user", "content": prompt}],
    }

    if creativity is not None:
        payload["temperature"] = creativity

    response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    result = response.json()

    # Parse response - Anthropic returns content as a list
    content = result.get("content", [])
    if content and content[0].get("type") == "text":
        return content[0].get("text", "")

    return ""


def get_google_top_n(
    prompt,
    user_model,
    specificity,
    model_source,
    api_key,
    research_question,
    creativity
):
    """
    Get response from Google Gemini API.
    """
    import requests
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{user_model}:generateContent"
    
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    # Build system-like content in the prompt
    if research_question:
        system_context = (f"You are a helpful assistant that extracts categories from survey responses. "
                        f"The specific task is to identify {specificity} categories of responses to a survey question. "
                        f"The research question is: {research_question}\n\n")
    else:
        system_context = "You are a helpful assistant.\n\n"
    
    full_prompt = system_context + prompt
    
    payload = {
        "contents": [{
            "parts": [{"text": full_prompt}]
        }],
        "generationConfig": {
            **({"temperature": creativity} if creativity is not None else {})
        }
    }
    
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    
    result = response.json()
    
    if "candidates" in result and result["candidates"]:
        return result["candidates"][0]["content"]["parts"][0]["text"]
    else:
        return "No response generated"


def get_mistral_top_n(
    prompt,
    user_model,
    specificity,
    model_source,
    api_key,
    research_question,
    creativity
):
    """
    Get response from Mistral AI API.
    """
    import requests

    endpoint = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Build system prompt
    if research_question:
        system_content = (f"You are a helpful assistant that extracts categories from survey responses. "
                        f"The specific task is to identify {specificity} categories of responses to a survey question. "
                        f"The research question is: {research_question}")
    else:
        system_content = "You are a helpful assistant."

    payload = {
        "model": user_model,
        "messages": [
            {'role': 'system', 'content': system_content},
            {'role': 'user', 'content': prompt}
        ],
    }
    if creativity is not None:
        payload["temperature"] = creativity

    response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    result = response.json()

    return result["choices"][0]["message"]["content"]

