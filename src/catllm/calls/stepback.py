# Stepback prompting functions for various LLM providers

def get_stepback_insight_openai(
    stepback,
    api_key,
    user_model,
    model_source="openai",
    creativity=None
):
    """
    Get stepback insight from OpenAI-compatible APIs.
    Supports OpenAI, Perplexity, Huggingface, and xAI.
    """
    from openai import OpenAI
    # Conditional base_url setting based on model source
    base_url = (
        "https://api.perplexity.ai" if model_source == "perplexity" 
        else "https://router.huggingface.co/v1" if model_source == "huggingface"
        else "https://api.x.ai/v1" if model_source == "xai"
        else None
    )
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    try:
        stepback_response = client.chat.completions.create(
            model=user_model,
            messages=[{'role': 'user', 'content': stepback}],
            **({"temperature": creativity} if creativity is not None else {})
        )
        stepback_insight = stepback_response.choices[0].message.content
        
        return stepback_insight, True
        
    except Exception as e:
        return None, False


def get_stepback_insight_anthropic(
    stepback,
    api_key,
    user_model,
    model_source="anthropic",
    creativity=None
):
    """
    Get stepback insight from Anthropic Claude.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    
    try:
        stepback_response = client.messages.create(
            model=user_model,
            max_tokens=4096,
            messages=[{'role': 'user', 'content': stepback}],
            **({"temperature": creativity} if creativity is not None else {})
        )
        stepback_insight = stepback_response.content[0].text
        
        return stepback_insight, True
        
    except Exception as e:
        return None, False


def get_stepback_insight_google(
    stepback,
    api_key,
    user_model,
    model_source="google",
    creativity=None
):
    """
    Get stepback insight from Google Gemini.
    """
    import requests
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{user_model}:generateContent?key={api_key}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "contents": [{
            "parts": [{"text": stepback}],

            **({"generationConfig": {"temperature": creativity}} if creativity is not None else {})
        }]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise error for bad status codes
        
        result = response.json()
        stepback_insight = result['candidates'][0]['content']['parts'][0]['text']
        
        return stepback_insight, True
        
    except Exception as e:
        return None, False


def get_stepback_insight_mistral(
    stepback,
    api_key,
    user_model,
    model_source="mistral",
    creativity=None
):
    """
    Get stepback insight from Mistral AI.
    """
    from mistralai import Mistral
    
    client = Mistral(api_key=api_key)
    
    try:
        stepback_response = client.chat.complete(
            model=user_model,
            messages=[{'role': 'user', 'content': stepback}],
            **({"temperature": creativity} if creativity is not None else {})
        )
        stepback_insight = stepback_response.choices[0].message.content
        
        return stepback_insight, True
        
    except Exception as e:
        return None, False
