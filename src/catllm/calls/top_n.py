# Top N category extraction functions for various LLM providers

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
    """
    from openai import OpenAI

    base_url = (
        "https://api.perplexity.ai" if model_source == "perplexity" 
        else "https://router.huggingface.co/v1" if model_source == "huggingface"
        else "https://api.x.ai/v1" if model_source == "xai"
        else None
    )

    client = OpenAI(api_key=api_key, base_url=base_url)

    response_obj = client.chat.completions.create(
        model=user_model,
        messages=[
            {'role': 'system', 'content': f"""You are a helpful assistant that extracts categories from survey responses. \
                                        The specific task is to identify {specificity} categories of responses to a survey question. \
             The research question is: {research_question}""" if research_question else "You are a helpful assistant."},
            {'role': 'user', 'content': prompt}
        ],
        **({"temperature": creativity} if creativity is not None else {})
    )
    
    return response_obj.choices[0].message.content


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
    """
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    # Build system prompt
    if research_question:
        system_content = (f"You are a helpful assistant that extracts categories from survey responses. "
                        f"The specific task is to identify {specificity} categories of responses to a survey question. "
                        f"The research question is: {research_question}")
    else:
        system_content = "You are a helpful assistant."
    
    response_obj = client.messages.create(
        model=user_model,
        max_tokens=4096,
        system=system_content,
        messages=[
            {'role': 'user', 'content': prompt}
        ],
        **({"temperature": creativity} if creativity is not None else {})
    )
    
    return response_obj.content[0].text


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
    from mistralai import Mistral
    
    client = Mistral(api_key=api_key)
    
    # Build system prompt
    if research_question:
        system_content = (f"You are a helpful assistant that extracts categories from survey responses. "
                        f"The specific task is to identify {specificity} categories of responses to a survey question. "
                        f"The research question is: {research_question}")
    else:
        system_content = "You are a helpful assistant."
    
    response_obj = client.chat.complete(
        model=user_model,
        messages=[
            {'role': 'system', 'content': system_content},
            {'role': 'user', 'content': prompt}
        ],
        **({"temperature": creativity} if creativity is not None else {})
    )
    
    return response_obj.choices[0].message.content

