"""
Unified HTTP-based multi-class text classification.

This module uses raw HTTP requests (via `requests`) for all LLM providers,
eliminating the need for provider-specific SDKs like openai, anthropic, mistralai, etc.
"""

import json
import time
import requests
import pandas as pd
import regex
from tqdm import tqdm


# =============================================================================
# Provider Configuration
# =============================================================================

PROVIDER_CONFIG = {
    "openai": {
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
    },
    "anthropic": {
        "endpoint": "https://api.anthropic.com/v1/messages",
        "auth_header": "x-api-key",
        "auth_prefix": "",
    },
    "google": {
        "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        "auth_header": "x-goog-api-key",
        "auth_prefix": "",
    },
    "mistral": {
        "endpoint": "https://api.mistral.ai/v1/chat/completions",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
    },
    "perplexity": {
        "endpoint": "https://api.perplexity.ai/chat/completions",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
    },
    "xai": {
        "endpoint": "https://api.x.ai/v1/chat/completions",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
    },
    "huggingface": {
        "endpoint": "https://router.huggingface.co/v1/chat/completions",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
    },
}


# =============================================================================
# Unified API Client
# =============================================================================

class UnifiedLLMClient:
    """A unified client for calling various LLM providers via HTTP."""

    def __init__(self, provider: str, api_key: str, model: str):
        self.provider = provider.lower()
        self.api_key = api_key
        self.model = model

        if self.provider not in PROVIDER_CONFIG:
            raise ValueError(f"Unsupported provider: {provider}. "
                           f"Supported: {list(PROVIDER_CONFIG.keys())}")

        self.config = PROVIDER_CONFIG[self.provider]

    def _get_endpoint(self) -> str:
        """Get the API endpoint, substituting model if needed."""
        endpoint = self.config["endpoint"]
        if "{model}" in endpoint:
            endpoint = endpoint.format(model=self.model)
        return endpoint

    def _get_headers(self) -> dict:
        """Build request headers for the provider."""
        headers = {"Content-Type": "application/json"}
        auth_header = self.config["auth_header"]
        auth_prefix = self.config["auth_prefix"]
        headers[auth_header] = f"{auth_prefix}{self.api_key}"

        # Anthropic requires additional headers
        if self.provider == "anthropic":
            headers["anthropic-version"] = "2023-06-01"

        return headers

    def _build_payload(
        self,
        messages: list,
        json_schema: dict = None,
        creativity: float = None,
        max_tokens: int = 4096,
    ) -> dict:
        """Build the request payload for the specific provider."""

        if self.provider == "anthropic":
            return self._build_anthropic_payload(messages, json_schema, creativity, max_tokens)
        elif self.provider == "google":
            return self._build_google_payload(messages, json_schema, creativity)
        else:
            # OpenAI-compatible providers
            return self._build_openai_payload(messages, json_schema, creativity)

    def _build_openai_payload(
        self,
        messages: list,
        json_schema: dict = None,
        creativity: float = None,
    ) -> dict:
        """Build payload for OpenAI-compatible APIs."""
        payload = {
            "model": self.model,
            "messages": messages,
        }

        # Structured output
        if json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "classification_result",
                    "strict": True,
                    "schema": json_schema,
                }
            }
        else:
            payload["response_format"] = {"type": "json_object"}

        if creativity is not None:
            payload["temperature"] = creativity

        return payload

    def _build_anthropic_payload(
        self,
        messages: list,
        json_schema: dict = None,
        creativity: float = None,
        max_tokens: int = 4096,
    ) -> dict:
        """Build payload for Anthropic API."""
        # Extract system message if present
        system_content = None
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                user_messages.append(msg)

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": user_messages,
        }

        if system_content:
            payload["system"] = system_content

        if creativity is not None:
            payload["temperature"] = creativity

        # Use tool calling for structured output (most reliable for Anthropic)
        if json_schema:
            payload["tools"] = [{
                "name": "return_categories",
                "description": "Return categorization results",
                "input_schema": json_schema,
            }]
            payload["tool_choice"] = {"type": "tool", "name": "return_categories"}

        return payload

    def _build_google_payload(
        self,
        messages: list,
        json_schema: dict = None,
        creativity: float = None,
    ) -> dict:
        """Build payload for Google Gemini API."""
        # Convert messages to Google format
        # Combine system + user messages into a single prompt
        combined_text = ""
        for msg in messages:
            if msg["role"] == "system":
                combined_text += msg["content"] + "\n\n"
            elif msg["role"] == "user":
                combined_text += msg["content"]

        payload = {
            "contents": [{"parts": [{"text": combined_text}]}],
            "generationConfig": {}
        }

        if json_schema:
            payload["generationConfig"]["responseMimeType"] = "application/json"
            payload["generationConfig"]["responseSchema"] = json_schema
        else:
            payload["generationConfig"]["responseMimeType"] = "application/json"

        if creativity is not None:
            payload["generationConfig"]["temperature"] = creativity

        return payload

    def _parse_response(self, response_json: dict) -> str:
        """Parse the response based on provider format."""
        if self.provider == "anthropic":
            return self._parse_anthropic_response(response_json)
        elif self.provider == "google":
            return self._parse_google_response(response_json)
        else:
            # OpenAI-compatible
            return self._parse_openai_response(response_json)

    def _parse_openai_response(self, response_json: dict) -> str:
        """Parse OpenAI-compatible response."""
        return response_json["choices"][0]["message"]["content"]

    def _parse_anthropic_response(self, response_json: dict) -> str:
        """Parse Anthropic response (handles both text and tool use)."""
        content = response_json.get("content", [])
        for block in content:
            if block.get("type") == "tool_use":
                # Return the tool input as JSON string
                return json.dumps(block.get("input", {}))
            elif block.get("type") == "text":
                return block.get("text", "")
        return ""

    def _parse_google_response(self, response_json: dict) -> str:
        """Parse Google Gemini response."""
        candidates = response_json.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                return parts[0].get("text", "")
        return ""

    def complete(
        self,
        messages: list,
        json_schema: dict = None,
        creativity: float = None,
        max_retries: int = 5,
        initial_delay: float = 2.0,
    ) -> tuple[str, str | None]:
        """
        Make a completion request to the LLM provider.

        Returns:
            tuple: (response_text, error_message)
                   error_message is None on success
        """
        endpoint = self._get_endpoint()
        headers = self._get_headers()
        payload = self._build_payload(messages, json_schema, creativity)

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=120,
                )

                # Check for HTTP errors
                if response.status_code == 404:
                    return None, f"Model '{self.model}' not found for {self.provider}"
                elif response.status_code in [401, 403]:
                    return None, f"Authentication failed for {self.provider}"
                elif response.status_code == 429:
                    # Rate limited - retry with backoff
                    if attempt < max_retries - 1:
                        wait_time = initial_delay * (2 ** attempt) * 5  # Longer wait for rate limits
                        print(f"⚠️ Rate limited. Waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return None, "Rate limit exceeded after retries"
                elif response.status_code >= 500:
                    # Server error - retry
                    if attempt < max_retries - 1:
                        wait_time = initial_delay * (2 ** attempt)
                        print(f"⚠️ Server error {response.status_code}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return None, f"Server error {response.status_code} after retries"

                response.raise_for_status()
                response_json = response.json()
                result = self._parse_response(response_json)
                return result, None

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    wait_time = initial_delay * (2 ** attempt)
                    print(f"⚠️ Request timeout. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    return None, "Request timeout after retries"

            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = initial_delay * (2 ** attempt)
                    print(f"⚠️ Request error: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    return None, f"Request failed: {e}"

            except json.JSONDecodeError as e:
                return None, f"Failed to parse response JSON: {e}"

        return None, "Max retries exceeded"


# =============================================================================
# Helper Functions
# =============================================================================

def detect_provider(model_name: str, provider: str = "auto") -> str:
    """Auto-detect provider from model name if not explicitly provided."""
    if provider and provider.lower() != "auto":
        return provider.lower()

    model_lower = model_name.lower()

    if "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower:
        return "openai"
    elif "claude" in model_lower:
        return "anthropic"
    elif "gemini" in model_lower or "gemma" in model_lower:
        return "google"
    elif "mistral" in model_lower or "mixtral" in model_lower:
        return "mistral"
    elif "sonar" in model_lower or "pplx" in model_lower:
        return "perplexity"
    elif "grok" in model_lower:
        return "xai"
    elif "llama" in model_lower or "meta" in model_lower or "deepseek" in model_lower or "qwen" in model_lower:
        return "huggingface"
    else:
        raise ValueError(
            f"Could not auto-detect provider from '{model_name}'. "
            "Please specify provider explicitly."
        )


def build_json_schema(categories: list) -> dict:
    """Build a JSON schema for the classification output."""
    properties = {}
    for i, cat in enumerate(categories, 1):
        properties[str(i)] = {
            "type": "string",
            "enum": ["0", "1"],
            "description": cat,
        }

    return {
        "type": "object",
        "properties": properties,
        "required": list(properties.keys()),
        "additionalProperties": False,
    }


def extract_json(reply: str) -> str:
    """Extract JSON from model reply."""
    if reply is None:
        return '{"1":"e"}'

    extracted = regex.findall(r'\{(?:[^{}]|(?R))*\}', reply, regex.DOTALL)
    if extracted:
        # Clean up the JSON string
        return extracted[0].replace('[', '').replace(']', '').replace('\n', '').replace(" ", '')
    else:
        return '{"1":"e"}'


# =============================================================================
# Main Classification Function
# =============================================================================

def multi_class_unified(
    survey_input,
    categories: list,
    api_key: str,
    model: str = "gpt-4o",
    provider: str = "auto",
    survey_question: str = "",
    creativity: float = None,
    chain_of_thought: bool = True,
    use_json_schema: bool = True,
    filename: str = None,
):
    """
    Multi-class text classification using a unified HTTP-based approach.

    This is a simplified test version that uses raw HTTP requests for all providers,
    eliminating SDK dependencies.

    Args:
        survey_input: List or Series of text responses to classify
        categories: List of category names
        api_key: API key for the LLM provider
        model: Model name (e.g., "gpt-4o", "claude-3-5-sonnet-20241022", "gemini-1.5-flash")
        provider: Provider name or "auto" to detect from model name
        survey_question: Optional context about what question was asked
        creativity: Temperature setting (None for provider default)
        chain_of_thought: Whether to use step-by-step reasoning in prompt
        use_json_schema: Whether to use strict JSON schema (vs just json_object mode)
        filename: Optional CSV filename to save results

    Returns:
        DataFrame with classification results
    """
    # Detect provider
    provider = detect_provider(model, provider)
    print(f"Using provider: {provider}, model: {model}")

    # Initialize client
    client = UnifiedLLMClient(provider=provider, api_key=api_key, model=model)

    # Build category string and schema
    categories_str = "\n".join(f"{i + 1}. {cat}" for i, cat in enumerate(categories))
    json_schema = build_json_schema(categories) if use_json_schema else None

    # Print categories
    print(f"\nCategories to classify ({len(categories)} total):")
    for i, cat in enumerate(categories, 1):
        print(f"  {i}. {cat}")
    print()

    # Build prompt template
    def build_prompt(response_text: str) -> list:
        """Build the classification prompt for a single response."""
        survey_context = f"A respondent was asked: {survey_question}." if survey_question else ""

        if chain_of_thought:
            user_content = f"""{survey_context}

Categorize this survey response "{response_text}" into the following categories that apply:
{categories_str}

Let's think step by step:
1. First, identify the main themes mentioned in the response
2. Then, match each theme to the relevant categories
3. Finally, assign 1 to matching categories and 0 to non-matching categories

Provide your answer in JSON format where the category number is the key and "1" if present, "0" if not."""
        else:
            user_content = f"""{survey_context}
Categorize this survey response "{response_text}" into the following categories that apply:
{categories_str}

Provide your answer in JSON format where the category number is the key and "1" if present, "0" if not."""

        return [
            {"role": "system", "content": "You are an expert at categorizing survey responses. Return results as JSON."},
            {"role": "user", "content": user_content},
        ]

    # Process each response
    results = []
    extracted_jsons = []

    for response in tqdm(survey_input, desc="Classifying responses"):
        if pd.isna(response):
            results.append(("Skipped NaN", "Skipped NaN input"))
            extracted_jsons.append('{"1":"e"}')
            continue

        messages = build_prompt(response)
        reply, error = client.complete(
            messages=messages,
            json_schema=json_schema,
            creativity=creativity,
        )

        if error:
            results.append((None, error))
            extracted_jsons.append('{"1":"e"}')
        else:
            results.append((reply, None))
            extracted_jsons.append(extract_json(reply))

    # Build output DataFrame
    normalized_data_list = []
    for json_str in extracted_jsons:
        try:
            parsed = json.loads(json_str)
            normalized_data_list.append(pd.json_normalize(parsed))
        except json.JSONDecodeError:
            normalized_data_list.append(pd.DataFrame({"1": ["e"]}))

    normalized_data = pd.concat(normalized_data_list, ignore_index=True)

    # Create main DataFrame
    df = pd.DataFrame({
        'survey_input': pd.Series(survey_input).reset_index(drop=True),
        'model_response': [r[0] for r in results],
        'error': [r[1] for r in results],
        'json': pd.Series(extracted_jsons).reset_index(drop=True),
    })

    df = pd.concat([df, normalized_data], axis=1)

    # Rename category columns
    df = df.rename(columns=lambda x: f'category_{x}' if str(x).isdigit() else x)

    # Process category columns
    cat_cols = [col for col in df.columns if col.startswith('category_')]

    # Identify invalid rows
    has_invalid = df[cat_cols].apply(
        lambda col: pd.to_numeric(col, errors='coerce').isna() & col.notna()
    ).any(axis=1)

    df['processing_status'] = (~has_invalid).map({True: 'success', False: 'error'})
    df.loc[has_invalid, cat_cols] = pd.NA

    # Convert to numeric
    for col in cat_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill NaN with 0 for valid rows
    df.loc[~has_invalid, cat_cols] = df.loc[~has_invalid, cat_cols].fillna(0)

    # Convert to Int64
    df[cat_cols] = df[cat_cols].astype('Int64')

    # Create categories_id
    df['categories_id'] = df[cat_cols].apply(
        lambda x: ','.join(x.dropna().astype(int).astype(str)), axis=1
    )

    if filename:
        df.to_csv(filename, index=False)
        print(f"\nResults saved to {filename}")

    return df
