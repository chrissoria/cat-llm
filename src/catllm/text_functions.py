"""
Text classification functions for CatLLM.

This module provides multi-class text classification using a unified HTTP-based approach
that works with multiple LLM providers (OpenAI, Anthropic, Google, Mistral, xAI,
Perplexity, HuggingFace, and Ollama) without requiring provider-specific SDKs.
"""

import json
import warnings

# Exported names (excludes deprecated multi_class)
__all__ = [
    "UnifiedLLMClient",
    "detect_provider",
    "set_ollama_endpoint",
    "check_ollama_running",
    "list_ollama_models",
    "check_ollama_model",
    "check_system_resources",
    "get_ollama_model_size_estimate",
    "pull_ollama_model",
    "build_json_schema",
    "extract_json",
    "validate_classification_json",
    "ollama_two_step_classify",
    "explore_corpus",
    "explore_common_categories",
    # Internal utilities used by other modules
    "_detect_model_source",
    "_get_stepback_insight",
    "_detect_huggingface_endpoint",
]
import time
import requests
import pandas as pd
import regex
from tqdm import tqdm

from .calls.stepback import (
    get_stepback_insight_openai,
    get_stepback_insight_anthropic,
    get_stepback_insight_google,
    get_stepback_insight_mistral
)
from .calls.CoVe import (
    chain_of_verification_openai,
    chain_of_verification_google,
    chain_of_verification_anthropic,
    chain_of_verification_mistral
)
from .calls.top_n import (
    get_openai_top_n,
    get_anthropic_top_n,
    get_google_top_n,
    get_mistral_top_n
)


# =============================================================================
# HuggingFace Endpoint Auto-Detection
# =============================================================================

def _detect_huggingface_endpoint(api_key: str, model: str) -> str:
    """
    Test which HuggingFace endpoint works for this model.
    Tries generic router first, then Together.

    Args:
        api_key: HuggingFace API key
        model: Model name to test

    Returns:
        Base URL for the working endpoint (without /chat/completions)
    """
    endpoints = [
        "https://router.huggingface.co/v1/chat/completions",
        "https://router.huggingface.co/together/v1/chat/completions",
    ]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 5
    }

    for endpoint in endpoints:
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                # Return the base URL (without /chat/completions)
                return endpoint.replace("/chat/completions", "")
        except Exception:
            continue

    # Default to generic (will fail with informative error)
    return "https://router.huggingface.co/v1"


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
    "huggingface-together": {
        "endpoint": "https://router.huggingface.co/together/v1/chat/completions",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
    },
    "ollama": {
        "endpoint": "http://localhost:11434/v1/chat/completions",
        "auth_header": None,  # No auth required for local Ollama
        "auth_prefix": "",
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

        # Auto-detect HuggingFace endpoint
        if self.provider == "huggingface":
            detected_url = _detect_huggingface_endpoint(api_key, model)
            if "together" in detected_url:
                self.provider = "huggingface-together"

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

        # Some providers (like Ollama) don't require auth
        if auth_header is not None:
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
        thinking_budget: int = None,
        force_json: bool = True,
    ) -> dict:
        """Build the request payload for the specific provider."""

        if self.provider == "anthropic":
            return self._build_anthropic_payload(messages, json_schema, creativity, max_tokens)
        elif self.provider == "google":
            return self._build_google_payload(messages, json_schema, creativity, thinking_budget, force_json)
        else:
            # OpenAI-compatible providers
            return self._build_openai_payload(messages, json_schema, creativity, force_json)

    def _build_openai_payload(
        self,
        messages: list,
        json_schema: dict = None,
        creativity: float = None,
        force_json: bool = True,
    ) -> dict:
        """Build payload for OpenAI-compatible APIs.

        Args:
            force_json: If False and no json_schema, don't set response_format (for text responses)
        """
        payload = {
            "model": self.model,
            "messages": messages,
        }

        # Structured output
        # Ollama and HuggingFace only support json_object mode, not strict json_schema
        if json_schema and self.provider not in ["ollama", "huggingface", "huggingface-together"]:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "classification_result",
                    "strict": True,
                    "schema": json_schema,
                }
            }
        elif json_schema:
            # Ollama/HuggingFace - use json_object mode
            payload["response_format"] = {"type": "json_object"}
        elif force_json:
            # No schema but force JSON output
            payload["response_format"] = {"type": "json_object"}
        # else: no response_format - allow text responses

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
        thinking_budget: int = None,
        force_json: bool = True,
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
            elif msg["role"] == "assistant":
                combined_text += "\n\nAssistant: " + msg["content"] + "\n\n"

        payload = {
            "contents": [{"parts": [{"text": combined_text}]}],
            "generationConfig": {}
        }

        if json_schema:
            payload["generationConfig"]["responseMimeType"] = "application/json"
            payload["generationConfig"]["responseSchema"] = json_schema
        elif force_json:
            payload["generationConfig"]["responseMimeType"] = "application/json"
        # else: no mime type - allow text responses

        if creativity is not None:
            payload["generationConfig"]["temperature"] = creativity

        # Add thinking budget for extended thinking (Google-specific)
        if thinking_budget and thinking_budget > 0:
            payload["thinkingConfig"] = {"thinkingBudget": thinking_budget}

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
        thinking_budget: int = None,
        force_json: bool = True,
        max_retries: int = 5,
        initial_delay: float = 2.0,
    ) -> tuple[str, str | None]:
        """
        Make a completion request to the LLM provider.

        Args:
            messages: List of message dicts with 'role' and 'content'
            json_schema: Optional JSON schema for structured output
            creativity: Temperature setting (None for default)
            thinking_budget: Token budget for Google's extended thinking (0 or None to disable)
            force_json: If True and no json_schema, still request JSON output.
                       Set to False for text-only responses (e.g., CoVe intermediate steps)
            max_retries: Maximum retry attempts
            initial_delay: Initial delay for exponential backoff

        Returns:
            tuple: (response_text, error_message)
                   error_message is None on success
        """
        endpoint = self._get_endpoint()
        headers = self._get_headers()
        payload = self._build_payload(messages, json_schema, creativity, thinking_budget=thinking_budget, force_json=force_json)

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
                        print(f"Rate limited. Waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return None, "Rate limit exceeded after retries"
                elif response.status_code >= 500:
                    # Server error - retry
                    if attempt < max_retries - 1:
                        wait_time = initial_delay * (2 ** attempt)
                        print(f"Server error {response.status_code}. Retrying in {wait_time}s...")
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
                    print(f"Request timeout. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    return None, "Request timeout after retries"

            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = initial_delay * (2 ** attempt)
                    print(f"Request error: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    return None, f"Request failed: {e}"

            except json.JSONDecodeError as e:
                return None, f"Failed to parse response JSON: {e}"

        return None, "Max retries exceeded"


# =============================================================================
# Helper Functions
# =============================================================================

def _detect_model_source(user_model, model_source):
    """Auto-detect model source from model name if not explicitly provided."""
    model_source = model_source.lower()

    if model_source is None or model_source == "auto":
        user_model_lower = user_model.lower()

        if "gpt" in user_model_lower:
            return "openai"
        elif "claude" in user_model_lower:
            return "anthropic"
        elif "gemini" in user_model_lower or "gemma" in user_model_lower:
            return "google"
        elif "llama" in user_model_lower or "meta" in user_model_lower:
            return "huggingface"
        elif "mistral" in user_model_lower or "mixtral" in user_model_lower:
            return "mistral"
        elif "sonar" in user_model_lower or "pplx" in user_model_lower:
            return "perplexity"
        elif "deepseek" in user_model_lower or "qwen" in user_model_lower:
            return "huggingface"
        elif "grok" in user_model_lower:
            return "xai"
        else:
            raise ValueError(
                f"Could not auto-detect model source from '{user_model}'. "
                "Please specify model_source explicitly: OpenAI, Anthropic, Perplexity, Google, xAI, Huggingface, or Mistral"
            )
    return model_source


def _get_stepback_insight(model_source, stepback, api_key, user_model, creativity):
    """Get step-back insight using the appropriate provider."""
    stepback_functions = {
        "openai": get_stepback_insight_openai,
        "perplexity": get_stepback_insight_openai,
        "huggingface": get_stepback_insight_openai,
        "huggingface-together": get_stepback_insight_openai,
        "xai": get_stepback_insight_openai,
        "anthropic": get_stepback_insight_anthropic,
        "google": get_stepback_insight_google,
        "mistral": get_stepback_insight_mistral,
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
            "Please specify provider explicitly: openai, anthropic, google, mistral, "
            "perplexity, xai, huggingface, or ollama."
        )


# =============================================================================
# Ollama Functions
# =============================================================================

def set_ollama_endpoint(host: str = "localhost", port: int = 11434):
    """
    Configure a custom Ollama endpoint.

    Useful if Ollama is running on a different host or port.

    Args:
        host: Hostname where Ollama is running (default: localhost)
        port: Port number (default: 11434)

    Example:
        set_ollama_endpoint("192.168.1.100", 11434)
    """
    PROVIDER_CONFIG["ollama"]["endpoint"] = f"http://{host}:{port}/v1/chat/completions"


def check_ollama_running(host: str = "localhost", port: int = 11434) -> bool:
    """
    Check if Ollama is running and accessible.

    Args:
        host: Hostname where Ollama should be running
        port: Port number

    Returns:
        True if Ollama is running, False otherwise
    """
    try:
        response = requests.get(f"http://{host}:{port}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def list_ollama_models(host: str = "localhost", port: int = 11434) -> list:
    """
    List all models available in the local Ollama installation.

    Args:
        host: Hostname where Ollama is running
        port: Port number

    Returns:
        List of model names, or empty list if Ollama is not running
    """
    try:
        response = requests.get(f"http://{host}:{port}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        return []
    except requests.exceptions.RequestException:
        return []


def check_ollama_model(model: str, host: str = "localhost", port: int = 11434) -> bool:
    """
    Check if a specific model is available in Ollama.

    Args:
        model: Model name to check (e.g., "llama3.2", "mistral")
        host: Hostname where Ollama is running
        port: Port number

    Returns:
        True if model is available, False otherwise
    """
    available_models = list_ollama_models(host, port)
    # Check for exact match or partial match (e.g., "llama3.2" matches "llama3.2:latest")
    model_lower = model.lower()
    return any(
        model_lower == m.lower() or
        m.lower().startswith(f"{model_lower}:") or
        model_lower.startswith(m.lower().split(":")[0])
        for m in available_models
    )


def _format_bytes(size_bytes: int) -> str:
    """Format bytes into human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
    else:
        return f"{size_bytes / (1024 ** 3):.2f} GB"


def _parse_size_string(size_str: str) -> int:
    """Parse a size string like '2.0 GB' into bytes."""
    if size_str == "unknown":
        return 0

    size_str = size_str.strip().upper()
    try:
        if "GB" in size_str:
            return int(float(size_str.replace("GB", "").strip()) * 1024 ** 3)
        elif "MB" in size_str:
            return int(float(size_str.replace("MB", "").strip()) * 1024 ** 2)
        elif "KB" in size_str:
            return int(float(size_str.replace("KB", "").strip()) * 1024)
        else:
            return int(float(size_str.replace("B", "").strip()))
    except ValueError:
        return 0


def check_system_resources(model: str) -> dict:
    """
    Check if system has enough resources to download and run a model.

    Args:
        model: Model name to check

    Returns:
        dict with 'can_download', 'can_run', 'warnings', and 'details'
    """
    import shutil
    import os

    result = {
        "can_download": True,
        "can_run": True,
        "warnings": [],
        "details": {}
    }

    size_estimate = get_ollama_model_size_estimate(model)
    model_size_bytes = _parse_size_string(size_estimate)

    # Check disk space (Ollama typically stores models in ~/.ollama)
    ollama_dir = os.path.expanduser("~/.ollama")
    if not os.path.exists(ollama_dir):
        ollama_dir = os.path.expanduser("~")

    try:
        disk_usage = shutil.disk_usage(ollama_dir)
        free_space = disk_usage.free
        result["details"]["free_disk_space"] = _format_bytes(free_space)
        result["details"]["model_size"] = size_estimate

        # Need at least 1.5x model size for download + extraction
        required_space = int(model_size_bytes * 1.5) if model_size_bytes > 0 else 0

        if required_space > 0 and free_space < required_space:
            result["can_download"] = False
            result["warnings"].append(
                f"Insufficient disk space. Need ~{_format_bytes(required_space)}, "
                f"but only {_format_bytes(free_space)} available."
            )
        elif required_space > 0 and free_space < required_space * 2:
            result["warnings"].append(
                f"Low disk space warning: {_format_bytes(free_space)} available."
            )
    except Exception:
        result["details"]["free_disk_space"] = "unknown"

    # Estimate RAM requirements (rough guide: model size * 1.2 for inference)
    # This is approximate - actual requirements vary by quantization
    if model_size_bytes > 0:
        estimated_ram = model_size_bytes * 1.2
        result["details"]["estimated_ram"] = _format_bytes(int(estimated_ram))

        # Try to get system RAM (works on most systems)
        try:
            import subprocess
            if os.name == 'posix':  # Linux/macOS
                if os.path.exists('/proc/meminfo'):  # Linux
                    with open('/proc/meminfo', 'r') as f:
                        for line in f:
                            if line.startswith('MemTotal:'):
                                total_ram = int(line.split()[1]) * 1024  # Convert KB to bytes
                                break
                else:  # macOS
                    output = subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True)
                    total_ram = int(output.strip())

                result["details"]["total_ram"] = _format_bytes(total_ram)

                if estimated_ram > total_ram * 0.8:
                    result["can_run"] = False
                    result["warnings"].append(
                        f"Model may be too large for your system. "
                        f"Requires ~{_format_bytes(int(estimated_ram))} RAM, "
                        f"but system has {_format_bytes(total_ram)}."
                    )
                elif estimated_ram > total_ram * 0.5:
                    result["warnings"].append(
                        f"Model will use significant RAM (~{_format_bytes(int(estimated_ram))})."
                    )
        except Exception:
            result["details"]["total_ram"] = "unknown"
            # If we can't check RAM, warn for large models
            if model_size_bytes > 8 * 1024 ** 3:  # > 8GB models
                result["warnings"].append(
                    f"Large model (~{size_estimate}). Ensure you have sufficient RAM."
                )

    return result


# Common model sizes (approximate) for user reference
OLLAMA_MODEL_SIZES = {
    "llama3.2": "2.0 GB",
    "llama3.2:1b": "1.3 GB",
    "llama3.2:3b": "2.0 GB",
    "llama3.1": "4.7 GB",
    "llama3.1:8b": "4.7 GB",
    "llama3.1:70b": "40 GB",
    "llama3": "4.7 GB",
    "llama2": "3.8 GB",
    "mistral": "4.1 GB",
    "mixtral": "26 GB",
    "phi3": "2.2 GB",
    "phi3:mini": "2.2 GB",
    "gemma": "5.0 GB",
    "gemma:2b": "1.7 GB",
    "gemma:7b": "5.0 GB",
    "gemma2": "5.4 GB",
    "gemma2:2b": "1.6 GB",
    "gemma2:9b": "5.4 GB",
    "gemma2:27b": "16 GB",
    "qwen2.5": "4.7 GB",
    "qwen2.5:0.5b": "397 MB",
    "qwen2.5:1.5b": "986 MB",
    "qwen2.5:3b": "1.9 GB",
    "qwen2.5:7b": "4.7 GB",
    "deepseek-r1": "4.7 GB",
    "codellama": "3.8 GB",
    "codegemma": "5.0 GB",
    "nomic-embed-text": "274 MB",
}


def get_ollama_model_size_estimate(model: str) -> str:
    """
    Get estimated download size for an Ollama model.

    Args:
        model: Model name

    Returns:
        Human-readable size estimate or "unknown"
    """
    model_lower = model.lower()

    # Check exact match first
    if model_lower in OLLAMA_MODEL_SIZES:
        return OLLAMA_MODEL_SIZES[model_lower]

    # Check base model name (without tag)
    base_model = model_lower.split(":")[0]
    if base_model in OLLAMA_MODEL_SIZES:
        return OLLAMA_MODEL_SIZES[base_model]

    return "unknown"


def pull_ollama_model(model: str, host: str = "localhost", port: int = 11434, auto_confirm: bool = False) -> bool:
    """
    Pull/download a model in Ollama.

    Args:
        model: Model name to pull (e.g., "llama3.2", "mistral")
        host: Hostname where Ollama is running
        port: Port number
        auto_confirm: If True, skip confirmation prompt

    Returns:
        True if model was pulled successfully, False otherwise
    """
    # Get size estimate and check system resources
    size_estimate = get_ollama_model_size_estimate(model)
    resources = check_system_resources(model)

    print(f"\n{'='*60}")
    print(f"  Model '{model}' not found locally")
    print(f"{'='*60}")
    print(f"  Model size:      {size_estimate}")
    if resources["details"].get("estimated_ram"):
        print(f"  RAM required:    ~{resources['details']['estimated_ram']}")
    if resources["details"].get("free_disk_space"):
        print(f"  Free disk space: {resources['details']['free_disk_space']}")
    if resources["details"].get("total_ram"):
        print(f"  System RAM:      {resources['details']['total_ram']}")

    # Show warnings
    if resources["warnings"]:
        print(f"\n  {'!'*50}")
        for warning in resources["warnings"]:
            print(f"  Warning: {warning}")
        print(f"  {'!'*50}")

    # Block if can't download
    if not resources["can_download"]:
        print(f"\n  Cannot download: insufficient disk space.")
        print(f"  Free up disk space and try again.")
        return False

    # Warn but allow if can't run (user might want to try anyway)
    if not resources["can_run"]:
        print(f"\n  Warning: Model may not run on this system.")
        print(f"  Consider a smaller model variant (e.g., '{model}:1b' or '{model}:3b').")

    print(f"{'='*60}")

    # Ask for confirmation
    if not auto_confirm:
        try:
            if not resources["can_run"]:
                prompt = f"\n  Download anyway? [y/N]: "
            else:
                prompt = f"\n  Download '{model}'? [y/N]: "
            response = input(prompt).strip().lower()
            if response not in ['y', 'yes']:
                print("  Download cancelled.")
                return False
        except (EOFError, KeyboardInterrupt):
            print("\n  Download cancelled.")
            return False

    print(f"\n  Downloading from Ollama registry...")
    print(f"  (Press Ctrl+C to cancel)\n")

    try:
        # Ollama pull endpoint streams the response
        response = requests.post(
            f"http://{host}:{port}/api/pull",
            json={"name": model},
            stream=True,
            timeout=None  # No timeout - large models can take a while
        )

        if response.status_code != 200:
            print(f"Failed to pull model: HTTP {response.status_code}")
            return False

        # Process streaming response to show progress
        last_status = ""
        total_size_shown = False

        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    status = data.get("status", "")

                    # Show progress for downloads
                    if "completed" in data and "total" in data:
                        completed = data["completed"]
                        total = data["total"]
                        pct = (completed / total * 100) if total > 0 else 0

                        # Show actual total size on first progress update
                        if not total_size_shown and total > 0:
                            print(f"  Actual size: {_format_bytes(total)}")
                            total_size_shown = True

                        print(f"\r  {status}: {pct:.1f}% ({_format_bytes(completed)}/{_format_bytes(total)})", end="", flush=True)
                    elif status != last_status:
                        if last_status and "completed" in str(last_status):
                            print()  # newline after progress bar
                        print(f"  {status}")
                        last_status = status

                    # Check for errors
                    if "error" in data:
                        print(f"\n  Error: {data['error']}")
                        return False

                except json.JSONDecodeError:
                    continue

        print(f"\n  Model '{model}' downloaded successfully!")
        return True

    except KeyboardInterrupt:
        print(f"\n\n  Download cancelled by user.")
        return False
    except requests.exceptions.Timeout:
        print(f"\n  Timeout while downloading model '{model}'.")
        print(f"  Try again or download manually: ollama pull {model}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"\n  Error pulling model: {e}")
        return False


# =============================================================================
# JSON Schema Functions
# =============================================================================

def build_json_schema(categories: list, include_additional_properties: bool = True) -> dict:
    """Build a JSON schema for the classification output.

    Args:
        categories: List of category names
        include_additional_properties: If True, includes additionalProperties: false
                                       (required by OpenAI strict mode, not supported by Google)
    """
    properties = {}
    for i, cat in enumerate(categories, 1):
        properties[str(i)] = {
            "type": "string",
            "enum": ["0", "1"],
            "description": cat,
        }

    schema = {
        "type": "object",
        "properties": properties,
        "required": list(properties.keys()),
    }

    if include_additional_properties:
        schema["additionalProperties"] = False

    return schema


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


def validate_classification_json(json_str: str, num_categories: int) -> tuple[bool, dict | None]:
    """
    Validate that a JSON string contains valid classification output.

    Args:
        json_str: The JSON string to validate
        num_categories: Expected number of categories

    Returns:
        tuple: (is_valid, parsed_dict or None)
    """
    try:
        parsed = json.loads(json_str)

        if not isinstance(parsed, dict):
            return False, None

        # Check that all expected keys are present and values are "0" or "1"
        for i in range(1, num_categories + 1):
            key = str(i)
            if key not in parsed:
                return False, None
            val = str(parsed[key]).strip()
            if val not in ("0", "1"):
                return False, None

        # Normalize values to strings
        normalized = {str(i): str(parsed[str(i)]).strip() for i in range(1, num_categories + 1)}
        return True, normalized

    except (json.JSONDecodeError, KeyError, TypeError):
        return False, None


def ollama_two_step_classify(
    client,
    response_text: str,
    categories: list,
    categories_str: str,
    survey_question: str = "",
    creativity: float = None,
    max_retries: int = 5,
) -> tuple[str, str | None]:
    """
    Two-step classification for Ollama models.

    Step 1: Classify the response (natural language output OK)
    Step 2: Convert classification to strict JSON format

    This approach is more reliable for local models that struggle with
    simultaneous reasoning and JSON formatting.

    Args:
        client: UnifiedLLMClient instance
        response_text: The survey response to classify
        categories: List of category names
        categories_str: Pre-formatted category string
        survey_question: Optional context
        creativity: Temperature setting
        max_retries: Number of retry attempts for JSON validation

    Returns:
        tuple: (json_string, error_message or None)
    """
    num_categories = len(categories)
    survey_context = f"A respondent was asked: {survey_question}." if survey_question else ""

    # ==========================================================================
    # Step 1: Classification (natural language - focus on accuracy)
    # ==========================================================================
    step1_messages = [
        {
            "role": "system",
            "content": "You are an expert at categorizing survey responses. Focus on accurate classification."
        },
        {
            "role": "user",
            "content": f"""{survey_context}

Analyze this survey response and determine which categories apply:

Response: "{response_text}"

Categories:
{categories_str}

For each category, explain briefly whether it applies (YES) or not (NO) to this response.
Format your answer as:
1. [Category name]: YES/NO - [brief reason]
2. [Category name]: YES/NO - [brief reason]
...and so on for all categories."""
        }
    ]

    step1_reply, step1_error = client.complete(
        messages=step1_messages,
        json_schema=None,  # No JSON requirement for step 1
        creativity=creativity,
    )

    if step1_error:
        return '{"1":"e"}', f"Step 1 failed: {step1_error}"

    # ==========================================================================
    # Step 2: JSON Formatting with validation and retry
    # ==========================================================================
    example_json = json.dumps({str(i): "0" for i in range(1, num_categories + 1)})

    for attempt in range(max_retries):
        step2_messages = [
            {
                "role": "system",
                "content": "You convert classification results to JSON. Output ONLY valid JSON, nothing else."
            },
            {
                "role": "user",
                "content": f"""Convert this classification to JSON format.

Classification results:
{step1_reply}

Rules:
- Output ONLY a JSON object, no other text
- Use category numbers as keys (1, 2, 3, etc.)
- Use "1" if the category was marked YES, "0" if NO
- Include ALL {num_categories} categories

Example format:
{example_json}

Your JSON output:"""
            }
        ]

        step2_reply, step2_error = client.complete(
            messages=step2_messages,
            json_schema=None,  # Ollama doesn't support strict schema anyway
            creativity=0.1,  # Low temperature for formatting task
        )

        if step2_error:
            if attempt < max_retries - 1:
                continue
            return '{"1":"e"}', f"Step 2 failed: {step2_error}"

        # Extract and validate JSON
        extracted = extract_json(step2_reply)
        is_valid, normalized = validate_classification_json(extracted, num_categories)

        if is_valid:
            return json.dumps(normalized), None

        # If invalid, try again with more explicit instructions
        if attempt < max_retries - 1:
            step1_reply = f"""Previous attempt produced invalid JSON.

Original classification:
{step1_reply}

Please be more careful to output EXACTLY {num_categories} categories numbered 1 through {num_categories}."""

    # All retries exhausted - try to salvage what we can
    extracted = extract_json(step2_reply) if step2_reply else '{"1":"e"}'
    return extracted, f"JSON validation failed after {max_retries} attempts"


# =============================================================================
# Category Exploration Functions
# =============================================================================

def explore_corpus(
    survey_question,
    survey_input,
    api_key: str = None,
    research_question=None,
    specificity="broad",
    categories_per_chunk=10,
    divisions=5,
    model: str = "gpt-4o",
    provider: str = "auto",
    creativity=None,
    filename="corpus_exploration.csv",
    focus: str = None,
):
    """
    Extract categories from survey corpus using LLM.

    Uses raw HTTP requests via UnifiedLLMClient - supports all providers.

    Args:
        survey_question: The survey question being analyzed
        survey_input: Series or list of survey responses
        api_key: API key for the LLM provider
        research_question: Optional research context
        specificity: "broad" or "specific" categories
        categories_per_chunk: Number of categories to extract per chunk
        divisions: Number of chunks to process
        model: Model name (e.g., "gpt-4o", "claude-3-haiku-20240307", "gemini-2.5-flash")
        provider: Provider name or "auto" to detect from model name
        creativity: Temperature setting
        filename: Output CSV filename (None to skip saving)
        focus: Optional focus instruction for category extraction (e.g., "decisions to move",
               "emotional responses", "financial considerations"). When provided, the model
               will prioritize extracting categories related to this focus.

    Returns:
        DataFrame with extracted categories and counts
    """
    # Detect provider
    provider = detect_provider(model, provider)

    # Validate api_key
    if provider != "ollama" and not api_key:
        raise ValueError(f"api_key is required for provider '{provider}'")

    print(f"Exploring categories for question: '{survey_question}'")
    print(f"Using provider: {provider}, model: {model}")
    if focus:
        print(f"Focus: {focus}")
    print(f"          {categories_per_chunk * divisions} unique categories to be extracted.")
    print()

    # Input normalization
    if not isinstance(survey_input, pd.Series):
        survey_input = pd.Series(survey_input)
    survey_input = survey_input.dropna()

    n = len(survey_input)
    if n == 0:
        raise ValueError("survey_input is empty after dropping NA.")

    # Auto-adjust divisions for small datasets
    original_divisions = divisions
    divisions = min(divisions, max(1, n // 3))
    if divisions != original_divisions:
        print(f"Auto-adjusted divisions from {original_divisions} to {divisions} for {n} responses.")

    chunk_size = int(round(max(1, n / divisions), 0))

    if chunk_size < (categories_per_chunk / 2):
        old_categories_per_chunk = categories_per_chunk
        categories_per_chunk = max(3, chunk_size * 2)
        print(f"Auto-adjusted categories_per_chunk from {old_categories_per_chunk} to {categories_per_chunk} for chunk size {chunk_size}.")

    # Initialize unified client
    client = UnifiedLLMClient(provider=provider, api_key=api_key, model=model)

    # Build system message
    if research_question:
        system_content = (
            f"You are a helpful assistant that extracts categories from survey responses. "
            f"The specific task is to identify {specificity} categories of responses to a survey question. "
            f"The research question is: {research_question}"
        )
    else:
        system_content = "You are a helpful assistant that extracts categories from survey responses."

    # Sample chunks
    random_chunks = []
    for i in range(divisions):
        chunk = survey_input.sample(n=chunk_size).tolist()
        random_chunks.append(chunk)

    responses = []
    responses_list = []

    for i in tqdm(range(divisions), desc="Processing chunks"):
        survey_participant_chunks = '; '.join(str(x) for x in random_chunks[i])
        focus_text = f" Focus specifically on {focus}." if focus else ""
        prompt = (
            f'Identify {categories_per_chunk} {specificity} categories of responses to the question "{survey_question}" '
            f"in the following list of responses.{focus_text} Responses are each separated by a semicolon. "
            f"Responses are contained within triple backticks here: ```{survey_participant_chunks}``` "
            f"Number your categories from 1 through {categories_per_chunk} and be concise with the category labels and provide no description of the categories."
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ]

        reply, error = client.complete(
            messages=messages,
            creativity=creativity,
            force_json=False,  # Text response, not JSON
        )

        if error:
            if "context_length_exceeded" in str(error) or "maximum context length" in str(error):
                raise ValueError(
                    f"Token limit exceeded for model {model}. "
                    f"Try increasing the 'divisions' parameter to create smaller chunks."
                )
            else:
                print(f"API error on chunk {i+1}: {error}")
                reply = ""

        responses.append(reply)

        # Extract just the text as a list
        items = []
        for line in (reply or "").split('\n'):
            if '. ' in line:
                try:
                    items.append(line.split('. ', 1)[1])
                except IndexError:
                    pass

        responses_list.append(items)

    flat_list = [item.lower() for sublist in responses_list for item in sublist]

    if not flat_list:
        raise ValueError("No categories were extracted from the model responses.")

    df = pd.DataFrame(flat_list, columns=['Category'])
    counts = pd.Series(flat_list).value_counts()
    df['counts'] = df['Category'].map(counts)
    df = df.sort_values(by='counts', ascending=False).reset_index(drop=True)
    df = df.drop_duplicates(subset='Category', keep='first').reset_index(drop=True)

    if filename is not None:
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

    return df


def explore_common_categories(
    survey_input,
    api_key: str = None,
    survey_question: str = "",
    max_categories: int = 12,
    categories_per_chunk: int = 10,
    divisions: int = 5,
    model: str = "gpt-4o",
    provider: str = "auto",
    creativity: float = None,
    specificity: str = "broad",
    research_question: str = None,
    filename: str = None,
    iterations: int = 5,
    random_state: int = None,
    focus: str = None,
    progress_callback: callable = None,
    # Legacy parameter names for backward compatibility
    user_model: str = None,
    model_source: str = None,
):
    """
    Extract and rank common categories from survey corpus.

    Uses raw HTTP requests via UnifiedLLMClient - supports all providers.

    Args:
        survey_input: Series or list of survey responses
        api_key: API key for the LLM provider
        survey_question: The survey question being analyzed
        max_categories: Maximum number of top categories to return
        categories_per_chunk: Number of categories to extract per chunk
        divisions: Number of chunks to process per iteration
        model: Model name (e.g., "gpt-4o", "claude-3-haiku-20240307", "gemini-2.5-flash")
        provider: Provider name or "auto" to detect from model name
        creativity: Temperature setting
        specificity: "broad" or "specific" categories
        research_question: Optional research context
        filename: Output CSV filename (None to skip saving)
        iterations: Number of passes over the data
        random_state: Random seed for reproducibility
        focus: Optional focus instruction for category extraction (e.g., "decisions to move",
               "emotional responses", "financial considerations"). When provided, the model
               will prioritize extracting categories related to this focus.
        progress_callback: Optional callback function for progress updates.
            Called as progress_callback(current_step, total_steps, step_label).

    Returns:
        dict with 'counts_df', 'top_categories', and 'raw_top_text'
    """
    import re
    import numpy as np

    # Handle legacy parameter names
    if user_model is not None:
        model = user_model
    if model_source is not None:
        provider = model_source

    # Detect provider
    provider = detect_provider(model, provider)

    # Validate api_key
    if provider != "ollama" and not api_key:
        raise ValueError(f"api_key is required for provider '{provider}'")

    # Input normalization
    if not isinstance(survey_input, pd.Series):
        survey_input = pd.Series(survey_input)
    survey_input = survey_input.dropna().astype("string")
    n = len(survey_input)
    if n == 0:
        raise ValueError("survey_input is empty after dropping NA.")

    # Auto-adjust divisions for small datasets
    original_divisions = divisions
    divisions = min(divisions, max(1, n // 3))
    if divisions != original_divisions:
        print(f"Auto-adjusted divisions from {original_divisions} to {divisions} for {n} responses.")

    # Chunk sizing
    chunk_size = int(round(max(1, n / divisions), 0))
    if chunk_size < (categories_per_chunk / 2):
        old_categories_per_chunk = categories_per_chunk
        categories_per_chunk = max(3, chunk_size * 2)
        print(f"Auto-adjusted categories_per_chunk from {old_categories_per_chunk} to {categories_per_chunk} for chunk size {chunk_size}.")

    print(f"Exploring categories for question: '{survey_question}'")
    print(f"Using provider: {provider}, model: {model}")
    if focus:
        print(f"Focus: {focus}")
    print(f"          {categories_per_chunk * divisions * iterations} total category extractions across {iterations} iterations.")
    print(f"          Top {max_categories} categories will be identified.\n")

    # RNG for reproducible re-sampling across passes
    rng = np.random.default_rng(random_state)

    # Initialize unified client
    client = UnifiedLLMClient(provider=provider, api_key=api_key, model=model)

    # Build system message
    if research_question:
        system_content = (
            f"You are a helpful assistant that extracts categories from survey responses. "
            f"The specific task is to identify {specificity} categories of responses to a survey question. "
            f"The research question is: {research_question}"
        )
    else:
        system_content = "You are a helpful assistant that extracts categories from survey responses."

    def make_prompt(responses_blob: str) -> str:
        focus_text = f" Focus specifically on {focus}." if focus else ""
        return (
            f'Identify {categories_per_chunk} {specificity} categories of responses to the question "{survey_question}" '
            f"in the following list of responses.{focus_text} Responses are separated by semicolons. "
            f"Responses are within triple backticks: ```{responses_blob}``` "
            f"Number your categories from 1 through {categories_per_chunk} and provide concise labels only (no descriptions)."
        )

    # Parse numbered list
    line_pat = re.compile(r"^\s*\d+\s*[\.\)\-]\s*(.+)$")

    all_items = []

    # Calculate total steps for progress tracking: (iterations * divisions) + 1 for final merge
    total_steps = (iterations * divisions) + 1
    current_step = 0

    for pass_idx in range(iterations):
        random_chunks = []
        for _ in range(divisions):
            seed = int(rng.integers(0, 2**32 - 1))
            chunk = survey_input.sample(n=chunk_size, random_state=seed).tolist()
            random_chunks.append(chunk)

        for i in tqdm(range(divisions), desc=f"Processing chunks (pass {pass_idx+1}/{iterations})"):
            survey_participant_chunks = "; ".join(str(x) for x in random_chunks[i])
            prompt = make_prompt(survey_participant_chunks)

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ]

            reply, error = client.complete(
                messages=messages,
                creativity=creativity,
                force_json=False,  # Text response, not JSON
            )

            if error:
                raise RuntimeError(
                    f"Model call failed on pass {pass_idx+1}, chunk {i+1}: {error}"
                )

            items = []
            for raw_line in (reply or "").splitlines():
                m = line_pat.match(raw_line.strip())
                if m:
                    items.append(m.group(1).strip())
            if not items:
                for raw_line in (reply or "").splitlines():
                    s = raw_line.strip()
                    if s:
                        items.append(s)

            all_items.extend(items)

            # Progress callback
            current_step += 1
            if progress_callback:
                progress_callback(current_step, total_steps, f"Pass {pass_idx+1}/{iterations}, chunk {i+1}/{divisions}")

    # Normalize and count
    def normalize_category(cat):
        terms = sorted([t.strip().lower() for t in str(cat).split("/")])
        return "/".join(terms)

    flat_list = [str(x).strip() for x in all_items if str(x).strip()]
    if not flat_list:
        raise ValueError("No categories were extracted from the model responses.")

    df = pd.DataFrame(flat_list, columns=["Category"])
    df["normalized"] = df["Category"].map(normalize_category)

    result = (
        df.groupby("normalized")
          .agg(Category=("Category", lambda x: x.value_counts().index[0]),
               counts=("Category", "size"))
          .sort_values("counts", ascending=False)
          .reset_index(drop=True)
    )

    # Second-pass semantic merge prompt
    seed_list = result["Category"].head(max_categories * 3).tolist()

    second_prompt = f"""
You are a data analyst reviewing categorized survey data.

Task: From the provided categories, identify and return the top {max_categories} CONCEPTUALLY UNIQUE categories.

Critical Instructions:
1) Exact duplicates are already removed.
2) Merge SEMANTIC duplicates (same concept, different wording). Examples:
   - "closer to work" = "commute/proximity to work"
   - "breakup/household conflict" = "relationship problems"
3) When merging:
   - Combine frequencies mentally
   - Keep the most frequent OR clearest label
   - Each concept appears ONLY ONCE
4) Keep category names {specificity}.
5) Return ONLY a numbered list of {max_categories} categories. No extra text.

Pre-processed Categories (sorted by frequency, top sample):
{seed_list}

Output:
1. category
2. category
...
{max_categories}. category
""".strip()

    # Second pass call
    reply2, error2 = client.complete(
        messages=[{"role": "user", "content": second_prompt}],
        creativity=creativity,
        force_json=False,  # Text response
    )

    # Final progress callback for the merge step
    if progress_callback:
        progress_callback(total_steps, total_steps, "Merging categories")

    if error2:
        print(f"Warning: Second pass failed: {error2}")
        top_categories_text = ""
    else:
        top_categories_text = reply2 or ""

    final = []
    for line in top_categories_text.splitlines():
        m = line_pat.match(line.strip())
        if m:
            final.append(m.group(1).strip())
    if not final:
        final = [l.strip("-* ").strip() for l in top_categories_text.splitlines() if l.strip()]

    # Fallback to counts_df if second pass failed
    if not final:
        final = result["Category"].head(max_categories).tolist()

    print("\nTop categories:\n" + "\n".join(f"{i+1}. {c}" for i, c in enumerate(final[:max_categories])))

    if filename:
        result.to_csv(filename, index=False)
        print(f"\nResults saved to {filename}")

    return {
        "counts_df": result,
        "top_categories": final[:max_categories],
        "raw_top_text": top_categories_text
    }


# =============================================================================
# Main Classification Function
# =============================================================================

def multi_class(
    survey_input,
    categories,
    api_key: str = None,
    model: str = "gpt-4o",
    provider: str = "auto",
    survey_question: str = "",
    example1: str = None,
    example2: str = None,
    example3: str = None,
    example4: str = None,
    example5: str = None,
    example6: str = None,
    creativity: float = None,
    safety: bool = False,
    chain_of_verification: bool = False,
    chain_of_thought: bool = True,
    step_back_prompt: bool = False,
    context_prompt: bool = False,
    thinking_budget: int = 0,
    max_categories: int = 12,
    categories_per_chunk: int = 10,
    divisions: int = 10,
    research_question: str = None,
    use_json_schema: bool = True,
    filename: str = None,
    save_directory: str = None,
    auto_download: bool = False,
):
    """
    Multi-class text classification using a unified HTTP-based approach.

    This function uses raw HTTP requests for all providers, eliminating SDK dependencies.
    Supports multiple prompting strategies including chain-of-thought, chain-of-verification,
    step-back prompting, and context prompting.

    Args:
        survey_input: List or Series of text responses to classify
        categories: List of category names, or "auto" to auto-detect categories
        api_key: API key for the LLM provider (not required for Ollama)
        model: Model name (e.g., "gpt-4o", "claude-sonnet-4-5-20250929", "gemini-2.5-flash",
               or any Ollama model like "llama3.2", "mistral", "phi3")
        provider: Provider name or "auto" to detect from model name.
                  For local models, use provider="ollama"
        survey_question: Optional context about what question was asked
        example1-6: Optional few-shot examples for classification
        creativity: Temperature setting (None for provider default)
        safety: If True, saves results incrementally during processing
        chain_of_verification: If True, uses 4-step CoVe prompting for verification
        chain_of_thought: If True, uses step-by-step reasoning in prompt
        step_back_prompt: If True, first asks about underlying factors before classifying
        context_prompt: If True, adds expert context prefix to prompts
        thinking_budget: Token budget for Google's extended thinking (0 to disable)
        max_categories: Maximum categories when using auto-detection
        categories_per_chunk: Categories per chunk for auto-detection
        divisions: Number of divisions for auto-detection
        research_question: Research context for auto-detection
        use_json_schema: Whether to use strict JSON schema (vs just json_object mode)
        filename: Optional CSV filename to save results
        save_directory: Optional directory for safety saves
        auto_download: If True, automatically download missing Ollama models

    Returns:
        DataFrame with classification results

    Example with Ollama (local):
        results = multi_class(
            survey_input=["I moved for work"],
            categories=["Employment", "Family"],
            model="llama3.2",
            provider="ollama",
        )

    Example with cloud provider:
        results = multi_class(
            survey_input=["I moved for work"],
            categories=["Employment", "Family"],
            api_key="your-api-key",
            model="gpt-4o",
        )

    Example with chain-of-verification:
        results = multi_class(
            survey_input=["I moved for work"],
            categories=["Employment", "Family"],
            api_key="your-api-key",
            model="gpt-4o",
            chain_of_verification=True,
            survey_question="Why did you move?",
        )

    .. deprecated::
        Use :func:`catllm.classify` instead. This function will be removed in a future version.
    """
    warnings.warn(
        "multi_class() is deprecated and will be removed in a future version. "
        "Use catllm.classify() instead, which supports single and multi-model classification.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Detect provider
    provider = detect_provider(model, provider)

    # Validate api_key requirement
    if provider != "ollama" and not api_key:
        raise ValueError(f"api_key is required for provider '{provider}'")

    # Handle categories="auto" - auto-detect categories from the data
    if categories == "auto":
        if survey_question == "":
            raise TypeError("survey_question is required when using categories='auto'. Please provide the survey question you are analyzing.")

        categories = explore_common_categories(
            survey_question=survey_question,
            survey_input=survey_input,
            research_question=research_question,
            api_key=api_key,
            model_source=provider,
            user_model=model,
            max_categories=max_categories,
            categories_per_chunk=categories_per_chunk,
            divisions=divisions
        )

    # Build examples text for few-shot prompting
    examples = [example1, example2, example3, example4, example5, example6]
    examples_text = "\n".join(
        f"Example {i}: {ex}" for i, ex in enumerate(examples, 1) if ex is not None
    )

    # Survey question context
    survey_question_context = f"A respondent was asked: {survey_question}." if survey_question else ""

    # Step-back insight initialization
    stepback_insight = None
    step_back_added = False
    if step_back_prompt:
        if survey_question == "":
            raise TypeError("survey_question is required when using step_back_prompt. Please provide the survey question you are analyzing.")

        stepback_question = f'What are the underlying factors or dimensions that explain how people typically answer "{survey_question}"?'
        stepback_insight, step_back_added = _get_stepback_insight(
            provider, stepback_question, api_key, model, creativity
        )

    # Ollama-specific checks
    if provider == "ollama":
        if not check_ollama_running():
            raise ConnectionError(
                "\n" + "="*60 + "\n"
                "  OLLAMA NOT RUNNING\n"
                "="*60 + "\n\n"
                "Ollama must be running to use local models.\n\n"
                "To start Ollama:\n"
                "  macOS:   Open the Ollama app, or run 'ollama serve'\n"
                "  Linux:   Run 'ollama serve' in terminal\n"
                "  Windows: Open the Ollama app\n\n"
                "Don't have Ollama installed?\n"
                "  Download from: https://ollama.ai/download\n\n"
                "After starting Ollama, run your code again.\n"
                + "="*60
            )

        # Check system resources before proceeding
        resources = check_system_resources(model)

        # Check if model needs to be downloaded
        model_installed = check_ollama_model(model)

        if not model_installed:
            if not pull_ollama_model(model, auto_confirm=auto_download):
                raise RuntimeError(
                    f"Model '{model}' not available. "
                    f"To download manually: ollama pull {model}"
                )
        else:
            # Model is installed - still check if it can run
            if resources["warnings"] or not resources["can_run"]:
                print(f"\n{'='*60}")
                print(f"  Model '{model}' - System Resource Check")
                print(f"{'='*60}")
                size_estimate = get_ollama_model_size_estimate(model)
                print(f"  Model size:      {size_estimate}")
                if resources["details"].get("estimated_ram"):
                    print(f"  RAM required:    ~{resources['details']['estimated_ram']}")
                if resources["details"].get("total_ram"):
                    print(f"  System RAM:      {resources['details']['total_ram']}")

                if resources["warnings"]:
                    print(f"\n  {'!'*50}")
                    for warning in resources["warnings"]:
                        print(f"  Warning: {warning}")
                    print(f"  {'!'*50}")

                if not resources["can_run"]:
                    print(f"\n  Warning: Model may not run well on this system.")
                    print(f"  Consider a smaller variant (e.g., '{model}:1b' or '{model}:3b').")
                    print(f"{'='*60}")

                    if not auto_download:
                        try:
                            response = input(f"\n  Continue anyway? [y/N]: ").strip().lower()
                            if response not in ['y', 'yes']:
                                raise RuntimeError(
                                    f"Model '{model}' may be too large for this system. "
                                    f"Try a smaller variant like '{model}:3b' or '{model}:1b'."
                                )
                        except (EOFError, KeyboardInterrupt):
                            raise RuntimeError("Operation cancelled by user.")

                print()

    print(f"Using provider: {provider}, model: {model}")

    # Initialize client
    client = UnifiedLLMClient(provider=provider, api_key=api_key, model=model)

    # Build category string and schema
    categories_str = "\n".join(f"{i + 1}. {cat}" for i, cat in enumerate(categories))
    # Build JSON schema - Google doesn't support additionalProperties
    if use_json_schema:
        include_additional = (provider != "google")
        json_schema = build_json_schema(categories, include_additional_properties=include_additional)
    else:
        json_schema = None

    # Print categories
    print(f"\nCategories to classify ({len(categories)} total):")
    for i, cat in enumerate(categories, 1):
        print(f"  {i}. {cat}")
    print()

    # Build prompt template
    def build_prompt(response_text: str) -> tuple:
        """Build the classification prompt for a single response.

        Returns:
            tuple: (messages list, user_prompt string for CoVe)
        """
        if chain_of_thought:
            user_prompt = f"""{survey_question_context}

Categorize this survey response "{response_text}" into the following categories that apply:
{categories_str}

Let's think step by step:
1. First, identify the main themes mentioned in the response
2. Then, match each theme to the relevant categories
3. Finally, assign 1 to matching categories and 0 to non-matching categories

{examples_text}

Provide your answer in JSON format where the category number is the key and "1" if present, "0" if not."""
        else:
            user_prompt = f"""{survey_question_context}
Categorize this survey response "{response_text}" into the following categories that apply:
{categories_str}
{examples_text}
Provide your answer in JSON format where the category number is the key and "1" if present, "0" if not."""

        # Add context prompt prefix if enabled
        if context_prompt:
            context = """You are an expert researcher in survey data categorization.
Apply multi-label classification and base decisions on explicit and implicit meanings.
When uncertain, prioritize precision over recall.

"""
            user_prompt = context + user_prompt

        # Build messages list
        messages = []

        # Add step-back insight if available
        if step_back_prompt and step_back_added and stepback_insight:
            messages.append({"role": "user", "content": stepback_question})
            messages.append({"role": "assistant", "content": stepback_insight})

        messages.append({"role": "user", "content": user_prompt})

        return messages, user_prompt

    # Build chain of verification prompts
    def build_cove_prompts(prompt: str, response_text: str) -> tuple:
        """Build chain of verification prompts."""
        step2_prompt = f"""You provided this initial categorization:
<<INITIAL_REPLY>>

Original task: {prompt}

Generate a focused list of 3-5 verification questions to fact-check your categorization. Each question should:
- Be concise and specific (one sentence)
- Address a distinct aspect of the categorization
- Be answerable independently

Focus on verifying:
- Whether each category assignment is accurate
- Whether the categories match the criteria in the original task
- Whether there are any logical inconsistencies

Provide only the verification questions as a numbered list."""

        step3_prompt = f"""Answer the following verification question based on the survey response provided.

Survey response: {response_text}

Verification question: <<QUESTION>>

Provide a brief, direct answer (1-2 sentences maximum).

Answer:"""

        step4_prompt = f"""Original task: {prompt}
Initial categorization:
<<INITIAL_REPLY>>
Verification questions and answers:
<<VERIFICATION_QA>>
If no categories are present, assign "0" to all categories.
Provide the final corrected categorization in the same JSON format:"""

        return step2_prompt, step3_prompt, step4_prompt

    def remove_numbering(line: str) -> str:
        """Remove numbering/bullets from a line for CoVe question parsing."""
        line = line.strip()
        if line.startswith('- '):
            return line[2:].strip()
        if line.startswith('• '):
            return line[2:].strip()
        if line and line[0].isdigit():
            i = 0
            while i < len(line) and line[i].isdigit():
                i += 1
            if i < len(line) and line[i] in '.)':
                return line[i+1:].strip()
        return line

    def run_chain_of_verification(initial_reply: str, step2_prompt: str, step3_prompt: str, step4_prompt: str) -> str:
        """Run chain of verification using the unified client."""
        # Step 2: Generate verification questions (text response, not JSON)
        step2_filled = step2_prompt.replace("<<INITIAL_REPLY>>", initial_reply)
        questions_reply, err = client.complete(
            messages=[{"role": "user", "content": step2_filled}],
            creativity=creativity,
            force_json=False,  # Text response
        )
        if err:
            return initial_reply  # Fall back to initial reply on error

        # Parse questions
        questions = [remove_numbering(line) for line in questions_reply.strip().split('\n') if line.strip()]

        # Step 3: Answer each verification question (text responses)
        qa_pairs = []
        for question in questions[:5]:  # Limit to 5 questions
            step3_filled = step3_prompt.replace("<<QUESTION>>", question)
            answer_reply, err = client.complete(
                messages=[{"role": "user", "content": step3_filled}],
                creativity=creativity,
                force_json=False,  # Text response
            )
            if not err:
                qa_pairs.append(f"Q: {question}\nA: {answer_reply.strip()}")

        verification_qa = "\n\n".join(qa_pairs)

        # Step 4: Final corrected categorization (JSON response)
        step4_filled = step4_prompt.replace("<<INITIAL_REPLY>>", initial_reply).replace("<<VERIFICATION_QA>>", verification_qa)
        final_reply, err = client.complete(
            messages=[{"role": "user", "content": step4_filled}],
            json_schema=json_schema,
            creativity=creativity,
        )

        if err:
            return initial_reply
        return final_reply

    # Process each response
    results = []
    extracted_jsons = []

    # Use two-step approach for Ollama (more reliable JSON output)
    use_two_step = (provider == "ollama")

    if use_two_step:
        print("Using two-step classification for Ollama (classify -> format JSON)")

    for idx, response in enumerate(tqdm(survey_input, desc="Classifying responses")):
        if pd.isna(response):
            results.append(("Skipped NaN", "Skipped NaN input"))
            extracted_jsons.append('{"1":"e"}')
            continue

        if use_two_step:
            json_result, error = ollama_two_step_classify(
                client=client,
                response_text=response,
                categories=categories,
                categories_str=categories_str,
                survey_question=survey_question,
                creativity=creativity,
                max_retries=5,
            )

            if error:
                results.append((json_result, error))
            else:
                results.append((json_result, None))
            extracted_jsons.append(json_result)

        else:
            messages, user_prompt = build_prompt(response)
            reply, error = client.complete(
                messages=messages,
                json_schema=json_schema,
                creativity=creativity,
                thinking_budget=thinking_budget if provider == "google" else None,
            )

            if error:
                results.append((None, error))
                extracted_jsons.append('{"1":"e"}')
            else:
                # Apply chain of verification if enabled
                if chain_of_verification and reply:
                    step2, step3, step4 = build_cove_prompts(user_prompt, response)
                    reply = run_chain_of_verification(reply, step2, step3, step4)

                results.append((reply, None))
                extracted_jsons.append(extract_json(reply))

        # Safety incremental save
        if safety:
            if filename is None:
                raise TypeError("filename is required when using safety=True. Please provide a filename to save to.")

            # Build partial DataFrame and save
            normalized_partial = []
            for json_str in extracted_jsons:
                try:
                    parsed = json.loads(json_str)
                    normalized_partial.append(pd.json_normalize(parsed))
                except json.JSONDecodeError:
                    normalized_partial.append(pd.DataFrame({"1": ["e"]}))

            if normalized_partial:
                normalized_df = pd.concat(normalized_partial, ignore_index=True)
                partial_df = pd.DataFrame({
                    'survey_input': pd.Series(survey_input[:len(results)]).reset_index(drop=True),
                    'model_response': [r[0] for r in results],
                    'error': [r[1] for r in results],
                    'json': pd.Series(extracted_jsons).reset_index(drop=True),
                })
                partial_df = pd.concat([partial_df, normalized_df], axis=1)
                partial_df = partial_df.rename(columns=lambda x: f'category_{x}' if str(x).isdigit() else x)

                save_path = filename
                if save_directory:
                    import os
                    os.makedirs(save_directory, exist_ok=True)
                    save_path = os.path.join(save_directory, filename)
                partial_df.to_csv(save_path, index=False)

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


# Note: For the legacy implementation with chain_of_verification, step_back_prompt,
# and other advanced features, see text_functions_old.py
