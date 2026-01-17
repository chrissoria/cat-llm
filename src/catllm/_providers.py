"""
Unified LLM provider infrastructure for CatLLM.

This module provides a unified HTTP-based approach for calling multiple LLM providers
(OpenAI, Anthropic, Google, Mistral, xAI, Perplexity, HuggingFace, and Ollama)
without requiring provider-specific SDKs.
"""

import json
import time
import requests

__all__ = [
    # Main client
    "UnifiedLLMClient",
    "PROVIDER_CONFIG",
    # Provider detection
    "detect_provider",
    "_detect_model_source",
    "_detect_huggingface_endpoint",
    # Ollama utilities
    "set_ollama_endpoint",
    "check_ollama_running",
    "list_ollama_models",
    "check_ollama_model",
    "check_system_resources",
    "get_ollama_model_size_estimate",
    "pull_ollama_model",
    "OLLAMA_MODEL_SIZES",
]


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
# Provider Detection
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
