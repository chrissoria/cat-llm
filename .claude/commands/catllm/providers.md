Check which LLM providers are configured in the current environment.

Usage: /catllm:providers

Instructions:
1. Run the following Python snippet to detect configured providers:

```python
import os

provider_keys = {
    "OPENAI_API_KEY":         ("OpenAI / xAI",      ["gpt-5", "gpt-4o-mini", "grok-3"]),
    "ANTHROPIC_API_KEY":      ("Anthropic",          ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"]),
    "GOOGLE_API_KEY":         ("Google",             ["gemini-2.0-flash", "gemini-1.5-pro"]),
    "MISTRAL_API_KEY":        ("Mistral",            ["mistral-large-latest", "mistral-small-latest"]),
    "PERPLEXITY_API_KEY":     ("Perplexity",         ["sonar-pro", "sonar"]),
    "HUGGINGFACE_API_TOKEN":  ("HuggingFace",        ["meta-llama/Llama-3.3-70B-Instruct"]),
}

print("=== cat-llm Provider Status ===\n")
configured = []
missing = []

for env_var, (provider, models) in provider_keys.items():
    val = os.environ.get(env_var, "")
    if val:
        masked = val[:4] + "..." + val[-4:] if len(val) > 8 else "***"
        print(f"[OK] {provider}")
        print(f"     Key: {env_var} = {masked}")
        print(f"     Models: {', '.join(models)}")
        configured.append(provider)
    else:
        missing.append(f"  - {provider} ({env_var})")

print(f"\nConfigured: {len(configured)} provider(s)")
if missing:
    print(f"\nNot configured:")
    for m in missing:
        print(m)
```

2. Also check for Ollama availability:

```python
import subprocess, sys
try:
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=3)
    if result.returncode == 0:
        print("\n[OK] Ollama (local)")
        lines = result.stdout.strip().split("\n")
        for line in lines[1:6]:  # show up to 5 models
            print(f"     {line}")
except Exception:
    print("\n[ ] Ollama not available")
```

3. Display the results in a clean, readable format.
4. If no providers are configured, suggest setting up a .env file with the required keys.

allowed-tools: Bash(python3*), Bash(env), Bash(ollama*)
