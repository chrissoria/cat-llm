"""
JSON formatter fallback for CatLLM.

Uses a fine-tuned Qwen2.5-0.5B model to convert messy LLM classification
output into valid cat-llm JSON format: {"1":"0","2":"1",...}

The formatter is opt-in via json_formatter=True on classify(). It only runs
when extract_json() produces invalid output — zero cost on the happy path.

Requires: pip install cat-llm[formatter]
"""

import sys

_MERGED_MODEL_REPO = "chrissoria/catllm-json-formatter"

_SYSTEM_PROMPT = (
    "You are a JSON formatter for a text classification pipeline. "
    "You will receive a list of categories (numbered 1 to N) and a raw "
    "classification output from another model. Your job is to convert that "
    'output into the exact JSON format required:\n'
    '{"1":"0","2":"1","3":"0",...}\n\n'
    "Rules:\n"
    '- Keys are 1-indexed strings: "1", "2", ..., "N"\n'
    '- Values are ONLY "0" (category absent) or "1" (category present)\n'
    "- Include ALL N categories, even if absent\n"
    "- Output ONLY the JSON object — no explanation, no markdown, no extra text\n"
    '- If a category\'s presence is ambiguous, default to "0"'
)


def _check_dependencies():
    """Check that torch and transformers are installed."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        raise ImportError(
            "The JSON formatter requires additional dependencies.\n"
            "Install them with: pip install cat-llm[formatter]\n"
            "  (requires: torch, transformers, accelerate)"
        )


def _is_model_cached() -> bool:
    """Check if the merged model is already in the HuggingFace cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
        result = try_to_load_from_cache(_MERGED_MODEL_REPO, "config.json")
        return result is not None and not isinstance(result, type(None))
    except Exception:
        return False


def ensure_formatter_available() -> bool:
    """
    Ensure the formatter model is available, prompting to download if needed.

    Returns:
        True if the formatter is ready to use, False if user declined download.
    """
    _check_dependencies()

    if _is_model_cached():
        return True

    print(
        "\n[CatLLM] The JSON formatter model (~1GB) will be downloaded from\n"
        f"  HuggingFace Hub ({_MERGED_MODEL_REPO}).\n"
        "  This is a one-time download — the model is cached locally after."
    )
    try:
        answer = input("  Continue? (Y/n): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "n"

    if answer in ("", "y", "yes"):
        return True
    else:
        print("  -> JSON formatter disabled for this run.\n")
        return False


def load_formatter(device=None):
    """
    Load the merged formatter model and tokenizer.

    Args:
        device: Target device. None = auto-detect (CUDA > CPU; MPS skipped).

    Returns:
        Tuple of (model, tokenizer, device_str).
    """
    _check_dependencies()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            # Skip MPS — known PEFT/generation crash issues
            device = "cpu"

    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[CatLLM] Loading JSON formatter on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(
        _MERGED_MODEL_REPO, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        _MERGED_MODEL_REPO, dtype=dtype, trust_remote_code=True
    )
    model = model.to(device)
    model.eval()

    print("[CatLLM] JSON formatter ready.")
    return model, tokenizer, device


def run_formatter(raw_output, categories, model, tokenizer, device):
    """
    Run the formatter model to fix malformed classification JSON.

    Args:
        raw_output: The raw (messy) output from the classification LLM.
        categories: List of category names.
        model: The loaded formatter model.
        tokenizer: The loaded tokenizer.
        device: Device string ("cuda" or "cpu").

    Returns:
        The formatter's output string (caller should run extract_json on it).
    """
    import torch

    # Build category list
    cat_lines = "\n".join(
        f"{i + 1}. {cat}" for i, cat in enumerate(categories)
    )
    user_msg = f"Categories:\n{cat_lines}\n\nRaw classification output:\n{raw_output}"

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only newly generated tokens
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
