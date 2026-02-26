"""
Category analysis utilities for CatLLM.

Provides functions for analyzing user-provided category lists,
such as detecting whether an "Other" catch-all category exists.
"""

import re

from .text_functions import UnifiedLLMClient, detect_provider

__all__ = ["has_other_category"]

# Max words for a category to be checked against broad phrase patterns.
# Real catch-all categories are short ("Other", "None of the above", "Does not fit").
# Longer categories using these words ("Does not fit the clinical profile") are
# specific descriptive labels, not catch-alls.
_MAX_HEURISTIC_WORDS = 4

# Tier 1: Anchored patterns — safe at any category length.
# These only match when the keyword IS the category label itself.
_ANCHORED_PATTERNS = [
    re.compile(r"^other\s*$", re.IGNORECASE),         # exact "Other"
    re.compile(r"^other\s*[:(]", re.IGNORECASE),       # "Other: ...", "Other (..."
    re.compile(r"^n/?a\s*$", re.IGNORECASE),           # exact "N/A", "NA"
    re.compile(r"^miscellaneous\s*$", re.IGNORECASE),  # exact "Miscellaneous"
    re.compile(r"^catch[\s-]?all\s*$", re.IGNORECASE), # exact "catch-all"
]

# Tier 2: Phrase patterns — only applied to short categories (≤ _MAX_HEURISTIC_WORDS).
# Multi-word phrases that clearly signal a catch-all when they dominate the category name.
_SHORT_ONLY_PATTERNS = [
    re.compile(r"\bnone of the above\b", re.IGNORECASE),
    re.compile(r"\bdoes not fit\b", re.IGNORECASE),
    re.compile(r"\bdoesn't fit\b", re.IGNORECASE),
    re.compile(r"\bnot applicable\b", re.IGNORECASE),
    re.compile(r"\bnone apply\b", re.IGNORECASE),
    re.compile(r"\bnone of these\b", re.IGNORECASE),
]

# Top-tier model per provider for the LLM fallback
_TOP_TIER_MODELS = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-5-20250929",
    "google": "gemini-2.5-flash",
    "mistral": "mistral-large-latest",
    "xai": "grok-2",
    "perplexity": "sonar-pro",
    "huggingface": "meta-llama/Llama-3.3-70B-Instruct",
}


def _heuristic_check(categories: list) -> bool:
    """
    Fast, free check for common "Other" category patterns.

    Uses a two-tier approach to avoid false positives:
      - Tier 1 (anchored): matches at any length — the pattern is specific enough
        (e.g. exact "Other", "N/A", or "Other: …" label prefix).
      - Tier 2 (phrase): only matches short categories (≤ _MAX_HEURISTIC_WORDS words).
        Phrases like "does not fit" are catch-alls when they ARE the category, but
        not when embedded in longer descriptions ("Does not fit the clinical profile").

    Returns True if any category matches a known catch-all pattern.
    """
    for cat in categories:
        cat_str = str(cat).strip()

        # Tier 1: anchored patterns — safe at any length
        for pattern in _ANCHORED_PATTERNS:
            if pattern.search(cat_str):
                return True

        # Tier 2: phrase patterns — only for short categories
        if len(cat_str.split()) <= _MAX_HEURISTIC_WORDS:
            for pattern in _SHORT_ONLY_PATTERNS:
                if pattern.search(cat_str):
                    return True

    return False


def _llm_check(categories: list, api_key: str, model: str, provider: str) -> bool:
    """
    Use an LLM to determine whether the category list contains a catch-all.

    Makes a single API call and parses a yes/no answer.

    Returns True if the LLM judges a catch-all category exists, False otherwise.
    """
    cat_list = "\n".join(f"- {c}" for c in categories)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer with ONLY 'yes' or 'no', "
                "nothing else."
            ),
        },
        {
            "role": "user",
            "content": (
                "Does the following list of categories contain a catch-all or "
                "'Other' category — i.e., a category meant to capture responses "
                "that don't fit any of the specific categories?\n\n"
                f"Categories:\n{cat_list}\n\n"
                "Answer 'yes' or 'no'."
            ),
        },
    ]

    client = UnifiedLLMClient(provider=provider, api_key=api_key, model=model)
    response_text, error = client.complete(
        messages=messages,
        force_json=False,
        max_retries=2,
        creativity=0.0,
    )

    if error or not response_text:
        return False

    # Strip whitespace and punctuation, then check for affirmative answer
    answer = response_text.strip().lower().rstrip(".!,;:")
    return answer in ("yes", "true")


def has_other_category(
    categories: list,
    api_key: str = None,
    user_model: str = None,
    model_source: str = "auto",
) -> bool:
    """
    Detect whether a list of categories contains a catch-all / "Other" category.

    Uses a two-stage approach:
      1. **Heuristic** (free, instant) — checks for common patterns like "Other",
         "None of the above", "Miscellaneous", etc.
      2. **LLM fallback** (1 API call) — if the heuristic finds nothing and an
         ``api_key`` is provided, asks an LLM to judge whether a catch-all exists.

    Args:
        categories: List of category strings to analyze.
        api_key: Optional API key for the LLM fallback. If not provided and the
                 heuristic doesn't match, the function returns ``False``.
        user_model: Optional model name for the LLM fallback. If not provided,
                    a top-tier default model is selected based on the provider.
        model_source: Provider to use for the LLM fallback (e.g. "openai",
                      "anthropic", "google"). Defaults to "auto" which auto-detects
                      from ``user_model``, or falls back to "openai" when no model
                      is specified.

    Returns:
        ``True`` if a catch-all / "Other" category is detected, ``False`` otherwise.

    Examples:
        >>> has_other_category(["Positive", "Negative", "Other"])
        True

        >>> has_other_category(["Positive", "Negative"])
        False

        >>> has_other_category(
        ...     ["Happy", "Sad", "Doesn't fit any category"],
        ...     api_key="sk-...",
        ... )
        True
    """
    if not categories:
        return False

    # Stage 1: heuristic
    if _heuristic_check(categories):
        return True

    # Stage 2: LLM fallback (only if api_key provided)
    if api_key is None:
        return False

    # Resolve provider and model
    if user_model is not None:
        provider = detect_provider(user_model, provider=model_source)
        model = user_model
    else:
        # No model specified — pick a default
        if model_source and model_source.lower() != "auto":
            provider = model_source.lower()
        else:
            provider = "openai"
        model = _TOP_TIER_MODELS.get(provider, "gpt-4o")

    return _llm_check(categories, api_key, model, provider)
