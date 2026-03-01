---
name: classify
description: Classify text, images, or PDFs into categories using LLMs. Use when the user wants to categorize survey responses, label images, or classify PDF documents into predefined categories. Supports single-model and multi-model ensemble classification.
---

# CatLLM Classification

You are helping the user classify data using the CatLLM Python package (`cat-llm`). This skill handles text, image, and PDF classification into predefined categories.

## Setup

Make sure the package is installed:

```bash
pip install cat-llm
```

For PDF support:
```bash
pip install cat-llm[pdf]
```

## How to Use

```python
import catllm as cat

# Text classification (auto-detected from list/Series input)
results = cat.classify(
    input_data=df['responses'],           # list of strings or pandas Series
    categories=["Category A", "Category B", "Other"],
    description="Description of the survey question",
    api_key="YOUR_API_KEY",               # or use environment variable
    user_model="gpt-4o",                  # default model
)

# Image classification (auto-detected from file paths)
results = cat.classify(
    input_data="/path/to/images/",
    categories=["Contains person", "Outdoor scene"],
    description="Product photos",
    api_key="YOUR_API_KEY",
)

# PDF classification (auto-detected)
results = cat.classify(
    input_data="/path/to/pdfs/",
    categories=["Contains table", "Has chart"],
    description="Financial reports",
    mode="both",                          # "image", "text", or "both"
    api_key="YOUR_API_KEY",
)

# Multi-model ensemble for higher accuracy
results = cat.classify(
    input_data=df['responses'],
    categories=["Positive", "Negative", "Neutral"],
    models=[
        ("gpt-4o", "openai", "sk-..."),
        ("claude-sonnet-4-5-20250929", "anthropic", "sk-ant-..."),
        ("gemini-2.5-flash", "google", "AIza..."),
    ],
    consensus_threshold=0.5,
)
```

## Best Practices (Empirically Validated)

ALWAYS follow these when helping users set up classification:

1. **Write detailed category descriptions** — the single biggest accuracy lever. Instead of `"Job change"`, use `"The person had a job or school or career change, including transferred and retired."`
2. **Always include an "Other" category** — prevents forcing ambiguous responses into wrong categories
3. **Use low temperature** (`creativity=0`) — deterministic output is better for classification
4. **Do NOT enable chain_of_thought** — testing shows no benefit, slightly hurts some models
5. **Do NOT enable chain_of_verification** — 4x cost, reduces accuracy by 1-2pp
6. **Do NOT enable step_back_prompt** — inconsistent results, 2x cost
7. **Do NOT enable thinking_budget** unless specifically asked — high latency, frequent failures

## Key Parameters

- `input_data`: list, Series, directory path, or file paths
- `categories`: list of category names (use verbose descriptions)
- `api_key`: API key string
- `user_model`: "gpt-4o" (default), "claude-sonnet-4-5-20250929", "gemini-2.5-flash", etc.
- `model_source`: "auto" (default), "openai", "anthropic", "google", "huggingface", "xai", "mistral", "perplexity"
- `creativity`: 0.0-1.0 (recommend 0)
- `filename`: output CSV filename
- `save_directory`: where to save CSV
- `models`: list of tuples for ensemble mode
- `consensus_threshold`: 0-1 for ensemble voting (default 0.5)

## Output

Returns a `pandas.DataFrame` with binary columns per category (0/1). In ensemble mode, includes individual model predictions, consensus columns, and agreement scores.

## Supported Providers

OpenAI, Anthropic, Google, HuggingFace, xAI, Mistral, Perplexity, Ollama (local).

## Arguments

If the user provides arguments after `/catllm:classify`, treat them as a description of what they want to classify. Ask for any missing required information (data source, categories, API key).

$ARGUMENTS
