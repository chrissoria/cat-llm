---
name: summarize
description: Summarize text or PDF documents using LLMs. Use when the user wants to generate concise summaries of survey responses, documents, or any text data. Supports single-model and multi-model ensemble summarization.
---

# CatLLM Summarization

You are helping the user summarize data using the CatLLM Python package. This handles both text and PDF summarization with single or multiple models.

## Setup

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

# Single model text summarization
results = cat.summarize(
    input_data=df['responses'],
    description="Customer feedback",
    api_key="YOUR_API_KEY",
    instructions="Provide bullet-point summaries",  # optional
    max_length=100,                                  # max words
    focus="key complaints",                          # optional focus
)

# PDF summarization (auto-detected from file paths)
results = cat.summarize(
    input_data="/path/to/pdfs/",
    description="Research papers",
    mode="image",                                    # "image", "text", or "both"
    api_key="YOUR_API_KEY",
)

# Multi-model ensemble with synthesis
results = cat.summarize(
    input_data=df['responses'],
    models=[
        ("gpt-4o", "openai", "sk-..."),
        ("claude-sonnet-4-5-20250929", "anthropic", "sk-ant-..."),
    ],
)
```

## Key Parameters

- `input_data`: list of strings, pandas Series, single string, directory path, or PDF paths
- `description`: description of the content (provides context)
- `api_key`: API key string
- `instructions`: specific summarization instructions
- `max_length`: maximum summary length in words
- `focus`: what to focus on (e.g., "main arguments", "emotional content")
- `user_model`: model to use (default "gpt-4o")
- `model_source`: provider ("auto", "openai", "anthropic", "google", etc.)
- `mode`: PDF processing mode — "image", "text", or "both"
- `filename`: output CSV filename
- `save_directory`: where to save results
- `models`: list of tuples for multi-model mode

## Output

Returns a `pandas.DataFrame` with columns:
- `survey_input`: original text or page label (for PDFs)
- `summary`: generated summary (or consensus for multi-model)
- `processing_status`: "success", "error", or "skipped"
- `pdf_path`: source PDF path (PDF mode only)
- `page_index`: page number, 0-indexed (PDF mode only)

$ARGUMENTS
