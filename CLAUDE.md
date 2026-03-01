# CatLLM — Claude Code Context

CatLLM is a Python package for automating the categorization of open-ended survey responses, images, and PDFs using LLMs. Designed for survey research at scale with 98% accuracy compared to human consensus.

## Project Structure

```
src/catllm/              # Main package source
  __init__.py            # Public API exports
  classify.py            # Unified classify() entry point
  extract.py             # Unified extract() entry point
  explore.py             # Raw extraction for saturation analysis
  summarize.py           # Unified summarize() entry point
  _providers.py          # UnifiedLLMClient + PROVIDER_CONFIG (single source of truth)
  text_functions.py      # Text classification logic
  text_functions_ensemble.py  # Multi-model ensemble (heaviest module ~2700 lines)
  image_functions.py     # Image classification & features
  pdf_functions.py       # PDF processing & classification
  _category_analysis.py  # Category validation utilities
  _utils.py              # Shared utilities
  CERAD_functions.py     # CERAD neuropsychology scoring
  calls/                 # Prompt templates (leaf modules, no internal deps)
examples/                # Test scripts (require API keys)
hf_space/                # HuggingFace Space web app (Streamlit)
hf_summarizer/           # HuggingFace summarization app
r-package/               # R wrapper via reticulate
```

## Architecture

See `ARCHITECTURE.md` for the full module dependency map.

Key principle: `_providers.py` is the **single source of truth** for all provider infrastructure. `text_functions.py` re-exports its names for backward compatibility.

Call chain: `classify()` → `classify_ensemble()` → per-model `classify_single()` → `UnifiedLLMClient.complete()`

## Development Setup

```bash
pip install -e ".[dev]"
```

## Running Tests

Tests in `examples/` require real API keys. Set environment variables:
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AIza..."
```

Run a specific test:
```bash
python examples/test_all_providers.py
```

## Supported Providers

OpenAI, Anthropic, Google, HuggingFace, xAI, Mistral, Perplexity, Ollama (local).

## Public API

Four main entry points: `classify()`, `extract()`, `explore()`, `summarize()`

Plus: `image_score_drawing()`, `image_features()`, `cerad_drawn_score()`, `has_other_category()`, `check_category_verbosity()`

## Code Conventions

- Python 3.8+ compatibility
- All API calls go through `UnifiedLLMClient` in `_providers.py`
- Output is always `pandas.DataFrame` ready for statistical analysis
- `calls/` submodules are leaf nodes with no internal package dependencies
- New providers: add to `PROVIDER_CONFIG` dict in `_providers.py`
- Build system: Hatchling (`pyproject.toml`)

## Important Notes

- Never commit API keys or `.env` files
- The `openai` and `anthropic` pip packages are listed as dependencies but the code primarily uses direct HTTP via `requests` through `UnifiedLLMClient`
- Multi-modal: input type (text/image/PDF) is auto-detected
- Ensemble mode works by running classification across multiple models and using majority voting
