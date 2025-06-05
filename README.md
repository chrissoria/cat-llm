![catllm Logo](https://github.com/chrissoria/cat-llm/blob/main/images/logo.png?raw=True)

# catllm

[![PyPI - Version](https://img.shields.io/pypi/v/cat-llm.svg)](https://pypi.org/project/cat-llm)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cat-llm.svg)](https://pypi.org/project/cat-llm)

-----

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Supported Models](#supported-models)
- [API Reference](#api-reference)
- [Academic Research](#academic-research)
- [License](#license)

## Installation

```console
pip install cat-llm
```

## Quick Start

The `explore_corpus` function extracts a list of all categories present in the corpus as identified by the model.
```
import catllm as cat
import os

categories = cat.explore_corpus(
survey_question="What motivates you most at work?",
survey_input=["flexible schedule", "good pay", "interesting projects"],
api_key="OPENAI_API_KEY",
cat_num=5,
divisions=10
)
print(categories)
```

## Configuration

### Get Your OpenAI API Key

1. **Create an OpenAI Developer Account**:
   - Go to [platform.openai.com](https://platform.openai.com) (separate from regular ChatGPT)
   - Sign up with email, Google, Microsoft, or Apple

2. **Generate an API Key**:
   - Log into your account and click your name in the top right corner
   - Click "View API keys" or navigate to the "API keys" section
   - Click "Create new secret key"
   - Give your key a descriptive name
   - Set permissions (choose "All" for full access)

3. **Add Payment Details**:
   - Add a payment method to your OpenAI account
   - Purchase credits (start with $5 - it lasts a long time for most research use)
   - **Important**: Your API key won't work without credits

4. **Save Your Key Securely**:
   - Copy the key immediately (you won't be able to see it again)
   - Store it safely and never share it publicly

5. Copy and paste your key into catllm in the api_key parameter

## Supported Models

- **OpenAI**: GPT-4o, GPT-4, GPT-3.5-turbo, etc.
- **Anthropic**: Claude Sonnet 3.7, Claude Haiku, etc.
- **Perplexity**: Sonnar Large, Sonnar Small, etc.
- **Mistral**: Mistral Large, Mistral Small, etc.

## API Reference

### `explore_corpus()`

Extracts categories from a corpus of text responses and returns frequency counts.

**Methodology:**
The function divides the corpus into random chunks to address the probabilistic nature of LLM outputs. By processing multiple chunks and averaging results across many API calls rather than relying on a single call, this approach significantly improves reproducibility and provides more stable categorical frequency estimates.

**Parameters:**
- `survey_question` (str): The survey question being analyzed
- `survey_input` (list): List of text responses to categorize
- `api_key` (str): API key for the LLM service
- `cat_num` (int): Number of categories to extract in each iteration
- `divisions` (int): Number of chunks to divide the data into (larger corpora might require larger divisions)

**Returns:**
- `pandas.DataFrame`: Two-column dataset with category names and frequencies

## Academic Research

This package implements methodology from research on LLM performance in social science applications, including the UC Berkeley Social Networks Study. The package addresses reproducibility challenges in LLM-assisted research by providing standardized interfaces and consistent output formatting.

## License

`cat-llm` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
