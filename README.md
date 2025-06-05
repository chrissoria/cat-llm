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
- `cat_num` (int, default=10): Number of categories to extract in each iteration
- `divisions` (int, default=5): Number of chunks to divide the data into (larger corpora might require larger divisions)
- `specificity` (str, default="broad"): Category precision level (e.g., "broad", "narrow")
- `model_source` (str, default="OpenAI"): Model provider ("OpenAI", "Anthropic", "Perplexity", "Mistral")
- `user_model` (str, default="got-4o"): Specific model (e.g., "gpt-4o", "claude-opus-4-20250514")
- `creativity` (float, default=0): Temperature/randomness setting (0.0-1.0)
- `filename` (str, optional): Output file path for saving results

**Returns:**
- `pandas.DataFrame`: Two-column dataset with category names and frequencies

**Example:***

```
import catllm as cat

categories = cat.explore_corpus(
survey_question="What motivates you most at work?",
survey_input=["flexible schedule", "good pay", "interesting projects"],
api_key="OPENAI_API_KEY",
cat_num=5,
divisions=10
)
```

### `explore_common_categories()`

Identifies the most frequently occurring categories across a text corpus and returns the top N categories by frequency count.

**Methodology:**
Divides the corpus into random chunks and averages results across multiple API calls to improve reproducibility and provide stable frequency estimates for the most prevalent categories, addressing the probabilistic nature of LLM outputs.

**Parameters:**
- `survey_question` (str): Survey question being analyzed
- `survey_input` (list): Text responses to categorize
- `api_key` (str): API key for the LLM service
- `top_n` (int, default=10): Number of top categories to return by frequency
- `cat_num` (int, default=10): Number of categories to extract per iteration
- `divisions` (int, default=5): Number of data chunks (increase for larger corpora)
- `user_model` (str, default="gpt-4o"): Specific model to use
- `creativity` (float, default=0): Temperature/randomness setting (0.0-1.0)
- `specificity` (str, default="broad"): Category precision level ("broad", "narrow")
- `research_question` (str, optional): Contextual research question to guide categorization
- `filename` (str, optional): File path to save output dataset
- `model_source` (str, default="OpenAI"): Model provider ("OpenAI", "Anthropic", "Perplexity", "Mistral")

**Returns:**
- `pandas.DataFrame`: Dataset with category names and frequencies, limited to top N most common categories

**Example:**

```
import catllm as cat

top_10_categories = cat.explore_common_categories(
survey_question="What motivates you most at work?",
survey_input=["flexible schedule", "good pay", "interesting projects"],
api_key="OPENAI_API_KEY",
top_n=10,
cat_num=5,
divisions=10
)
print(categories)
```
### `multi_class()`

Performs multi-label classification of text responses into user-defined categories, returning structured results with optional CSV export.

**Methodology:**
Processes each text response individually, assigning one or more categories from the provided list. Supports flexible output formatting and optional saving of results to CSV for easy integration with data analysis workflows[2].

**Parameters:**
- `survey_question` (str): The survey question being analyzed
- `survey_input` (list): List of text responses to classify
- `categories` (list): List of predefined categories for classification
- `api_key` (str): API key for the LLM service
- `user_model` (str, default="gpt-4o"): Specific model to use
- `creativity` (float, default=0): Temperature/randomness setting (0.0-1.0)
- `safety` (bool, default=False): Enable safety checks on responses and saves to CSV at each API call step
- `filename` (str, default="categorized_data.csv"): Filename for CSV output
- `save_directory` (str, optional): Directory path to save the CSV file
- `model_source` (str, default="OpenAI"): Model provider ("OpenAI", "Anthropic", "Perplexity", "Mistral")

**Returns:**
- `pandas.DataFrame`: DataFrame with classification results, columns formatted as specified

**Example:**

```
import catllm as cat

user_categories = ["to start living with or to stay with partner/spouse",
                   "relationship change (divorce, breakup, etc)",
                   "the person had a job or school or career change, including transferred and retired",
                   "the person's partner's job or school or career change, including transferred and retired",
                   "financial reasons (rent is too expensive, pay raise, etc)",
                   "related specifically features of the home, such as a bigger or smaller yard"]

question = "Why did you move?"                   

move_reasons = cat.multi_class(
    survey_question=question, 
    survey_input= df[column1], 
    user_model="gpt-4o",
    creativity=0,
    categories=user_categories,
    safety =TRUE,
    api_key="OPENAI_API_KEY")
```

## Academic Research

This package implements methodology from research on LLM performance in social science applications, including the UC Berkeley Social Networks Study. The package addresses reproducibility challenges in LLM-assisted research by providing standardized interfaces and consistent output formatting.

## License

`cat-llm` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
