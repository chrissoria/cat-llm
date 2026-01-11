![catllm Logo](https://github.com/chrissoria/cat-llm/blob/main/images/logo.png?raw=True)

# cat-llm

CatLLM: A Reproducible LLM Pipeline for Coding Open-Ended Survey Responses

[![PyPI - Version](https://img.shields.io/pypi/v/cat-llm.svg)](https://pypi.org/project/cat-llm)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cat-llm.svg)](https://pypi.org/project/cat-llm)

-----

## The Problem

If you work with open-ended survey data, you know the pain: hundreds or thousands of free-text responses that need to be categorized before you can do any quantitative analysis. The traditional approach is manual coding—either doing it yourself or hiring research assistants. It's slow, expensive, and doesn't scale.

## The Solution

CatLLM is a Python package designed specifically for survey research that uses LLMs to automate the categorization of open-ended responses. It handles both:

- **Category Assignment**: Classify responses into your predefined categories (multi-label supported)
- **Category Extraction**: Automatically discover and extract categories from your data when you don't have a predefined scheme

With leading models like GPT-5, Gemini, and Qwen 3, CatLLM achieves **98% accuracy compared to human consensus** on classification tasks.

**Try the web app:** [https://huggingface.co/spaces/CatLLM/survey-classifier](https://huggingface.co/spaces/CatLLM/survey-classifier)

-----

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Supported Models](#supported-models)
- [API Reference](#api-reference)
  - [classify()](#classify) - Unified function for text, image, and PDF (auto-detects input type)
  - [extract()](#extract) - Unified function for category extraction
  - [summarize()](#summarize) - Unified function for text and PDF summarization
  - [image_score_drawing()](#image_score_drawing)
  - [image_features()](#image_features)
  - [cerad_drawn_score()](#cerad_drawn_score)
- [Deprecated Functions](#deprecated-functions)
- [Related Projects](#related-projects)
- [Academic Research](#academic-research)
- [Contact](#contact)
- [License](#license)

## Installation

```console
pip install cat-llm
```

For PDF support:
```console
pip install cat-llm[pdf]
```

-----

## Quick Start

**This package is designed for building datasets at scale**, not one-off queries. While you can categorize individual responses, its primary purpose is batch processing entire survey columns or image collections into structured research datasets.

Simply provide your survey responses and category list—the package handles the rest and outputs clean data ready for statistical analysis. It works with single or multiple categories per response and automatically skips missing data to save API costs.

Also supports **image and PDF classification** using the same methodology: extract features, count objects, identify categories, or determine presence of elements based on your research questions.

All outputs are formatted for immediate statistical analysis and can be exported directly to CSV.

*Not to be confused with CAT-LLM for Chinese article‐style transfer ([Tao et al. 2024](https://arxiv.org/html/2401.05707v1)).*



## Configuration

### Get Your API Key

Get an API key from your preferred provider:

- **OpenAI**: [platform.openai.com](https://platform.openai.com)
- **Anthropic**: [console.anthropic.com](https://console.anthropic.com)
- **Google**: [aistudio.google.com](https://aistudio.google.com)
- **Huggingface**: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- **xAI**: [console.x.ai](https://console.x.ai)
- **Mistral**: [console.mistral.ai](https://console.mistral.ai)
- **Perplexity**: [perplexity.ai/settings/api](https://www.perplexity.ai/settings/api)

Most providers require adding a payment method and purchasing credits. Store your key securely and never share it publicly.

## Supported Models

- **OpenAI**: GPT-4o, GPT-4, GPT-5, etc.
- **Anthropic**: Claude Sonnet 4, Claude 3.5 Sonnet, Claude Haiku, etc.
- **Google**: Gemini 2.5 Flash, Gemini 2.5 Pro, etc.
- **Huggingface**: Qwen, Llama 4, DeepSeek, and thousands of community models
- **xAI**: Grok models
- **Mistral**: Mistral Large, Pixtral, etc.
- **Perplexity**: Sonar Large, Sonar Small, etc.

**Fully Tested:**
- OpenAI (GPT-4, GPT-4o, GPT-5, etc.)
- Anthropic (Claude Sonnet 4, Claude 3.5 Sonnet, Haiku)
- Perplexity (Sonar models)
- Google Gemini - Free tier has severe rate limits (5 RPM). Requires Google AI Studio billing account for large-scale use.
- Huggingface - Access to Qwen, Llama 4, DeepSeek, and thousands of user-trained models for specific tasks. API routing can occasionally be unstable.
- xAI (Grok models)
- Mistral (Mistral Large, Pixtral, etc.)

**Note:** For best results, I recommend starting with OpenAI or Anthropic.


## API Reference

### `classify()`

Unified classification function for text, image, and PDF inputs. **Input type is auto-detected** from your data—no need to specify whether you're classifying text, images, or PDFs.

Supports both **single-model** and **multi-model ensemble** classification for improved accuracy through consensus voting.

**Parameters:**
- `input_data`: The data to classify. Can be:
  - Text: list of strings or pandas Series
  - Images: directory path, single file, or list of image paths
  - PDFs: directory path, single file, or list of PDF paths
- `categories` (list): List of category names for classification
- `api_key` (str): API key for the LLM service (single-model mode)
- `description` (str): Description of the input data context
- `user_model` (str, default="gpt-4o"): Model to use
- `mode` (str, default="image"): PDF processing mode - "image", "text", or "both"
- `creativity` (float, optional): Temperature setting (0.0-1.0)
- `chain_of_thought` (bool, default=True): Enable step-by-step reasoning
- `filename` (str, optional): Output filename for CSV
- `save_directory` (str, optional): Directory to save results
- `model_source` (str, default="auto"): Provider - "auto", "openai", "anthropic", "google", "mistral", "perplexity", "huggingface", "xai"
- `models` (list, optional): For multi-model ensemble, list of `(model, provider, api_key)` tuples
- `consensus_threshold` (float, default=0.5): Agreement threshold for ensemble mode (0-1)

**Returns:**
- `pandas.DataFrame`: Classification results with category columns

**Examples:**

```python
import catllm as cat

# Text classification (auto-detected)
results = cat.classify(
    input_data=df['responses'],
    categories=["Positive feedback", "Negative feedback", "Neutral"],
    description="Customer satisfaction survey",
    api_key=api_key
)

# Image classification (auto-detected from file paths)
results = cat.classify(
    input_data="/path/to/images/",
    categories=["Contains person", "Outdoor scene", "Has text"],
    description="Product photos",
    api_key=api_key
)

# PDF classification (auto-detected, processes each page separately)
results = cat.classify(
    input_data="/path/to/reports/",
    categories=["Contains table", "Has chart", "Is summary page"],
    description="Financial reports",
    mode="both",  # Use both image and extracted text
    api_key=api_key
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
    consensus_threshold=0.5,  # Majority vote
)
```

**Multi-Model Ensemble:**

When you provide the `models` parameter, CatLLM runs classification across multiple models in parallel and combines results using majority voting. This can significantly improve accuracy by reducing individual model biases.

The output includes:
- Individual model predictions (e.g., `category_1_gpt_4o`, `category_1_claude`)
- Consensus columns (e.g., `category_1_consensus`)
- Agreement scores showing how many models agreed

---

### `extract()`

Unified category extraction function for text, image, and PDF inputs. Automatically discovers categories in your data when you don't have a predefined scheme.

**Parameters:**
- `input_data`: The data to explore (text list, image paths, or PDF paths)
- `api_key` (str): API key for the LLM service
- `input_type` (str, default="text"): Type of input - "text", "image", or "pdf"
- `description` (str): Description of the input data
- `max_categories` (int, default=12): Maximum number of categories to return
- `categories_per_chunk` (int, default=10): Categories to extract per chunk
- `divisions` (int, default=5): Number of chunks to divide data into
- `user_model` (str, default="gpt-4o"): Model to use
- `specificity` (str, default="broad"): "broad" or "specific" category granularity
- `research_question` (str, optional): Research context to guide extraction
- `focus` (str, optional): Focus instruction for category extraction (e.g., "emotional responses")
- `filename` (str, optional): Output filename for CSV

**Returns:**
- `dict` with keys:
  - `counts_df`: DataFrame of categories with counts
  - `top_categories`: List of top category names
  - `raw_top_text`: Raw model output

**Example:**

```python
import catllm as cat

# Extract categories from survey responses
results = cat.extract(
    input_data=df['responses'],
    description="Why did you move?",
    api_key=api_key,
    max_categories=10,
    focus="decisions to relocate"  # Optional focus
)

print(results['top_categories'])
# ['Employment opportunity', 'Family reasons', 'Cost of living', ...]
```

---

### `summarize()`

Unified summarization function for text and PDF inputs. Generates concise summaries of survey responses, documents, or any text data. **Input type is auto-detected** from your data.

Supports both **single-model** and **multi-model ensemble** summarization. In multi-model mode, summaries from all models are synthesized into a consensus summary.

**Parameters:**
- `input_data`: The data to summarize. Can be:
  - Text: list of strings, pandas Series, or single string
  - PDF: directory path, single PDF path, or list of PDF paths
- `api_key` (str): API key for the LLM service (single-model mode)
- `description` (str): Description of what the content contains (provides context)
- `instructions` (str): Specific summarization instructions (e.g., "bullet points")
- `max_length` (int): Maximum summary length in words
- `focus` (str): What to focus on (e.g., "main arguments", "emotional content")
- `user_model` (str, default="gpt-4o"): Model to use
- `model_source` (str, default="auto"): Provider - "auto", "openai", "anthropic", "google", etc.
- `mode` (str, default="image"): PDF processing mode:
  - "image": Render pages as images (best for visual documents)
  - "text": Extract text only (faster, good for text-heavy PDFs)
  - "both": Send both image and extracted text (most comprehensive)
- `filename` (str): Output CSV filename
- `save_directory` (str): Directory to save results
- `models` (list): For multi-model mode, list of `(model, provider, api_key)` tuples

**Returns:**
- `pandas.DataFrame`: Results with summary columns:
  - `survey_input`: Original text or page label (for PDFs)
  - `summary`: Generated summary (or consensus for multi-model)
  - `processing_status`: "success", "error", "skipped"
  - `pdf_path`: Path to source PDF (PDF mode only)
  - `page_index`: Page number, 0-indexed (PDF mode only)

**Examples:**

```python
import catllm as cat

# Single model text summarization
results = cat.summarize(
    input_data=df['responses'],
    description="Customer feedback",
    api_key=api_key
)

# PDF summarization (auto-detected from file paths)
results = cat.summarize(
    input_data="/path/to/pdfs/",
    description="Research papers",
    mode="image",
    api_key=api_key
)

# PDF summarization with specific files and focus
results = cat.summarize(
    input_data=["doc1.pdf", "doc2.pdf"],
    description="Financial reports",
    mode="both",
    focus="key metrics and trends",
    max_length=100,
    api_key=api_key
)

# Multi-model with synthesis
results = cat.summarize(
    input_data=df['responses'],
    models=[
        ("gpt-4o", "openai", "sk-..."),
        ("claude-sonnet-4-5-20250929", "anthropic", "sk-ant-..."),
    ],
)
```

---

### `image_score_drawing()`

Performs quality scoring of images against a reference description and optional reference image, returning structured results with optional CSV export.

**Methodology:**
Processes each image individually, assigning a drawing quality score on a 5-point scale based on similarity to the expected description:

- **1**: No meaningful similarity (fundamentally different)
- **2**: Barely recognizable similarity (25% match)
- **3**: Partial match (50% key features)
- **4**: Strong alignment (75% features)
- **5**: Near-perfect match (90%+ similarity)

**Parameters:**
- `reference_image_description` (str): A description of what the model should expect to see
- `image_input` (list): List of image file paths or folder path containing images
- `reference_image` (str): A file path to the reference image
- `api_key` (str): API key for the LLM service
- `user_model` (str, default="gpt-4o"): Specific vision model to use
- `creativity` (float, default=0): Temperature/randomness setting (0.0-1.0)
- `safety` (bool, default=False): Enable safety checks and save results at each API call step
- `filename` (str, default="image_scores.csv"): Filename for CSV output
- `save_directory` (str, optional): Directory path to save the CSV file
- `model_source` (str, default="OpenAI"): Model provider

**Returns:**
- `pandas.DataFrame`: DataFrame with image paths, quality scores, and analysis details

**Example:**

```python
import catllm as cat

image_scores = cat.image_score_drawing(
    reference_image_description='A hand-drawn circle',
    image_input=['image1.jpg', 'image2.jpg', 'image3.jpg'],
    user_model="gpt-4o",
    api_key="OPENAI_API_KEY"
)
```

---

### `image_features()`

Extracts specific features and attributes from images, returning exact answers to user-defined questions (e.g., counts, colors, presence of objects).

**Methodology:**
Processes each image individually using vision models to extract precise information about specified features. Unlike scoring and classification functions, this returns factual data such as object counts, color identification, or presence/absence of specific elements.

**Parameters:**
- `image_description` (str): A description of what the model should expect to see
- `image_input` (list): List of image file paths or folder path containing images
- `features_to_extract` (list): List of specific features to extract (e.g., ["number of people", "primary color", "contains text"])
- `api_key` (str): API key for the LLM service
- `user_model` (str, default="gpt-4o"): Specific vision model to use
- `creativity` (float, default=0): Temperature/randomness setting (0.0-1.0)
- `to_csv` (bool, default=False): Whether to save the output to a CSV file
- `safety` (bool, default=False): Enable safety checks and save results at each API call step
- `filename` (str, default="categorized_data.csv"): Filename for CSV output
- `save_directory` (str, optional): Directory path to save the CSV file
- `model_source` (str, default="OpenAI"): Model provider

**Returns:**
- `pandas.DataFrame`: DataFrame with image paths and extracted feature values

**Example:**

```python
import catllm as cat

features = cat.image_features(
    image_description='Product photos from e-commerce site',
    features_to_extract=['number of items', 'primary color', 'has price tag'],
    image_input='/path/to/images/',
    user_model="gpt-4o",
    api_key="OPENAI_API_KEY"
)
```

---

### `cerad_drawn_score()`

Automatically scores drawings of circles, diamonds, overlapping rectangles, and cubes according to the official Consortium to Establish a Registry for Alzheimer's Disease (CERAD) scoring system.

**Methodology:**
Processes each image individually, evaluating the drawn shapes based on CERAD criteria. Works even with images that contain other drawings or writing.

**Parameters:**
- `shape` (str): The type of shape to score ("circle", "diamond", "rectangles", "cube")
- `image_input` (list): List of image file paths or folder path containing images
- `api_key` (str): API key for the LLM service
- `user_model` (str, default="gpt-4o"): Specific model to use
- `creativity` (float, default=0): Temperature/randomness setting (0.0-1.0)
- `safety` (bool, default=False): Enable safety checks and save results at each API call step
- `filename` (str, optional): Filename for CSV output
- `model_source` (str, default="auto"): Model provider

**Returns:**
- `pandas.DataFrame`: DataFrame with image paths, CERAD scores, and analysis details

**Example:**

```python
import catllm as cat

diamond_scores = cat.cerad_drawn_score(
    shape="diamond",
    image_input=df['diamond_pic_path'],
    api_key=api_key,
    safety=True,
    filename="diamond_scores.csv",
)
```

---

## Deprecated Functions

The following functions are deprecated and will be removed in a future version. Please use `classify()` instead, which auto-detects input type and supports all the same features.

| Deprecated Function | Replacement |
|---------------------|-------------|
| `multi_class()` | `classify(input_data=texts, ...)` |
| `image_multi_class()` | `classify(input_data=images, ...)` |
| `pdf_multi_class()` | `classify(input_data=pdfs, ...)` |
| `explore_corpus()` | `extract(input_data=texts, ...)` |
| `explore_common_categories()` | `extract(input_data=texts, ...)` |

These functions still work but will show deprecation warnings. Migration is straightforward—simply use `classify()` with your data and it will automatically detect whether you're passing text, images, or PDFs.

---

## Related Projects

**Looking for web research capabilities?** Check out [llm-web-research](https://github.com/chrissoria/llm-web-research) - a precision-focused LLM-powered web research tool that uses a novel Funnel of Verification (FoVe) methodology to reduce false positives. It's designed for use cases where accuracy matters more than completeness.

```bash
pip install llm-web-research
```

## Academic Research

This package implements methodology from research on LLM performance in social science applications, including the UC Berkeley Social Networks Study. The package addresses reproducibility challenges in LLM-assisted research by providing standardized interfaces and consistent output formatting.

If you use this package for research, please cite:

Soria, C. (2025). CatLLM (0.1.0). Zenodo. https://doi.org/10.5281/zenodo.15532317

## Contact

**Interested in research collaboration?** Email: [ChrisSoria@Berkeley.edu](mailto:ChrisSoria@Berkeley.edu)

## License

`cat-llm` is distributed under the terms of the [GNU](https://www.gnu.org/licenses/gpl-3.0.en.html) license.
