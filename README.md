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
  - [classify()](#classify) - Unified function for text, image, and PDF
  - [explore_corpus()](#explore_corpus)
  - [explore_common_categories()](#explore_common_categories)
  - [multi_class()](#multi_class)
  - [image_multi_class()](#image_multi_class)
  - [pdf_multi_class()](#pdf_multi_class)
  - [image_score_drawing()](#image_score_drawing)
  - [image_features()](#image_features)
  - [cerad_drawn_score()](#cerad_drawn_score)
- [Related Projects](#related-projects)
- [Academic Research](#academic-research)
- [Contact](#contact)
- [License](#license)

## Installation

```console
pip install cat-llm
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
- ✅ OpenAI (GPT-4, GPT-4o, GPT-5, etc.)
- ✅ Anthropic (Claude Sonnet 4, Claude 3.5 Sonnet, Haiku)
- ✅ Perplexity (Sonar models)
- ✅ Google Gemini - Free tier has severe rate limits (5 RPM). Requires Google AI Studio billing account for large-scale use.
- ✅ Huggingface - Access to Qwen, Llama 4, DeepSeek, and thousands of user-trained models for specific tasks. API routing can occasionally be unstable.
- ✅ xAI (Grok models)
- ✅ Mistral (Mistral Large, Pixtral, etc.)

**Note:** For best results, I recommend starting with OpenAI or Anthropic.


## API Reference

### `classify()`

Unified classification function for text, image, and PDF inputs. This is the recommended entry point for most users—it dispatches to the appropriate specialized function based on the `input_type` parameter.

**Parameters:**
- `input_data`: The data to classify (text list, image paths, or PDF paths)
- `categories` (list): List of category names for classification
- `api_key` (str): API key for the LLM service
- `input_type` (str, default="text"): Type of input - "text", "image", or "pdf"
- `description` (str): Description of the input data
- `user_model` (str, default="gpt-4o"): Model to use
- `mode` (str, default="image"): PDF processing mode - "image", "text", or "both" (only used when input_type="pdf")
- `creativity` (float, optional): Temperature setting (0.0-1.0)
- `safety` (bool, default=False): Save progress after each item
- `chain_of_thought` (bool, default=True): Enable step-by-step reasoning
- `filename` (str, optional): Output filename for CSV
- `save_directory` (str, optional): Directory to save results
- `model_source` (str, default="auto"): Provider - "auto", "openai", "anthropic", "google", "mistral", "perplexity", "huggingface", "xai"

**Note:** For PDF classification, each page is processed separately and labeled as `{filename}_p{page_number}` (e.g., "report_p1", "report_p2").

**Returns:**
- `pandas.DataFrame`: Classification results with category columns

**Examples:**

```python
import catllm as cat

# Text classification (default)
results = cat.classify(
    input_data=df['responses'],
    categories=["Positive feedback", "Negative feedback", "Neutral"],
    description="Customer satisfaction survey",
    api_key=api_key
)

# Image classification
results = cat.classify(
    input_data="/path/to/images/",
    categories=["Contains person", "Outdoor scene", "Has text"],
    description="Product photos",
    input_type="image",
    api_key=api_key
)

# PDF classification (processes each page separately)
results = cat.classify(
    input_data="/path/to/reports/",
    categories=["Contains table", "Has chart", "Is summary page"],
    description="Financial reports",
    input_type="pdf",
    mode="both",  # Use both image and extracted text
    api_key=api_key
)
```

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
Processes each text response individually, assigning one or more categories from the provided list. Supports flexible output formatting and optional saving of results to CSV for easy integration with data analysis workflows.

**Parameters:**
- `survey_input` (list): List of text responses to classify
- `categories` (list or "auto"): List of predefined categories for classification, or "auto" to automatically extract categories
- `api_key` (str): API key for the LLM service
- `user_model` (str, default="gpt-5"): Specific model to use
- `survey_question` (str, default=""): The survey question being analyzed
- `example1` through `example6` (str, optional): Few-shot learning examples for guiding categorization
- `creativity` (float, optional): Temperature/randomness setting (0.0-1.0, varies by model)
- `safety` (bool, default=False): Enable safety checks on responses and saves to CSV at each API call step
- `chain_of_verification` (bool, default=False): Enable Chain-of-Verification prompting technique for improved accuracy. **⚠️ Warning: CoVe consumes significantly more tokens (3-5x) as it makes multiple API calls per response. Use only if you have a sufficient budget and are willing to pay for marginal improvements in classification accuracy.**
- `chain_of_thought` (bool, default=True): Enable Chain-of-Thought prompting technique for step-by-step reasoning
- `step_back_prompt` (bool, default=False): Enable step-back prompting to analyze higher-level context before classification
- `context_prompt` (bool, default=False): Add expert role and behavioral guidelines to the prompt
- `thinking_budget` (int, default=0): Thinking budget for Google models with extended reasoning capabilities
- `max_categories` (int, default=12): Maximum categories when using "auto" mode
- `categories_per_chunk` (int, default=10): Categories per chunk when using "auto" mode
- `divisions` (int, default=10): Number of divisions when using "auto" mode
- `research_question` (str, optional): Research question to guide auto-categorization
- `filename` (str, optional): Filename for CSV output (triggers save when provided)
- `save_directory` (str, optional): Directory path to save the CSV file
- `model_source` (str, default="auto"): Model provider ("auto", "OpenAI", "Anthropic", "Google", "Mistral", "Perplexity", "Huggingface", "xAI")

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

### `image_multi_class()`

Performs multi-label image classification into user-defined categories, returning structured results with optional CSV export.

**Methodology:**
Processes each image individually, assigning one or more categories from the provided list. Supports flexible output formatting and optional saving of results to CSV for easy integration with data analysis workflows. Includes advanced prompting techniques for improved accuracy.

**Parameters:**
- `image_description` (str): A description of what the model should expect to see
- `image_input` (list): List of file paths or a folder to pull file paths from
- `categories` (list): List of predefined categories for classification
- `api_key` (str): API key for the LLM service
- `user_model` (str, default="gpt-4o"): Specific model to use
- `creativity` (float, optional): Temperature/randomness setting (0.0-1.0)
- `safety` (bool, default=False): Enable safety checks on responses and saves to CSV at each API call step
- `chain_of_verification` (bool, default=False): Enable Chain-of-Verification prompting - re-examines the image to verify categorization accuracy. **⚠️ Warning: CoVe consumes significantly more tokens (3-5x) as it makes multiple API calls per response. Use only if you have a sufficient budget and are willing to pay for marginal improvements in classification accuracy.**
- `chain_of_thought` (bool, default=True): Enable Chain-of-Thought prompting for step-by-step visual analysis
- `step_back_prompt` (bool, default=False): Enable step-back prompting to analyze key visual features before classification
- `context_prompt` (bool, default=False): Add expert visual analyst role and behavioral guidelines to the prompt
- `thinking_budget` (int, default=0): Thinking budget for Google models with extended reasoning capabilities
- `example1` through `example6` (str, optional): Few-shot learning examples for guiding image categorization
- `filename` (str, optional): Filename for CSV output (triggers save when provided)
- `save_directory` (str, optional): Directory path to save the CSV file
- `model_source` (str, default="auto"): Model provider ("auto", "OpenAI", "Anthropic", "Google", "Mistral", "Perplexity", "Huggingface", "xAI")

**Returns:**
- `pandas.DataFrame`: DataFrame with classification results, columns formatted as specified

**Example:**

```
import catllm as cat

user_categories = ["has a cat somewhere in it",
                   "looks cartoonish",
                   "Adrian Brody is in it"]

description = "Should be an image of a child's drawing"

image_categories = cat.image_multi_class(
    image_description=description,
    image_input=['desktop/image1.jpg','desktop/image2.jpg', 'desktop/image3.jpg'],
    user_model="gpt-4o",
    creativity=0,
    categories=user_categories,
    chain_of_thought=True,
    safety=True,
    api_key="OPENAI_API_KEY")
```

### `pdf_multi_class()`

Performs multi-label classification of PDF pages into user-defined categories, returning structured results with optional CSV export. Each page of each PDF is processed separately.

**Installation:**
```console
pip install cat-llm[pdf]
```
Requires PyMuPDF for PDF processing.

**Methodology:**
Processes each PDF page individually, assigning one or more categories from the provided list. Pages are labeled as `{filename}_p{page_number}` (e.g., "report_p1", "report_p2"). Supports three processing modes for different document types and includes advanced prompting techniques for improved accuracy.

**Parameters:**
- `pdf_description` (str): A description of what the PDF documents contain
- `pdf_input` (str or list): Directory path containing PDFs, or list of PDF file paths
- `categories` (list): List of predefined categories for classification
- `api_key` (str): API key for the LLM service
- `user_model` (str, default="gpt-4o"): Specific model to use
- `mode` (str, default="image"): How to process PDF pages:
  - `"image"`: Render pages as images. Best for documents with visual elements (charts, tables, figures, layouts). Uses more tokens but captures visual structure.
  - `"text"`: Extract text only. Faster and cheaper for text-heavy documents like research papers or reports. Won't detect visual elements but processes text more accurately.
  - `"both"`: Send both image and extracted text. Most comprehensive analysis but slowest and most expensive. Use when documents have both important visual elements and dense text.
- `creativity` (float, optional): Temperature/randomness setting (0.0-1.0)
- `safety` (bool, default=False): Enable safety checks and save results at each API call step
- `chain_of_verification` (bool, default=False): Enable Chain-of-Verification prompting - re-examines pages to verify categorization accuracy. **⚠️ Warning: CoVe consumes significantly more tokens (3-5x).**
- `chain_of_thought` (bool, default=True): Enable Chain-of-Thought prompting for step-by-step analysis
- `step_back_prompt` (bool, default=False): Enable step-back prompting to analyze key content patterns before classification
- `context_prompt` (bool, default=False): Add expert document analyst role and behavioral guidelines to the prompt
- `thinking_budget` (int, default=0): Thinking budget for Google models with extended reasoning capabilities
- `example1` through `example6` (str, optional): Few-shot learning examples for guiding categorization
- `filename` (str, optional): Filename for CSV output (triggers save when provided)
- `save_directory` (str, optional): Directory path to save the CSV file
- `model_source` (str, default="auto"): Model provider ("auto", "OpenAI", "Anthropic", "Google", "Mistral", "Perplexity", "Huggingface", "xAI")

**Native PDF Support:**
- ✅ Anthropic and Google: Send PDFs directly without conversion
- 🖼️ Other providers: Automatically convert PDF pages to images

**Returns:**
- `pandas.DataFrame`: DataFrame with classification results including:
  - `pdf_input`: Page label (e.g., "report_p1")
  - `model_response`: Raw model response
  - `category_1`, `category_2`, ...: Binary category assignments (1 = present, 0 = absent)
  - `processing_status`: "success" or "error"

**Example:**

```python
import catllm as cat

# Image mode (default) - best for documents with charts, tables, figures
page_categories = cat.pdf_multi_class(
    pdf_description="Financial quarterly reports",
    pdf_input="/path/to/reports/",  # or list of PDF paths
    categories=[
        "Contains a financial table",
        "Contains a chart or graph",
        "Is a summary or executive page",
        "Contains footnotes or disclaimers"
    ],
    user_model="gpt-4o",
    mode="image",
    creativity=0,
    chain_of_thought=True,
    safety=True,
    filename="report_analysis.csv",
    api_key="OPENAI_API_KEY"
)

# Text mode - faster/cheaper for text-heavy documents
text_categories = cat.pdf_multi_class(
    pdf_description="Research paper pages",
    pdf_input=["paper1.pdf", "paper2.pdf"],
    categories=["Discusses methodology", "Contains results"],
    mode="text",
    api_key="OPENAI_API_KEY"
)

# Both mode - most comprehensive analysis
comprehensive = cat.pdf_multi_class(
    pdf_description="Mixed content documents",
    pdf_input="/path/to/docs/",
    categories=["Has data visualization", "Contains key findings"],
    mode="both",
    api_key="ANTHROPIC_API_KEY",
    user_model="claude-sonnet-4-20250514"
)
```

### `image_score_drawing()`

Performs quality scoring of images against a reference description and optional reference image, returning structured results with optional CSV export.

**Methodology:**
Processes each image individually, assigning a drawing quality score on a 5-point scale based on similarity to the expected description:

- **1**: No meaningful similarity (fundamentally different)
- **2**: Barely recognizable similarity (25% match)  
- **3**: Partial match (50% key features)
- **4**: Strong alignment (75% features)
- **5**: Near-perfect match (90%+ similarity)

Supports flexible output formatting and optional saving of results to CSV for easy integration with data analysis workflows[5].

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
- `model_source` (str, default="OpenAI"): Model provider ("OpenAI", "Anthropic", "Perplexity", "Mistral")

**Returns:**
- `pandas.DataFrame`: DataFrame with image paths, quality scores, and analysis details

**Example:**

```
import catllm as cat          

image_scores = cat.image_score(
    reference_image_description='Adrien Brody sitting in a lawn chair, 
    image_input= ['desktop/image1.jpg','desktop/image2.jpg', desktop/image3.jpg'], 
    user_model="gpt-4o",
    creativity=0,
    safety =TRUE,
    api_key="OPENAI_API_KEY")
```

### `image_features()`

Extracts specific features and attributes from images, returning exact answers to user-defined questions (e.g., counts, colors, presence of objects).

**Methodology:**
Processes each image individually using vision models to extract precise information about specified features. Unlike scoring and multi-class functions, this returns factual data such as object counts, color identification, or presence/absence of specific elements. Supports flexible output formatting and optional CSV export for quantitative analysis workflows.

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
- `model_source` (str, default="OpenAI"): Model provider ("OpenAI", "Anthropic", "Perplexity", "Mistral")

**Returns:**
- `pandas.DataFrame`: DataFrame with image paths and extracted feature values for each specified attribute[1][4]

**Example:**

```
import catllm as cat          

image_scores = cat.image_features(
    image_description='An AI generated image of Spongebob dancing with Patrick', 
    features_to_extract=['Spongebob is yellow','Both are smiling','Patrick is chunky']
    image_input= ['desktop/image1.jpg','desktop/image2.jpg', desktop/image3.jpg'], 
    model_source= 'OpenAI',
    user_model="gpt-4o",
    creativity=0,
    safety =TRUE,
    api_key="OPENAI_API_KEY")
```

### `cerad_drawn_score()`

Automatically scores drawings of circles, diamonds, overlapping rectangles, and cubes according to the official Consortium to Establish a Registry for Alzheimer's Disease (CERAD) scoring system, returning structured results with optional CSV export. Works even with images that contain other drawings or writing.

**Methodology:**
Processes each image individually, evaluating the drawn shapes based on CERAD criteria. Supports optional inclusion of reference shapes within images and can provide reference examples if requested. The function outputs standardized scores facilitating reproducible analysis and integrates optional safety checks and CSV export for research workflows.

**Parameters:**
- `shape` (str): The type of shape to score (e.g., "circle", "diamond", "overlapping rectangles", "cube")
- `image_input` (list): List of image file paths or folder path containing images
- `api_key` (str): API key for the LLM service
- `user_model` (str, default="gpt-4o"): Specific model to use
- `creativity` (float, default=0): Temperature/randomness setting (0.0-1.0)
- `reference_in_image` (bool, default=False): Whether a reference shape is present in the image for comparison
- `provide_reference` (bool, default=False): Whether to provide a reference example image (built in reference image)
- `safety` (bool, default=False): Enable safety checks and save results at each API call step
- `filename` (str, default="categorized_data.csv"): Filename for CSV output
- `model_source` (str, default="OpenAI"): Model provider ("OpenAI", "Anthropic", "Mistral")

**Returns:**
- `pandas.DataFrame`: DataFrame with image paths, CERAD scores, and analysis details

**Example:**

```
import catllm as cat  

diamond_scores = cat.cerad_score(
    shape="diamond",
    image_input=df['diamond_pic_path'],
    api_key=open_ai_key,
    safety=True,
    filename="diamond_gpt_score.csv",
)
```

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
