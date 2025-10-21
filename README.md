![catllm Logo](https://github.com/chrissoria/cat-llm/blob/main/images/logo.png?raw=True)

# cat-llm

[![PyPI - Version](https://img.shields.io/pypi/v/cat-llm.svg)](https://pypi.org/project/cat-llm)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cat-llm.svg)](https://pypi.org/project/cat-llm)

-----

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Supported Models](#supported-models)
- [API Reference](#api-reference)
  - [explore_corpus()](#explore_corpus)
  - [explore_common_categories()](#explore_common_categories)
  - [multi_class()](#multi_class)
  - [image_score()](#image_score)
  - [image_features()](#image_features)
  - [build_web_research_dataset()](#build_web_research_dataset)
  - [cerad_drawn_score()](#cerad_drawn_score)
- [Academic Research](#academic-research)
- [License](#license)

## Installation

```console
pip install cat-llm
```

## Quick Start

CatLLM helps social scientists and researchers automatically categorize open-ended survey responses, images, and web-scraped data using AI models like GPT-4 and Claude.

Text Analysis: Simply provide your survey responses and category list - the package handles the rest and outputs clean data ready for statistical analysis. It works with single or multiple categories per response and automatically skips missing data to save API costs.

Image Categorization: Uses the same intelligent categorization method to analyze images, extracting specific features, counting objects, identifying colors, or determining the presence of elements based on your research questions.

Web Data Collection: Builds comprehensive datasets by scraping web data and using Large Language Models to extract exactly the information you need. The function searches across multiple sources, processes the findings through AI models, and structures everything into clean dataframe format ready for export to CSV.

Whether you're working with messy text responses, analyzing visual content, or gathering information from across the web, CatLLM consistently transforms unstructured data into structured categories and datasets you can actually analyze. All outputs are formatted for immediate statistical analysis and can be exported directly to CSV for further research workflows.

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
Processes each text response individually, assigning one or more categories from the provided list. Supports flexible output formatting and optional saving of results to CSV for easy integration with data analysis workflows.

**Parameters:**
- `survey_input` (list): List of text responses to classify
- `categories` (list): List of predefined categories for classification
- `api_key` (str): API key for the LLM service
- `user_model` (str, default="gpt-5"): Specific model to use
- `user_prompt` (str, optional): Custom prompt template to override default prompting
- `survey_question` (str, default=""): The survey question being analyzed
- `example1` through `example6` (dict, optional): Few-shot learning examples (format: {"response": "...", "categories": [...]})
- `creativity` (float, optional): Temperature/randomness setting (0.0-1.0, varies by model)
- `safety` (bool, default=False): Enable safety checks on responses and saves to CSV at each API call step
- `to_csv` (bool, default=False): Whether to save results to CSV
- `chain_of_verification` (bool, default=False): Enable Chain-of-Verification prompting technique for improved accuracy
- `chain_of_thought` (bool, default=False): Enable Chain-of-Thought prompting technique for improved accuracy
- `step_back_prompt` (bool, default=False): Enable step-back prompting to analyze higher-level context before classification
- `context_prompt` (bool, default=False): Add expert role and behavioral guidelines to the prompt
- `filename` (str, default="categorized_data.csv"): Filename for CSV output
- `save_directory` (str, optional): Directory path to save the CSV file
- `model_source` (str, default="auto"): Model provider ("auto", "OpenAI", "Anthropic", "Google", "Mistral", "Perplexity", "Huggingface")

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
Processes each image individually, assigning one or more categories from the provided list. Supports flexible output formatting and optional saving of results to CSV for easy integration with data analysis workflows.

**Parameters:**
- `image_description` (str): A description of what the model should expect to see
- `image_input` (list): List of file paths or a folder to pull file paths from
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

user_categories = ["has a cat somewhere in it",
                   "looks cartoonish",
                   "Adrian Brody is in it"]

description = "Should be an image of a child's drawing"                   

image_categories = cat.image_multi_class(
    image_description=description, 
    image_input= ['desktop/image1.jpg','desktop/image2.jpg', desktop/image3.jpg'], 
    user_model="gpt-4o",
    creativity=0,
    categories=user_categories,
    safety =TRUE,
    api_key="OPENAI_API_KEY")
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

### `build_web_research_dataset()`

Conducts automated web research on specified topics and compiles the findings into a structured dataset, extracting answers and source URLs for comprehensive research workflows. 

NOTE: This function currently only works with Anthropic models and requires an Anthropic API key. It is strongly recommended to increase your API rate limits before using this function to avoid interruptions during web research tasks.

SECOND NOTE: This function works best if you are specific with your search question. For example, instead of search_question="Hottest temperature in 2024?" you should use "Hottest temperature in 2024 from extremeweatherwatch.com?" or "Hottest temperature in 2024 from weatherundeground.com?". Another example is use "Where these UC Berkeley professors got their PhD according to Linkedin?" instead of "Where they got their PhD according to Linkedin?" to avoid matching people with the same name. 

THIRD NOTE: This function works by scraping data from the web. Be aware that not all websites allow webscraping from Anthropic and therefore the function won't be able to retrieve information from these sites.

**Methodology:**
Performs systematic web searches using the specified search questions and processes the results through Anthropic's language models to extract relevant information. The function handles multiple search queries sequentially, applying time delays between requests to respect rate limits. Results are categorized according to user-defined criteria and can be exported to CSV format for further analysis and research documentation.

**Rate Limits:**
Before using this function, review and increase your Anthropic API rate limits at: https://console.anthropic.com/settings/limits. For general information about API rate limits, consult the Anthropic documentation at: https://docs.anthropic.com/claude/reference/rate-limits

**Parameters:**
- `search_question` (str): Primary research question or topic to guide the search strategy
- `search_input` (list): List of specific search queries or questions to investigate
- `features_to_extract` (list): List of specific features to extract (e.g., ["number of people", "primary color", "contains text"])
- `api_key` (str): API key for the LLM service
- `answer_format`: (str, default="concise"): Response detail level ("concise", "detailed", "comprehensive")
- `additional_instructions` (str, default="claude-3-7-sonnet-20250219"): Specific Anthropic model to use for processing results
- `user_model` (str, default="gpt-4o"): Specific vision model to use
- `creativity` (float, default=0): Temperature/randomness setting (0.0-1.0)
- `safety` (bool, default=False): Enable safety checks and save results at each API call step
- `filename` (str, default="categorized_data.csv"): Filename for CSV output
- `save_directory` (str, optional): Directory path to save the CSV file
- `model_source` (str, default="OpenAI"): Model provider ("OpenAI", "Anthropic", "Perplexity", "Mistral")
- `time_delay` (int, default=15): Delay in seconds between search requests to manage API rate limits

**Returns:**
- `pandas.DataFrame`: DataFrame with image paths and extracted feature values for each specified attribute[1][4]

**Example:**

```
import catllm as cat          

research_data = cat.build_web_research_dataset(
    search_question="What are the latest developments in renewable energy technology?",
    search_input=["solar panel efficiency 2025", "wind turbine innovations", "battery storage breakthroughs"],
    api_key="ANTHROPIC_API_KEY",
    answer_format="detailed",
    additional_instructions="Focus on recent technological advances and commercial applications",
    categories=['Answer', 'URL', 'Date', 'Key_Technology'],
    model_source="Anthropic",
    user_model="claude-3-7-sonnet-20250219",
    creativity=0.1,
    safety=True,
    time_delay=3
)
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


## Academic Research

This package implements methodology from research on LLM performance in social science applications, including the UC Berkeley Social Networks Study. The package addresses reproducibility challenges in LLM-assisted research by providing standardized interfaces and consistent output formatting.

If you use this package for research, please cite:

Soria, C. (2025). CatLLM (0.0.8). Zenodo. https://doi.org/10.5281/zenodo.15532317

## License

`cat-llm` is distributed under the terms of the [GNU](https://www.gnu.org/licenses/gpl-3.0.en.html) license.
