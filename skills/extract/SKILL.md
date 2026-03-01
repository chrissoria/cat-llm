---
name: extract
description: Automatically discover and extract categories from text, images, or PDFs when no predefined scheme exists. Use when the user has open-ended data and needs to find what categories naturally emerge.
---

# CatLLM Category Extraction

You are helping the user discover categories in their data using the CatLLM Python package. Use this when the user does NOT have predefined categories and wants to find what themes or categories naturally emerge from their data.

## Setup

```bash
pip install cat-llm
```

## How to Use

```python
import catllm as cat

# Extract categories from text data
results = cat.extract(
    input_data=df['responses'],
    description="Why did you move to a new city?",
    api_key="YOUR_API_KEY",
    max_categories=10,                    # max categories to return (default 12)
    specificity="broad",                  # "broad" or "specific"
    focus="decisions to relocate",        # optional focus instruction
    research_question="What factors drive relocation decisions?",  # optional
)

# Access results
print(results['top_categories'])          # list of category names
print(results['counts_df'])               # DataFrame with category counts
print(results['raw_top_text'])            # raw model output

# Extract from images
results = cat.extract(
    input_data="/path/to/images/",
    input_type="image",
    description="Product packaging designs",
    api_key="YOUR_API_KEY",
)

# Extract from PDFs
results = cat.extract(
    input_data="/path/to/pdfs/",
    input_type="pdf",
    description="Academic paper abstracts",
    api_key="YOUR_API_KEY",
)
```

## Key Parameters

- `input_data`: list of strings, pandas Series, directory path, or file paths
- `description`: description of the survey question or data context (REQUIRED)
- `api_key`: API key string
- `input_type`: "text" (default), "image", or "pdf"
- `max_categories`: maximum categories to return (default 12)
- `categories_per_chunk`: categories to extract per chunk (default 10)
- `divisions`: number of data chunks (default 12)
- `iterations`: number of extraction passes (default 8)
- `user_model`: model to use (default "gpt-4o")
- `specificity`: "broad" or "specific" granularity
- `research_question`: optional research context
- `focus`: optional focus instruction
- `filename`: output CSV filename

## Default Parameter Rationale

The defaults (`divisions=12`, `iterations=8`) were determined through a 6x6 grid search (360 runs). Consistency peaked at these values with no meaningful improvement beyond them.

## Output

Returns a dict with:
- `counts_df`: DataFrame of categories with frequency counts
- `top_categories`: list of top category names
- `raw_top_text`: raw model output text

## Workflow Tip

A common workflow is: first use `extract()` to discover categories, review them, then use `classify()` with those categories to label all your data.

$ARGUMENTS
