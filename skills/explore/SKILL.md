---
name: explore
description: Raw category extraction for frequency and saturation analysis. Use when the user wants to analyze category stability, build saturation curves, or see how consistently categories emerge across multiple extraction runs.
---

# CatLLM Category Exploration

You are helping the user perform raw category extraction for saturation analysis using CatLLM. Unlike `extract()` which normalizes and deduplicates, `explore()` returns every category string from every chunk across every iteration — duplicates intact. This is useful for analyzing category robustness.

## Setup

```bash
pip install cat-llm
```

## How to Use

```python
import catllm as cat

# Run extraction with many iterations for saturation analysis
raw_categories = cat.explore(
    input_data=df['responses'],
    description="Why did you move?",
    api_key="YOUR_API_KEY",
    iterations=20,                        # more iterations = better saturation data
    divisions=5,
    categories_per_chunk=10,
)

# Count how often each category appears across runs
from collections import Counter
counts = Counter(raw_categories)
for category, freq in counts.most_common(15):
    print(f"{freq:3d}x  {category}")

# Build a saturation curve
import pandas as pd
cumulative = []
seen = set()
for i, cat_name in enumerate(raw_categories):
    seen.add(cat_name)
    cumulative.append(len(seen))

pd.Series(cumulative).plot(
    title="Category Saturation Curve",
    xlabel="Extraction Index",
    ylabel="Unique Categories Discovered",
)
```

## Key Parameters

- `input_data`: list of strings or pandas Series
- `description`: the survey question or data description (REQUIRED)
- `api_key`: API key string
- `divisions`: number of data chunks (default 12)
- `iterations`: number of extraction passes (default 8)
- `categories_per_chunk`: categories per chunk (default 10)
- `user_model`: model to use (default "gpt-4o")
- `specificity`: "broad" or "specific"
- `research_question`: optional research context
- `focus`: optional focus instruction
- `random_state`: random seed for reproducibility
- `filename`: output CSV filename (one category per row)

## Output

Returns a `list[str]` — every category extracted from every chunk across every iteration. Length is approximately `iterations x divisions x categories_per_chunk`.

## When to Use explore() vs extract()

- **`extract()`**: Clean, deduplicated final category set. Use when you need categories to classify with.
- **`explore()`**: Raw output with duplicates. Use when you want to analyze stability, build saturation curves, or assess how robust your categories are.

$ARGUMENTS
