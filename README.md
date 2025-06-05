![catllm Logo](https://github.com/chrissoria/cat-llm/blob/main/images/logo.png?raw=True)

# catllm

[![PyPI - Version](https://img.shields.io/pypi/v/cat-llm.svg)](https://pypi.org/project/cat-llm)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cat-llm.svg)](https://pypi.org/project/cat-llm)

-----

## Table of Contents

- [Installation](#installation)
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


## License

`cat-llm` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
