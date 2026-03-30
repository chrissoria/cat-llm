# cat.survey

Survey response classification with LLMs. A domain wrapper around
[cat.stack](../cat.stack/) that adds `survey_question` context for open-ended
survey response analysis.

`cat.survey` wraps the Python
[cat-survey](https://pypi.org/project/cat-survey/) package via
[reticulate](https://rstudio.github.io/reticulate/).

## Installation

```r
devtools::install("path/to/cat.stack")
devtools::install("path/to/cat.survey")

# Install the Python backend
pip install cat-survey
```

## Quick Start

### Classify survey responses

```r
library(cat.survey)

results <- classify(
  input_data      = df$responses,
  categories      = c("Economic", "Family", "Education", "Other"),
  survey_question = "Why did you move to this city?",
  api_key         = Sys.getenv("OPENAI_API_KEY")
)
```

### Extract categories

```r
result <- extract(
  input_data      = df$responses,
  survey_question = "What do you like about your neighborhood?",
  api_key         = Sys.getenv("OPENAI_API_KEY")
)
print(result$top_categories)
```

### Explore raw categories

```r
raw_cats <- explore(
  input_data      = df$responses,
  survey_question = "Why did you move?",
  api_key         = Sys.getenv("OPENAI_API_KEY"),
  iterations      = 3L
)
```

## Functions

| Function | Description |
|----------|-------------|
| `classify()` | Classify survey responses into categories |
| `extract()` | Discover and extract categories from survey data |
| `explore()` | Get raw category extractions for saturation analysis |

## License

MIT
