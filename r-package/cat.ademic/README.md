# cat.ademic

Academic paper classification with LLMs. A domain wrapper around
[cat.stack](../cat.stack/) that adds journal and topic sourcing parameters
for classifying, extracting, exploring, and summarizing academic literature.

`cat.ademic` wraps the Python
[catademic](https://pypi.org/project/catademic/) package via
[reticulate](https://rstudio.github.io/reticulate/).

## Installation

```r
devtools::install("path/to/cat.stack")
devtools::install("path/to/cat.ademic")

# Install the Python backend
pip install catademic
```

## Quick Start

### Classify papers by journal

```r
library(cat.ademic)

results <- classify(
  categories   = c("Quantitative", "Qualitative", "Mixed Methods"),
  journal_name = "American Sociological Review",
  paper_limit  = 100L,
  polite_email = "you@university.edu",
  api_key      = Sys.getenv("OPENAI_API_KEY")
)
```

### Extract categories by topic

```r
result <- extract(
  topic_name   = "climate change adaptation",
  paper_limit  = 200L,
  polite_email = "you@university.edu",
  api_key      = Sys.getenv("OPENAI_API_KEY")
)
print(result$top_categories)
```

### Summarize papers

```r
results <- summarize(
  input_data   = df$abstracts,
  description  = "Sociology journal abstracts",
  instructions = "Summarize the key findings in 2 sentences",
  api_key      = Sys.getenv("OPENAI_API_KEY")
)
```

## Functions

| Function | Description |
|----------|-------------|
| `classify()` | Classify academic papers into categories |
| `extract()` | Discover and extract categories from paper data |
| `explore()` | Get raw category extractions for saturation analysis |
| `summarize()` | Summarize academic papers |

## License

MIT
