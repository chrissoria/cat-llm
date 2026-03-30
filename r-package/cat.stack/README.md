# cat.stack

General-purpose LLM text classification engine for R. This is the base package
in the CatLLM R ecosystem, providing domain-agnostic classification, extraction,
exploration, and summarization of text, images, and PDFs using large language
models.

`cat.stack` wraps the Python [cat-stack](https://pypi.org/project/cat-stack/)
package via [reticulate](https://rstudio.github.io/reticulate/). It makes no
domain assumptions and can be used for any classification task.

## Installation

```r
# Install the R package (from source or devtools)
devtools::install("path/to/cat.stack")

# Install the Python backend
cat.stack::install_cat_stack()

# With PDF support
cat.stack::install_cat_stack(pdf = TRUE)
```

## Quick Start

### Classify text

```r
library(cat.stack)

results <- classify(
  input_data  = c("I love this product!", "Terrible experience.", "It was fine."),
  categories  = c("Positive", "Negative", "Neutral"),
  description = "Customer feedback sentiment",
  api_key     = Sys.getenv("OPENAI_API_KEY")
)
```

### Extract categories from data

```r
result <- extract(
  input_data  = df$responses,
  description = "Why did you move to this city?",
  api_key     = Sys.getenv("OPENAI_API_KEY")
)
print(result$top_categories)
```

### Summarize text or PDFs

```r
results <- summarize(
  input_data   = df$articles,
  description  = "News articles",
  instructions = "Provide a 2-sentence summary of each article",
  api_key      = Sys.getenv("OPENAI_API_KEY")
)
```

### Multi-model ensemble

```r
results <- classify(
  input_data  = df$responses,
  categories  = c("Positive", "Negative", "Neutral"),
  models      = list(
    c("gpt-4o",              "openai",    Sys.getenv("OPENAI_API_KEY")),
    c("claude-sonnet-4-5-20250929", "anthropic", Sys.getenv("ANTHROPIC_API_KEY"))
  ),
  consensus_threshold = "unanimous"
)
```

## Functions

| Function | Description |
|----------|-------------|
| `classify()` | Classify text, images, or PDFs into categories |
| `extract()` | Discover and extract categories from data |
| `explore()` | Get raw category extractions for saturation analysis |
| `summarize()` | Summarize text, images, or PDFs |
| `install_cat_stack()` | Install the Python backend |

## License

MIT
