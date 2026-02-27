# catllm <img src="https://github.com/chrissoria/cat-llm/blob/main/images/logo.png?raw=True" align="right" height="120" />

R interface to the [CatLLM](https://github.com/chrissoria/cat-llm) Python package for LLM-powered text classification, category extraction, and corpus exploration.

## Installation

```r
# Install the R package from GitHub
devtools::install_github("chrissoria/cat-llm", subdir = "r-package/catllm")

# Install the Python backend (required, one-time setup)
catllm::install_catllm()
```

For PDF support:

```r
catllm::install_catllm(pdf = TRUE)
```

### Prerequisites

- R >= 3.5
- Python >= 3.9
- [reticulate](https://rstudio.github.io/reticulate/) (installed automatically)
- An API key from a supported provider (OpenAI, Anthropic, Google, etc.)

## Quick Start

### Classify text

```r
library(catllm)

results <- classify(
  input_data  = c("I love this product!", "Terrible experience.", "It was fine."),
  categories  = c("Positive", "Negative", "Neutral"),
  description = "Customer satisfaction survey",
  api_key     = Sys.getenv("OPENAI_API_KEY")
)
```

### Multi-model ensemble

```r
results <- classify(
  input_data  = df$responses,
  categories  = c("Positive", "Negative", "Neutral"),
  models      = list(
    c("gpt-4o",              "openai",    Sys.getenv("OPENAI_API_KEY")),
    c("claude-sonnet-4-5-20250929", "anthropic", Sys.getenv("ANTHROPIC_API_KEY")),
    c("gemini-2.5-flash",    "google",    Sys.getenv("GOOGLE_API_KEY"))
  ),
  consensus_threshold = "majority"
)
```

### Extract categories

```r
result <- extract(
  input_data  = df$responses,
  description = "Why did you move to this city?",
  api_key     = Sys.getenv("OPENAI_API_KEY")
)
print(result$top_categories)
```

### Explore category stability

```r
raw_cats <- explore(
  input_data  = df$responses,
  description = "Why did you move?",
  api_key     = Sys.getenv("OPENAI_API_KEY"),
  iterations  = 20L,
  divisions   = 5L
)

# Count how often each category appears across runs
sort(table(raw_cats), decreasing = TRUE)[1:15]
```

## Available Functions

| Function | Description |
|----------|-------------|
| `classify()` | Classify text, images, or PDFs into predefined categories |
| `extract()` | Discover and extract categories from unstructured data |
| `explore()` | Raw category extraction for saturation analysis |
| `install_catllm()` | Install or upgrade the Python backend |

## Configuration

Store API keys as environment variables (e.g., in `~/.Renviron`):

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
```

Then access them with `Sys.getenv("OPENAI_API_KEY")`.

## More Information

- **Full Python documentation and best practices:** [github.com/chrissoria/cat-llm](https://github.com/chrissoria/cat-llm)
- **Supported models:** OpenAI, Anthropic, Google Gemini, HuggingFace, xAI, Mistral, Perplexity
- **Bug reports:** [github.com/chrissoria/cat-llm/issues](https://github.com/chrissoria/cat-llm/issues)

## License

MIT
