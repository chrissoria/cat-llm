# cat.vader

Social media content classification with LLMs. A domain wrapper around
[cat.stack](../cat.stack/) that adds social media sourcing parameters for
classifying posts from Reddit, Twitter/X, YouTube, and other platforms.

`cat.vader` wraps the Python [catvader](https://pypi.org/project/catvader/)
package via [reticulate](https://rstudio.github.io/reticulate/).

## Installation

```r
devtools::install("path/to/cat.stack")
devtools::install("path/to/cat.vader")

# Install the Python backend
pip install catvader
```

## Quick Start

### Classify social media posts

```r
library(cat.vader)

# Fetch and classify Reddit posts
results <- classify(
  categories     = c("Political", "Economic", "Social", "Other"),
  sm_source      = "reddit",
  sm_handle      = "politics",
  sm_limit       = 100L,
  sm_months      = 3L,
  api_key        = Sys.getenv("OPENAI_API_KEY"),
  sm_credentials = list(
    client_id     = Sys.getenv("REDDIT_CLIENT_ID"),
    client_secret = Sys.getenv("REDDIT_CLIENT_SECRET")
  )
)

# Classify pre-loaded data
results <- classify(
  input_data  = df$posts,
  categories  = c("Positive", "Negative", "Neutral"),
  description = "Social media sentiment",
  api_key     = Sys.getenv("OPENAI_API_KEY")
)
```

### Extract categories

```r
result <- extract(
  sm_source      = "reddit",
  sm_handle      = "technology",
  sm_limit       = 200L,
  api_key        = Sys.getenv("OPENAI_API_KEY"),
  sm_credentials = list(
    client_id     = Sys.getenv("REDDIT_CLIENT_ID"),
    client_secret = Sys.getenv("REDDIT_CLIENT_SECRET")
  )
)
print(result$top_categories)
```

## Functions

| Function | Description |
|----------|-------------|
| `classify()` | Classify social media content into categories |
| `extract()` | Discover and extract categories from social media data |
| `explore()` | Get raw category extractions for saturation analysis |

## License

MIT
