# cat.pol

Political and policy document classification with LLMs. A domain wrapper
around [cat.stack](../cat.stack/) that adds a registered-source fetcher
(city ordinances, federal laws, executive orders, presidential speeches,
social-media archives) and policy-document prompt framing.

`cat.pol` wraps the Python
[catpol](https://pypi.org/project/cat-pol/) package via
[reticulate](https://rstudio.github.io/reticulate/).

## Installation

```r
# From R-universe (recommended once published):
install.packages("cat.pol", repos = "https://chrissoria.r-universe.dev")

# Or from a local clone:
devtools::install("path/to/cat.stack")
devtools::install("path/to/cat.pol")

# Install the Python backend
# pip install cat-pol
```

## Quick Start

### Classify ordinances from a built-in source

```r
library(cat.pol)

results <- classify(
  source     = "city_san_diego",
  doc_type   = "ordinance",
  since      = "2024-01-01",
  n          = 50,
  categories = c("Housing", "Public Safety", "Finance",
                 "Infrastructure", "Health"),
  api_key    = Sys.getenv("OPENAI_API_KEY")
)
```

### Discover categories from your own text

```r
result <- extract(
  input_data        = df$bill_text,
  document_context  = "California state legislation",
  api_key           = Sys.getenv("OPENAI_API_KEY")
)
print(result$top_categories)
```

### Summarize policy documents in plain English

```r
results <- summarize(
  source = "federal_executive_orders",
  since  = "2025-01-01",
  format = "paragraph",
  tone   = "eli5",
  api_key = Sys.getenv("OPENAI_API_KEY")
)
```

### List every available data source

```r
list_sources()
#> [1] "city_san_diego"            "city_san_francisco"
#> [3] "federal_laws"              "federal_executive_orders"
#> [5] "social_trump_truth"        ...
```

## Functions

| Function         | Description                                          |
|------------------|------------------------------------------------------|
| `classify()`     | Classify policy documents into categories            |
| `extract()`      | Discover and extract categories from policy text     |
| `explore()`      | Get raw category extractions for saturation analysis |
| `summarize()`    | Summarize policy documents (with `tone` parameter)   |
| `list_sources()` | List every registered political data source          |

## License

MIT
