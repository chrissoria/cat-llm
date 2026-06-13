# Discover categories from web content using LLMs

Wraps the Python `catweb.extract()` function. Accepts URLs
(auto-fetched) or raw text. Returns a normalised, deduplicated set of
categories.

## Usage

``` r
extract(
  input_data = NULL,
  api_key = NULL,
  source_domain = NULL,
  content_type = NULL,
  web_metadata = NULL,
  description = "",
  timeout = 30L,
  max_categories = 12L,
  categories_per_chunk = 10L,
  divisions = 12L,
  user_model = "gpt-4o",
  creativity = NULL,
  specificity = "broad",
  research_question = NULL,
  mode = "text",
  filename = NULL,
  model_source = "auto",
  iterations = 8L,
  random_state = NULL,
  focus = NULL,
  chunk_delay = 0
)
```

## Arguments

- input_data:

  A character vector / list of URLs or text. Default `NULL`.

- api_key:

  Character or `NULL`. API key for the LLM provider.

- source_domain:

  Character or `NULL`. Source domain context.

- content_type:

  Character or `NULL`. Content type context.

- web_metadata:

  Named list or `NULL`. Additional metadata.

- description:

  Character. Default `""`.

- timeout:

  Integer. URL fetch timeout (seconds). Default `30L`.

- max_categories:

  Integer. Default `12L`.

- categories_per_chunk:

  Integer. Default `10L`.

- divisions:

  Integer. Default `12L`.

- user_model:

  Character. Default `"gpt-4o"`.

- creativity:

  Numeric or `NULL`. Default `NULL`.

- specificity:

  Character. Default `"broad"`.

- research_question:

  Character or `NULL`.

- mode:

  Character. Default `"text"`.

- filename:

  Character or `NULL`.

- model_source:

  Character. Default `"auto"`.

- iterations:

  Integer. Default `8L`.

- random_state:

  Integer or `NULL`.

- focus:

  Character or `NULL`.

- chunk_delay:

  Numeric. Default `0.0`.

## Value

A named list with `counts_df`, `top_categories`, and `raw_top_text`.

## Examples

``` r
if (FALSE) { # \dontrun{
result <- extract(
  input_data    = c("https://example.com/page1",
                    "https://example.com/page2"),
  source_domain = "example.com",
  api_key       = Sys.getenv("OPENAI_API_KEY"),
  user_model    = "gpt-4o-mini"
)
print(result$top_categories)
} # }
```
