# Explore raw categories in social media data

Wraps the Python `catvader.explore()` function. Returns every category
string extracted from every chunk across every iteration – with
duplicates intact.

## Usage

``` r
explore(
  input_data = NULL,
  api_key = NULL,
  description = "",
  sm_source = NULL,
  sm_limit = 50L,
  sm_months = NULL,
  sm_credentials = NULL,
  platform = NULL,
  handle = NULL,
  hashtags = NULL,
  post_metadata = NULL,
  max_categories = 12L,
  categories_per_chunk = 10L,
  divisions = 12L,
  user_model = "gpt-4o",
  creativity = NULL,
  specificity = "broad",
  research_question = NULL,
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

  A character vector, list, or `NULL` to fetch from social media.
  Default `NULL`.

- api_key:

  Character or `NULL`. API key for the LLM provider.

- description:

  Character. Context description. Default `""`.

- sm_source:

  Character or `NULL`. Social media source.

- sm_limit:

  Integer. Max posts to fetch. Default `50L`.

- sm_months:

  Integer or `NULL`. Fetch posts from last N months.

- sm_credentials:

  Named list or `NULL`. API credentials.

- platform:

  Character or `NULL`. Alias for `sm_source`.

- handle:

  Character or `NULL`. Social media handle.

- hashtags:

  Character vector or `NULL`. Hashtags to filter by.

- post_metadata:

  Named list or `NULL`. Additional post metadata.

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

A character vector of every category string extracted.

## Examples

``` r
if (FALSE) { # \dontrun{
raw_cats <- explore(
  input_data = df$posts,
  api_key    = Sys.getenv("OPENAI_API_KEY"),
  user_model = "gpt-4o-mini",
  iterations = 4L
)
table(raw_cats)
} # }
```
