# Explore raw categories in web content

Wraps the Python `catweb.explore()` function. Returns every category
string extracted from every chunk across every iteration – with
duplicates intact.

## Usage

``` r
explore(
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
  input_data    = urls,
  source_domain = "example.com",
  api_key       = Sys.getenv("OPENAI_API_KEY"),
  user_model    = "gpt-4o-mini",
  iterations    = 4L
)
table(raw_cats)
} # }
```
