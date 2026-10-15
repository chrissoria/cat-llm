# Discover categories from political and policy documents using LLMs

Wraps the Python `catpol.extract()` function. Returns a normalised,
deduplicated set of categories from policy text or a registered source.

## Usage

``` r
extract(
  input_data = NULL,
  api_key = NULL,
  source = NULL,
  doc_type = NULL,
  since = NULL,
  until = NULL,
  n = NULL,
  document_context = "",
  description = "",
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

  A character vector, list, or `NULL` to fetch from a registered source.
  Default `NULL`.

- api_key:

  Character or `NULL`. API key for the LLM provider.

- source:

  Character or `NULL`. Registered source name. Use
  [`list_sources()`](https://christophersoria.com/cat-llm/cat.pol/reference/list_sources.md)
  for options.

- doc_type:

  Character or `NULL`. Filter source by document type.

- since:

  Character or `NULL`. Earliest source row date (YYYY-MM-DD).

- until:

  Character or `NULL`. Latest source row date (YYYY-MM-DD).

- n:

  Integer or `NULL`. Max number of source rows.

- document_context:

  Character. Context about the document. Default `""`.

- description:

  Character. Additional context. Default `""`.

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
  input_data       = df$bill_text,
  document_context = "California state legislation",
  api_key          = Sys.getenv("OPENAI_API_KEY"),
  user_model       = "gpt-4o-mini"
)
print(result$top_categories)
} # }
```
