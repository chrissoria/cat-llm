# Summarize web content using LLMs

Wraps the Python `catweb.summarize()` function. Accepts URLs
(auto-fetched) or raw text. Web context (source domain, content type,
metadata) is injected into the summarization prompt.

## Usage

``` r
summarize(
  input_data = NULL,
  source_domain = NULL,
  content_type = NULL,
  web_metadata = NULL,
  timeout = 30L,
  api_key = NULL,
  description = "",
  instructions = "",
  format = "paragraph",
  max_length = NULL,
  focus = NULL,
  user_model = "gpt-4o",
  model_source = "auto",
  mode = "image",
  input_mode = NULL,
  input_type = "auto",
  pdf_dpi = 150L,
  creativity = NULL,
  thinking_budget = 0L,
  chain_of_thought = TRUE,
  context_prompt = FALSE,
  step_back_prompt = FALSE,
  filename = NULL,
  save_directory = NULL,
  models = NULL,
  max_workers = NULL,
  parallel = NULL,
  auto_download = FALSE,
  safety = FALSE,
  max_retries = 5L,
  batch_retries = 2L,
  retry_delay = 1,
  row_delay = 0,
  fail_strategy = "partial",
  batch_mode = FALSE,
  batch_poll_interval = 30,
  batch_timeout = 86400
)
```

## Arguments

- input_data:

  Data to summarize: URLs, text, or `data.frame` column.

- source_domain:

  Character or `NULL`. Source domain context.

- content_type:

  Character or `NULL`. Content type context.

- web_metadata:

  Named list or `NULL`. Additional metadata.

- timeout:

  Integer. URL fetch timeout (seconds). Default `30L`.

- api_key:

  Character or `NULL`. API key for the LLM provider.

- description:

  Character. Default `""`.

- instructions:

  Character. Specific instructions for the summary. Default `""`.

- format:

  Character. Default `"paragraph"`.

- max_length:

  Integer or `NULL`. Default `NULL`.

- focus:

  Character or `NULL`. Default `NULL`.

- user_model:

  Character. Default `"gpt-4o"`.

- model_source:

  Character. Default `"auto"`.

- mode:

  Character. Default `"image"`.

- input_mode:

  Character or `NULL`. Default `NULL`.

- input_type:

  Character. Default `"auto"`.

- pdf_dpi:

  Integer. Default `150L`.

- creativity:

  Numeric or `NULL`. Default `NULL`.

- thinking_budget:

  Integer. Default `0L`.

- chain_of_thought:

  Logical. Default `TRUE`.

- context_prompt:

  Logical. Default `FALSE`.

- step_back_prompt:

  Logical. Default `FALSE`.

- filename:

  Character or `NULL`.

- save_directory:

  Character or `NULL`.

- models:

  List of model specs for ensemble mode. Default `NULL`.

- max_workers:

  Integer or `NULL`. Default `NULL`.

- parallel:

  Logical or `NULL`. Default `NULL`.

- auto_download:

  Logical. Default `FALSE`.

- safety:

  Logical. Default `FALSE`.

- max_retries:

  Integer. Default `5L`.

- batch_retries:

  Integer. Default `2L`.

- retry_delay:

  Numeric. Default `1.0`.

- row_delay:

  Numeric. Default `0.0`.

- fail_strategy:

  Character. Default `"partial"`.

- batch_mode:

  Logical. Default `FALSE`.

- batch_poll_interval:

  Numeric. Default `30.0`.

- batch_timeout:

  Numeric. Default `86400.0`.

## Value

A `data.frame` with summarization results.

## Examples

``` r
if (FALSE) { # \dontrun{
summaries <- summarize(
  input_data    = c("https://example.com/article-1",
                    "https://example.com/article-2"),
  source_domain = "example.com",
  content_type  = "news article",
  format        = "bullets",
  api_key       = Sys.getenv("OPENAI_API_KEY"),
  user_model    = "gpt-4o-mini"
)
} # }
```
