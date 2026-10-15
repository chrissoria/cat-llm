# Classify web content using LLMs

Wraps the Python `catweb.classify()` function. Accepts URLs
(auto-fetched to text) or raw text strings. Injects web context (source
domain, content type, metadata) into the classification prompt.

## Usage

``` r
classify(
  categories,
  input_data = NULL,
  api_key = NULL,
  source_domain = NULL,
  content_type = NULL,
  web_metadata = NULL,
  description = "",
  filename = NULL,
  save_directory = NULL,
  timeout = 30L,
  user_model = "gpt-4o",
  mode = "image",
  creativity = NULL,
  safety = FALSE,
  chain_of_verification = FALSE,
  chain_of_thought = FALSE,
  step_back_prompt = FALSE,
  context_prompt = FALSE,
  thinking_budget = 0L,
  example1 = NULL,
  example2 = NULL,
  example3 = NULL,
  example4 = NULL,
  example5 = NULL,
  example6 = NULL,
  model_source = "auto",
  max_categories = 12L,
  categories_per_chunk = 10L,
  divisions = 10L,
  research_question = NULL,
  models = NULL,
  consensus_threshold = "unanimous",
  use_json_schema = TRUE,
  max_workers = NULL,
  fail_strategy = "partial",
  max_retries = 5L,
  batch_retries = 2L,
  retry_delay = 1,
  row_delay = 0,
  pdf_dpi = 150L,
  auto_download = FALSE,
  add_other = "prompt",
  check_verbosity = TRUE,
  prompt_tune = NULL,
  tune_iterations = 1L,
  tune_ui = "browser",
  tune_optimize = "balanced"
)
```

## Arguments

- categories:

  A character vector of category names.

- input_data:

  A character vector / list / `data.frame` column of URLs or text
  strings. Default `NULL`.

- api_key:

  Character or `NULL`. API key for the LLM provider.

- source_domain:

  Character or `NULL`. Source domain injected into the prompt as context
  (e.g. `"nytimes.com"`).

- content_type:

  Character or `NULL`. Content type (e.g. `"news article"`,
  `"blog post"`).

- web_metadata:

  Named list or `NULL`. Additional metadata injected into the prompt.

- description:

  Character. Context description. Default `""`.

- filename:

  Character or `NULL`. Output CSV filename.

- save_directory:

  Character or `NULL`. Output directory.

- timeout:

  Integer. URL fetch timeout (seconds). Default `30L`.

- user_model:

  Character. Model name. Default `"gpt-4o"`.

- mode:

  Character. Processing mode. Default `"image"`.

- creativity:

  Numeric or `NULL`. Temperature. Default `NULL`.

- safety:

  Logical. Default `FALSE`.

- chain_of_verification:

  Logical. Default `FALSE`.

- chain_of_thought:

  Logical. Default `FALSE`.

- step_back_prompt:

  Logical. Default `FALSE`.

- context_prompt:

  Logical. Default `FALSE`.

- thinking_budget:

  Integer. Default `0L`.

- example1, example2, example3, example4, example5, example6:

  Optional few-shot examples.

- model_source:

  Character. Default `"auto"`.

- max_categories:

  Integer. Default `12L`.

- categories_per_chunk:

  Integer. Default `10L`.

- divisions:

  Integer. Default `10L`.

- research_question:

  Character or `NULL`.

- models:

  List of model specs for ensemble mode. Default `NULL`.

- consensus_threshold:

  Character or numeric. Default `"unanimous"`.

- use_json_schema:

  Logical. Default `TRUE`.

- max_workers:

  Integer or `NULL`. Default `NULL`.

- fail_strategy:

  Character. Default `"partial"`.

- max_retries:

  Integer. Default `5L`.

- batch_retries:

  Integer. Default `2L`.

- retry_delay:

  Numeric. Default `1.0`.

- row_delay:

  Numeric. Default `0.0`.

- pdf_dpi:

  Integer. Default `150L`.

- auto_download:

  Logical. Default `FALSE`.

- add_other:

  Logical or `"prompt"`. Default `"prompt"`.

- check_verbosity:

  Logical. Default `TRUE`.

- prompt_tune:

  Integer or `NULL`. Rows sampled per APO correction round. Default
  `NULL`.

- tune_iterations:

  Integer. APO optimization passes. Default `1L`.

- tune_ui:

  Character. Correction UI: `"browser"` or `"terminal"`. Default
  `"browser"`.

- tune_optimize:

  Character. Metric to optimize: `"balanced"`, `"sensitivity"`, or
  `"precision"`. Default `"balanced"`.

## Value

A `data.frame` with classification results.

## Examples

``` r
if (FALSE) { # \dontrun{
# Classify a list of URLs (auto-fetched to text)
results <- classify(
  categories    = c("News", "Opinion", "Tutorial"),
  input_data    = c("https://example.com/article-1",
                    "https://example.com/article-2"),
  source_domain = "example.com",
  content_type  = "blog post",
  api_key       = Sys.getenv("OPENAI_API_KEY"),
  user_model    = "gpt-4o-mini"
)

# Or classify raw text (no fetching)
results <- classify(
  categories = c("News", "Opinion", "Tutorial"),
  input_data = df$article_text,
  api_key    = Sys.getenv("OPENAI_API_KEY")
)
} # }
```
