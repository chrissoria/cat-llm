# Classify political and policy documents using LLMs

Wraps the Python `catpol.classify()` function. Can classify either raw
text (via `input_data`) or pull directly from a registered political
data source (via `source`). All catstack classification arguments are
supported.

## Usage

``` r
classify(
  categories,
  input_data = NULL,
  source = NULL,
  doc_type = NULL,
  since = NULL,
  until = NULL,
  n = NULL,
  document_context = "",
  description = "",
  api_key = NULL,
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
  filename = NULL,
  save_directory = NULL,
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
  check_verbosity = TRUE
)
```

## Arguments

- categories:

  A character vector of category names.

- input_data:

  A character vector, list, or `data.frame` column, or `NULL` to fetch
  from a registered source. Default `NULL`.

- source:

  Character or `NULL`. Registered source name (e.g. `"city_san_diego"`,
  `"federal_laws"`, `"federal_executive_orders"`,
  `"social_trump_truth"`). Use
  [`list_sources()`](https://christophersoria.com/cat-llm/cat.pol/reference/list_sources.md)
  for all options.

- doc_type:

  Character or `NULL`. Filter source by document type (e.g.
  `"ordinance"`, `"resolution"`).

- since:

  Character or `NULL`. Earliest source row date (YYYY-MM-DD).

- until:

  Character or `NULL`. Latest source row date (YYYY-MM-DD).

- n:

  Integer or `NULL`. Max number of source rows to classify.

- document_context:

  Character. Context about the policy document being analyzed. Default
  `""`.

- description:

  Character. Additional context description. Default `""`.

- api_key:

  Character or `NULL`. API key for the LLM provider.

- user_model:

  Character. Model name. Default `"gpt-4o"`.

- mode:

  Character. Processing mode. Default `"image"`.

- creativity:

  Numeric or `NULL`. Temperature. Default `NULL`.

- safety:

  Logical. Save progress after each item. Default `FALSE`.

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

- filename:

  Character or `NULL`. Output CSV filename.

- save_directory:

  Character or `NULL`. Output directory.

- model_source:

  Character. Provider hint. Default `"auto"`.

- max_categories:

  Integer. Default `12L`.

- categories_per_chunk:

  Integer. Default `10L`.

- divisions:

  Integer. Default `10L`.

- research_question:

  Character or `NULL`. Optional research context.

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

## Value

A `data.frame` with classification results.

## Examples

``` r
if (FALSE) { # \dontrun{
# Pull recent San Diego ordinances from a registered source
results <- classify(
  source     = "city_san_diego",
  doc_type   = "ordinance",
  since      = "2024-01-01",
  n          = 50L,
  categories = c("Housing", "Public Safety", "Finance",
                 "Infrastructure", "Health"),
  api_key    = Sys.getenv("OPENAI_API_KEY"),
  user_model = "gpt-4o-mini"
)

# Or classify your own text directly
results <- classify(
  input_data = df$bill_text,
  categories = c("Housing", "Public Safety", "Finance"),
  api_key    = Sys.getenv("OPENAI_API_KEY")
)
} # }
```
