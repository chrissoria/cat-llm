# Classify academic papers using LLMs

Wraps the Python `catademic.classify()` function. Adds journal and topic
sourcing parameters to the base cat.stack classification engine.

## Usage

``` r
classify(
  categories,
  input_data = NULL,
  api_key = NULL,
  journal_issn = NULL,
  journal_name = NULL,
  journal_field = NULL,
  topic_name = NULL,
  topic_id = NULL,
  paper_limit = 50L,
  date_from = NULL,
  date_to = NULL,
  polite_email = NULL,
  journal = NULL,
  field = NULL,
  research_focus = NULL,
  paper_metadata = NULL,
  description = "",
  filename = NULL,
  save_directory = NULL,
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

  A character vector of category names, or `"auto"`.

- input_data:

  A character vector, list, or `data.frame` column, or `NULL` to fetch
  from academic sources. Default `NULL`.

- api_key:

  Character or `NULL`. API key for the LLM provider.

- journal_issn:

  Character or `NULL`. Journal ISSN to fetch papers from.

- journal_name:

  Character or `NULL`. Journal name to fetch papers from.

- journal_field:

  Character or `NULL`. Academic field to filter by.

- topic_name:

  Character or `NULL`. Topic name to search for.

- topic_id:

  Character or `NULL`. OpenAlex topic ID.

- paper_limit:

  Integer. Max papers to fetch. Default `50L`.

- date_from:

  Character or `NULL`. Start date (YYYY-MM-DD).

- date_to:

  Character or `NULL`. End date (YYYY-MM-DD).

- polite_email:

  Character or `NULL`. Email for polite API pool.

- journal:

  Character or `NULL`. Alias for `journal_name`.

- field:

  Character or `NULL`. Alias for `journal_field`.

- research_focus:

  Character or `NULL`. Research focus filter.

- paper_metadata:

  Named list or `NULL`. Additional paper metadata.

- description:

  Character. Context description. Default `""`.

- filename:

  Character or `NULL`. Output CSV filename.

- save_directory:

  Character or `NULL`. Output directory.

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

  List of model specs for ensemble mode.

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
# Classify abstracts directly
results <- classify(
  categories = c("Methods", "Theory", "Review", "Other"),
  input_data = df$abstract,
  mode       = "text",
  api_key    = Sys.getenv("OPENAI_API_KEY"),
  user_model = "gpt-4o-mini"
)

# Fetch papers from a journal via OpenAlex
results <- classify(
  categories   = c("Empirical", "Theoretical", "Review"),
  journal_name = "American Sociological Review",
  paper_limit  = 100L,
  polite_email = "you@university.edu",
  api_key      = Sys.getenv("OPENAI_API_KEY")
)
} # }
```
