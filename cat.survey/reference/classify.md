# Classify survey responses using LLMs

Wraps the Python `cat_survey.classify()` function. Adds
`survey_question` context to the base cat.stack classification engine.

## Usage

``` r
classify(
  input_data,
  categories,
  survey_question = "",
  description = "",
  add_other = "prompt",
  check_verbosity = TRUE,
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
  batch_mode = FALSE,
  batch_poll_interval = 30,
  batch_timeout = 86400,
  system_prompt = "",
  json_formatter = NULL,
  prompt_tune = NULL,
  tune_iterations = 1L,
  tune_ui = "browser",
  tune_optimize = "balanced",
  ...
)
```

## Arguments

- input_data:

  A character vector, list, or `data.frame` column of survey responses
  to classify.

- categories:

  A character vector of category names, or `"auto"` to infer categories
  from the data.

- survey_question:

  Character. The survey question text. Default `""`.

- description:

  Character. Additional context for the classification task. Default
  `""`.

- add_other:

  Logical or `"prompt"`. Controls addition of an "Other" category.
  Default `"prompt"`.

- check_verbosity:

  Logical. Check category descriptions. Default `TRUE`.

- api_key:

  API key for the model provider (single-model mode).

- user_model:

  Character. Model name. Default `"gpt-4o"`.

- mode:

  Character. PDF processing mode. Default `"image"`.

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

  Integer. Extended thinking budget. Default `0L`.

- example1, example2, example3, example4, example5, example6:

  Optional few-shot examples.

- filename:

  Character or `NULL`. Output CSV filename.

- save_directory:

  Character or `NULL`. Output directory.

- model_source:

  Character. Provider hint. Default `"auto"`.

- max_categories:

  Integer. Max categories for auto mode. Default `12L`.

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

- batch_mode:

  Logical. If `TRUE`, use async batch APIs for ~50% cost savings and
  higher rate limits. Supported providers: OpenAI, Anthropic, Google,
  Mistral, xAI. HuggingFace / Perplexity / Ollama fall back to
  synchronous calls. Text input only. Default `FALSE`.

- batch_poll_interval:

  Numeric. Seconds between batch-job status polls when
  `batch_mode = TRUE`. Default `30.0`.

- batch_timeout:

  Numeric. Maximum seconds to wait for a batch job to complete. Default
  `86400.0` (24 hours).

- system_prompt:

  Character. Custom system-level instruction prepended to every
  classification call, for example the instructions given to human
  coders or a prompt returned by
  [`cat.stack::prompt_tune()`](https://christophersoria.com/cat-llm/cat.stack/reference/prompt_tune.html).
  Takes precedence over `context_prompt`. Default `""`.

- json_formatter:

  Logical or `NULL`. Local JSON-repair model that rescues a reply
  cat-stack cannot use (no category coded 0/1) before the row is marked
  failed; it never runs on usable replies, sparse ones included. `TRUE`
  turns it on, `FALSE` off, and `NULL` (default) keeps the engine's own
  default (on for Ollama models, otherwise it asks on the first unusable
  reply in an interactive session). Synchronous calls only; batch jobs
  do not use it.

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

- ...:

  Advanced arguments passed through to catstack.classify().

## Value

A `data.frame` with classification results.

## Examples

``` r
if (FALSE) { # \dontrun{
results <- classify(
  input_data      = c("Took a new job in Chicago",
                      "Wanted to be closer to grandkids",
                      "Couldn't afford rent in the Bay Area"),
  categories      = c("Job/school", "Family", "Cost of living", "Other"),
  survey_question = "Why did you move?",
  api_key         = Sys.getenv("OPENAI_API_KEY"),
  user_model      = "gpt-4o-mini"
)
} # }
```
