# Optimize a classification prompt with human-in-the-loop feedback

Wraps the Python `catstack.prompt_tune()` function. Runs a
coordinate-descent loop: classifies a small sample, asks you to correct
the model's output, then has a meta-LLM rewrite the classification
instructions for each category that had errors. Returns the best system
prompt found plus per-iteration metrics.

## Usage

``` r
prompt_tune(
  input_data,
  categories,
  api_key = NULL,
  user_model = "gpt-4o",
  model_source = "auto",
  models = NULL,
  description = "",
  survey_question = "",
  sample_size = 10L,
  max_iterations = 3L,
  multi_label = TRUE,
  creativity = NULL,
  use_json_schema = TRUE,
  consensus_threshold = "unanimous",
  max_retries = 5L,
  input_mode = NULL,
  ui = "terminal",
  optimize = "balanced",
  add_other = "prompt",
  thinking_budget = 0L,
  auto_start_ollama = TRUE
)
```

## Arguments

- input_data:

  A character vector, list, or `data.frame` column of items to classify
  during tuning.

- categories:

  A character vector of category names. The labels themselves are never
  modified by tuning — only the classification instructions change.

- api_key:

  Character or `NULL`. API key for the LLM provider.

- user_model:

  Character. Model name. Default `"gpt-4o"`.

- model_source:

  Character. Provider hint. Default `"auto"`.

- models:

  List of model specs for ensemble mode (each
  `c(model, provider, api_key)`). Overrides `user_model`/`api_key`/
  `model_source` if given. Default `NULL`.

- description:

  Character. Context description. Default `""`.

- survey_question:

  Character. Survey question for context. Default `""`.

- sample_size:

  Integer. Items to test per iteration. Default `10L`.

- max_iterations:

  Integer. Max instruction attempts per category. Default `3L`.

- multi_label:

  Logical. Multi-label classification. Default `TRUE`.

- creativity:

  Numeric or `NULL`. Temperature. Default `NULL`.

- use_json_schema:

  Logical. Default `TRUE`.

- consensus_threshold:

  Character or numeric. For ensemble mode. Default `"unanimous"`.

- max_retries:

  Integer. Default `5L`.

- input_mode:

  Character or `NULL`. Input mode override.

- ui:

  Character. Review interface for corrections. `"terminal"` (default
  in R) reads from stdin. `"browser"` opens a local web page with
  checkboxes (may not auto-launch from R sessions).

- optimize:

  Character. Which metric to maximize. `"balanced"` (default),
  `"precision"`, or `"sensitivity"`.

- add_other:

  Logical or `"prompt"`. Controls auto-addition of an "Other" catch-all
  category. Default `"prompt"`.

- thinking_budget:

  Integer. Default `0L`.

- auto_start_ollama:

  Logical. If `TRUE` (default), automatically call
  [`ensure_ollama_running()`](https://christophersoria.com/cat-llm/cat.stack/reference/ensure_ollama_running.md)
  when `model_source = "ollama"` or any ensemble entry uses the
  `"ollama"` provider. Set `FALSE` to skip the check.

## Value

A named list with components:

- `system_prompt` — the optimized system prompt (best found)

- `iterations` — list of per-iteration records (label, system_prompt,
  metrics, per_category, total_flips)

- `per_category_summary` — per-category metrics from the best-scoring
  iteration

## Details

This function is **interactive** — you'll be asked to review and correct
the model's labels at least once. From an R session, the default
`ui = "terminal"` reads your corrections from stdin (works in R,
Rscript, and most IDE consoles). `ui = "browser"` opens a local web page
with checkboxes; depending on your R setup this may or may not
auto-launch the browser, so terminal is the safer default for R users.

Use the returned `system_prompt` with
[`classify()`](https://christophersoria.com/cat-llm/cat.stack/reference/classify.md)
via the `system_prompt =` argument to apply the tuned instructions.

## Examples

``` r
if (FALSE) { # \dontrun{
result <- prompt_tune(
  input_data    = df$open_response,
  categories    = c("Positive", "Negative", "Neutral"),
  api_key       = Sys.getenv("OPENAI_API_KEY"),
  user_model    = "gpt-4o-mini",
  sample_size   = 10L,
  max_iterations = 3L,
  ui            = "terminal"
)

# Inspect the optimized prompt
cat(result$system_prompt)

# Use it in classify() via the system_prompt argument
results <- classify(
  input_data    = df$open_response,
  categories    = c("Positive", "Negative", "Neutral"),
  api_key       = Sys.getenv("OPENAI_API_KEY"),
  user_model    = "gpt-4o-mini",
  system_prompt = result$system_prompt
)
} # }
```
