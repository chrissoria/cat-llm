# Classify text, images, or PDFs using LLMs

Wraps the Python `cat_stack.classify()` function. Supports both
single-model and multi-model (ensemble) classification.

## Usage

``` r
classify(
  input_data,
  categories,
  api_key = NULL,
  description = "",
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
  survey_question = "",
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
  auto_start_ollama = TRUE
)
```

## Arguments

- input_data:

  A character vector, list of text strings, or `data.frame` column
  containing the items to classify. For image or PDF classification, a
  directory path or character vector of file paths.

- categories:

  A character vector of category names, or `"auto"` to infer categories
  from the data (requires `survey_question`).

- api_key:

  API key for the model provider (single-model mode). Not required when
  `models` is supplied.

- description:

  Character. Context description for the classification task (e.g., the
  survey question or image subject). Default `""`.

- user_model:

  Character. Model name to use in single-model mode. Default `"gpt-4o"`.

- mode:

  Character. PDF processing mode: `"image"` (default), `"text"`, or
  `"both"`.

- creativity:

  Numeric or `NULL`. Temperature setting (0-2). `NULL` uses the provider
  default. Default `NULL`.

- safety:

  Logical. If `TRUE`, saves progress after each item. Default `FALSE`.

- chain_of_verification:

  Logical. Enable Chain of Verification. Empirically degrades accuracy –
  provided for research only. Default `FALSE`.

- chain_of_thought:

  Logical. Enable chain-of-thought reasoning. Default `FALSE`.

- step_back_prompt:

  Logical. Enable step-back prompting. Default `FALSE`.

- context_prompt:

  Logical. Add expert context to prompts. Default `FALSE`.

- thinking_budget:

  Integer. Extended thinking token budget (0 = off). Default `0L`.

- example1, example2, example3, example4, example5, example6:

  Optional few-shot example strings. Empirically degrades accuracy –
  provided for research only.

- filename:

  Character or `NULL`. Output CSV filename. Default `NULL`.

- save_directory:

  Character or `NULL`. Directory to save results. Default `NULL`.

- model_source:

  Character. Provider hint for single-model mode: `"auto"`, `"openai"`,
  `"anthropic"`, `"google"`, `"mistral"`, `"perplexity"`,
  `"huggingface"`, `"xai"`, `"ollama"`, or `"claude-code"`. Default
  `"auto"` (detects from model name; falls back to `"huggingface"` for
  Qwen/Llama/DeepSeek-style names — use `"ollama"` explicitly to route
  those to a local Ollama server).

- max_categories:

  Integer. Maximum number of categories when `categories = "auto"`.
  Default `12L`.

- categories_per_chunk:

  Integer. Categories extracted per chunk when `categories = "auto"`.
  Default `10L`.

- divisions:

  Integer. Number of data chunks when `categories = "auto"`. Default
  `10L`.

- research_question:

  Character or `NULL`. Optional research context. Default `NULL`.

- models:

  A list of model specifications for multi-model ensemble mode. Each
  element is either a 3-element character vector
  `c("model", "provider", "api_key")` or a 4-element list
  `list("model", "provider", "api_key", list(creativity = 0.5))`. When
  `models` is supplied, `api_key` and `user_model` are ignored.

- consensus_threshold:

  Character or numeric. Agreement threshold for ensemble mode. Options:
  `"unanimous"` (default, 100%), `"majority"` (50%), `"two-thirds"`
  (67%), or a numeric value between 0 and 1.

- survey_question:

  Character. The survey question text (used when `categories = "auto"`).
  Default `""`.

- use_json_schema:

  Logical. Use JSON schema for structured output. Default `TRUE`.

- max_workers:

  Integer or `NULL`. Max parallel workers. `NULL` = auto. Default
  `NULL`.

- fail_strategy:

  Character. How to handle failures: `"partial"` (default) or
  `"strict"`.

- max_retries:

  Integer. Max retries per API call. Default `5L`.

- batch_retries:

  Integer. Max retries for batch-level failures. Default `2L`.

- retry_delay:

  Numeric. Seconds between retries. Default `1.0`.

- row_delay:

  Numeric. Seconds between processing each row (useful for rate
  limiting). Default `0.0`.

- pdf_dpi:

  Integer. DPI for PDF page rendering. Default `150L`.

- auto_download:

  Logical. Auto-download Ollama models. Default `FALSE`.

- add_other:

  Logical or `"prompt"`. Controls auto-addition of an "Other" catch-all
  category. `"prompt"` (default) asks interactively – in non-interactive
  sessions this silently defaults to "no". `TRUE` silently adds "Other".
  `FALSE` never adds it.

- check_verbosity:

  Logical. Check whether each category has a description and examples (1
  API call). Default `TRUE`.

- auto_start_ollama:

  Logical. If `TRUE` (default), automatically call
  [`ensure_ollama_running()`](https://christophersoria.com/cat-llm/cat.stack/reference/ensure_ollama_running.md)
  when `model_source = "ollama"` or any ensemble entry uses the
  `"ollama"` provider. Set `FALSE` to skip the check (e.g. on CI runners
  where you don't want to launch Ollama).

## Value

A `data.frame` with one row per input item and classification columns.
In single-model mode the columns are the category names. In ensemble
mode additional `consensus_*` and `agreement_*` columns are included.

## Examples

``` r
if (FALSE) { # \dontrun{
# Single-model classification
results <- classify(
  input_data  = c("I love this!", "Terrible service.", "It was okay."),
  categories  = c("Positive", "Negative", "Neutral"),
  description = "Customer feedback",
  api_key     = Sys.getenv("OPENAI_API_KEY")
)

# Multi-model ensemble
results <- classify(
  input_data  = df$responses,
  categories  = c("Positive", "Negative", "Neutral"),
  models      = list(
    c("gpt-4o",              "openai",    Sys.getenv("OPENAI_API_KEY")),
    c("claude-sonnet-4-5-20250929", "anthropic", Sys.getenv("ANTHROPIC_API_KEY"))
  ),
  consensus_threshold = "unanimous"
)
} # }
```
