# Summarize text, images, or PDFs using LLMs

Wraps the Python `cat_stack.summarize()` function. Generates summaries
of input data using one or more LLM models. Supports single-model and
multi-model (ensemble) summarization.

## Usage

``` r
summarize(
  input_data,
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
  batch_timeout = 86400,
  auto_start_ollama = TRUE
)
```

## Arguments

- input_data:

  A character vector, list, or `data.frame` column. For images/PDFs, a
  directory path or character vector of file paths.

- api_key:

  Character or `NULL`. API key for the model provider (single-model
  mode). Not required when `models` is supplied. Default `NULL`.

- description:

  Character. Context description for the summarization task. Default
  `""`.

- instructions:

  Character. Specific instructions for the summary. Default `""`.

- format:

  Character. Output format: `"paragraph"` (default) or other supported
  formats.

- max_length:

  Integer or `NULL`. Maximum length of the summary. `NULL` uses the
  model default. Default `NULL`.

- focus:

  Character or `NULL`. Optional focus for the summary. Default `NULL`.

- user_model:

  Character. Model name. Default `"gpt-4o"`.

- model_source:

  Character. Provider hint: `"auto"`, `"openai"`, `"anthropic"`,
  `"google"`, etc. Default `"auto"`.

- mode:

  Character. Processing mode for images/PDFs: `"image"` (default),
  `"text"`, or `"both"`.

- input_mode:

  Character or `NULL`. Explicit input mode override. Default `NULL`.

- input_type:

  Character. Type of input: `"auto"` (default), `"text"`, `"image"`, or
  `"pdf"`.

- pdf_dpi:

  Integer. DPI for PDF page rendering. Default `150L`.

- creativity:

  Numeric or `NULL`. Temperature setting. `NULL` uses the provider
  default. Default `NULL`.

- thinking_budget:

  Integer. Extended thinking token budget (0 = off). Default `0L`.

- chain_of_thought:

  Logical. Enable chain-of-thought reasoning. Default `TRUE`.

- context_prompt:

  Logical. Add expert context to prompts. Default `FALSE`.

- step_back_prompt:

  Logical. Enable step-back prompting. Default `FALSE`.

- filename:

  Character or `NULL`. Output filename. Default `NULL`.

- save_directory:

  Character or `NULL`. Directory to save results. Default `NULL`.

- models:

  A list of model specifications for multi-model ensemble mode. Each
  element is either a 3-element character vector
  `c("model", "provider", "api_key")` or a 4-element list
  `list("model", "provider", "api_key", list(creativity = 0.5))`.
  Default `NULL`.

- max_workers:

  Integer or `NULL`. Max parallel workers. `NULL` = auto. Default
  `NULL`.

- parallel:

  Logical or `NULL`. Enable parallel processing. Default `NULL`.

- auto_download:

  Logical. Auto-download Ollama models. Default `FALSE`.

- safety:

  Logical. If `TRUE`, saves progress after each item. Default `FALSE`.

- max_retries:

  Integer. Max retries per API call. Default `5L`.

- batch_retries:

  Integer. Max retries for batch-level failures. Default `2L`.

- retry_delay:

  Numeric. Seconds between retries. Default `1.0`.

- row_delay:

  Numeric. Seconds between processing each row. Default `0.0`.

- fail_strategy:

  Character. How to handle failures: `"partial"` (default) or
  `"strict"`.

- batch_mode:

  Logical. Use batch processing mode. Default `FALSE`.

- batch_poll_interval:

  Numeric. Seconds between batch status polls. Default `30.0`.

- batch_timeout:

  Numeric. Maximum seconds to wait for batch completion. Default
  `86400.0`.

- auto_start_ollama:

  Logical. If `TRUE` (default), automatically call
  [`ensure_ollama_running()`](https://christophersoria.com/cat-llm/cat.stack/reference/ensure_ollama_running.md)
  when `model_source = "ollama"` or any ensemble entry uses the
  `"ollama"` provider. Set `FALSE` to skip the check.

## Value

A `data.frame` with summarization results.

## Examples

``` r
if (FALSE) { # \dontrun{
# Single-model summarization
results <- summarize(
  input_data   = c("A long article about climate change...",
                    "A detailed report on economic trends..."),
  description  = "News articles",
  instructions = "Provide a 2-sentence summary",
  api_key      = Sys.getenv("OPENAI_API_KEY")
)

# PDF summarization
results <- summarize(
  input_data = "path/to/documents/",
  input_type = "pdf",
  api_key    = Sys.getenv("OPENAI_API_KEY")
)
} # }
```
