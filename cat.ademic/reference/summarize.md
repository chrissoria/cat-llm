# Summarize academic papers using LLMs

Wraps the Python `catademic.summarize()` function. Generates summaries
of academic paper data. The Python function accepts `input_data` and
passes all other arguments through via `**kwargs` to
`cat_stack.summarize()`.

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
  batch_timeout = 86400
)
```

## Arguments

- input_data:

  A character vector, list, or `data.frame` column of paper abstracts or
  text.

- api_key:

  Character or `NULL`. API key for the model provider.

- description:

  Character. Context description. Default `""`.

- instructions:

  Character. Specific instructions for the summary. Default `""`.

- format:

  Character. Output format. Default `"paragraph"`.

- max_length:

  Integer or `NULL`. Max summary length. Default `NULL`.

- focus:

  Character or `NULL`. Optional focus. Default `NULL`.

- user_model:

  Character. Model name. Default `"gpt-4o"`.

- model_source:

  Character. Provider hint. Default `"auto"`.

- mode:

  Character. Processing mode. Default `"image"`.

- input_mode:

  Character or `NULL`. Explicit input mode. Default `NULL`.

- input_type:

  Character. Input type. Default `"auto"`.

- pdf_dpi:

  Integer. DPI for PDFs. Default `150L`.

- creativity:

  Numeric or `NULL`. Temperature. Default `NULL`.

- thinking_budget:

  Integer. Default `0L`.

- chain_of_thought:

  Logical. Default `TRUE`.

- context_prompt:

  Logical. Default `FALSE`.

- step_back_prompt:

  Logical. Default `FALSE`.

- filename:

  Character or `NULL`. Output filename.

- save_directory:

  Character or `NULL`. Output directory.

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
  input_data   = df$abstracts,
  description  = "Sociology journal abstracts",
  instructions = "Summarize the key findings in 2 sentences",
  format       = "paragraph",
  api_key      = Sys.getenv("OPENAI_API_KEY"),
  user_model   = "gpt-4o-mini"
)
} # }
```
