# Extract categories from text, images, or PDFs using LLMs

Wraps the Python `cat_stack.extract()` function. Discovers and returns a
normalised, deduplicated set of categories found in the input data.

## Usage

``` r
extract(
  input_data,
  api_key,
  input_type = "text",
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
  chunk_delay = 0,
  auto_start_ollama = TRUE
)
```

## Arguments

- input_data:

  A character vector, list, or `data.frame` column. For images/PDFs, a
  directory path or character vector of file paths.

- api_key:

  Character. API key for the model provider.

- input_type:

  Character. Type of input: `"text"` (default), `"image"`, or `"pdf"`.

- description:

  Character. The survey question or data description. Default `""`.

- max_categories:

  Integer. Maximum number of final categories to return. Default `12L`.

- categories_per_chunk:

  Integer. Categories to extract per data chunk. Default `10L`.

- divisions:

  Integer. Number of chunks to divide the data into. Default `12L`.

- user_model:

  Character. Model name. Default `"gpt-4o"`.

- creativity:

  Numeric or `NULL`. Temperature setting. `NULL` uses the provider
  default. Default `NULL`.

- specificity:

  Character. Category granularity: `"broad"` (default) or `"specific"`.

- research_question:

  Character or `NULL`. Optional research context.

- mode:

  Character. Processing mode. For PDFs: `"text"` (default), `"image"`,
  or `"both"`. For images: `"image"` (default) or `"both"`.

- filename:

  Character or `NULL`. Optional CSV filename to save results.

- model_source:

  Character. Provider hint: `"auto"`, `"openai"`, `"anthropic"`,
  `"google"`, etc. Default `"auto"`.

- iterations:

  Integer. Number of passes over the data. Default `8L`.

- random_state:

  Integer or `NULL`. Random seed for reproducibility.

- focus:

  Character or `NULL`. Optional focus for extraction (e.g.,
  `"decisions to move"`).

- chunk_delay:

  Numeric. Seconds between API calls (rate limiting). Default `0.0`.

- auto_start_ollama:

  Logical. If `TRUE` (default), automatically call
  [`ensure_ollama_running()`](https://christophersoria.com/cat-llm/cat.stack/reference/ensure_ollama_running.md)
  when `model_source = "ollama"`. Set `FALSE` to skip the check (e.g. on
  CI).

## Value

A named list with:

- `counts_df`:

  A `data.frame` of discovered categories with counts.

- `top_categories`:

  A character vector of the top category names.

- `raw_top_text`:

  The raw model output from the final merge step.

## Examples

``` r
if (FALSE) { # \dontrun{
result <- extract(
  input_data  = df$responses,
  description = "Why did you move to this city?",
  api_key     = Sys.getenv("OPENAI_API_KEY")
)
print(result$top_categories)
print(result$counts_df)
} # }
```
