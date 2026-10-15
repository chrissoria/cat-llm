# Explore raw categories in text data

Wraps the Python `cat_stack.explore()` function. Returns every category
string extracted from every chunk across every iteration – with
duplicates intact. Useful for analysing category stability and
saturation across repeated extraction runs.

## Usage

``` r
explore(
  input_data,
  api_key,
  description = "",
  max_categories = 12L,
  categories_per_chunk = 10L,
  divisions = 12L,
  user_model = "gpt-4o",
  creativity = NULL,
  specificity = "broad",
  research_question = NULL,
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

  A character vector, list, or `data.frame` column of text responses.

- api_key:

  Character. API key for the model provider.

- description:

  Character. The survey question or data description. Default `""`.

- max_categories:

  Integer. Maximum categories per chunk. Default `12L`.

- categories_per_chunk:

  Integer. Categories to extract per chunk. Default `10L`.

- divisions:

  Integer. Number of data chunks. Default `12L`.

- user_model:

  Character. Model name. Default `"gpt-4o"`.

- creativity:

  Numeric or `NULL`. Temperature setting. `NULL` uses the provider
  default. Default `NULL`.

- specificity:

  Character. `"broad"` (default) or `"specific"`.

- research_question:

  Character or `NULL`. Optional research context.

- filename:

  Character or `NULL`. Optional CSV filename to save the raw category
  list.

- model_source:

  Character. Provider hint. Default `"auto"`.

- iterations:

  Integer. Number of passes over the data. Default `8L`.

- random_state:

  Integer or `NULL`. Random seed for reproducibility.

- focus:

  Character or `NULL`. Optional focus instruction.

- chunk_delay:

  Numeric. Seconds between API calls. Default `0.0`.

- auto_start_ollama:

  Logical. If `TRUE` (default), automatically call
  [`ensure_ollama_running()`](https://christophersoria.com/cat-llm/cat.stack/reference/ensure_ollama_running.md)
  when `model_source = "ollama"`. Set `FALSE` to skip the check (e.g. on
  CI).

## Value

A character vector of every category string extracted across all chunks
and iterations. Length is approximately
`iterations * divisions * categories_per_chunk`.

## Details

Unlike
[`extract()`](https://christophersoria.com/cat-llm/cat.stack/reference/extract.md),
which normalises and deduplicates categories, `explore()` returns the
raw unprocessed output suitable for frequency and saturation analysis.

## Examples

``` r
if (FALSE) { # \dontrun{
raw_cats <- explore(
  input_data  = df$responses,
  description = "Why did you move?",
  api_key     = Sys.getenv("OPENAI_API_KEY"),
  iterations  = 3L,
  divisions   = 5L
)
length(raw_cats)   # ~150
head(raw_cats, 10)
} # }
```
