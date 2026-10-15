# Explore raw categories in survey response data

Wraps the Python `cat_survey.explore()` function. Returns every category
string extracted from every chunk across every iteration – with
duplicates intact. Useful for analysing category stability and
saturation.

## Usage

``` r
explore(
  input_data,
  api_key,
  survey_question = "",
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
  chunk_delay = 0
)
```

## Arguments

- input_data:

  A character vector, list, or `data.frame` column of survey responses.

- api_key:

  Character. API key for the model provider.

- survey_question:

  Character. The survey question text. Default `""`.

- description:

  Character. Additional context. Default `""`.

- max_categories:

  Integer. Max categories per chunk. Default `12L`.

- categories_per_chunk:

  Integer. Default `10L`.

- divisions:

  Integer. Number of data chunks. Default `12L`.

- user_model:

  Character. Model name. Default `"gpt-4o"`.

- creativity:

  Numeric or `NULL`. Temperature. Default `NULL`.

- specificity:

  Character. `"broad"` or `"specific"`. Default `"broad"`.

- research_question:

  Character or `NULL`. Optional research context.

- filename:

  Character or `NULL`. Output CSV filename.

- model_source:

  Character. Provider hint. Default `"auto"`.

- iterations:

  Integer. Number of passes. Default `8L`.

- random_state:

  Integer or `NULL`. Random seed.

- focus:

  Character or `NULL`. Optional focus.

- chunk_delay:

  Numeric. Seconds between API calls. Default `0.0`.

## Value

A character vector of every category string extracted.

## Examples

``` r
if (FALSE) { # \dontrun{
raw_categories <- explore(
  input_data      = df$open_response,
  survey_question = "What concerns you most about your community?",
  api_key         = Sys.getenv("OPENAI_API_KEY"),
  user_model      = "gpt-4o-mini",
  iterations      = 4L
)
table(raw_categories)
} # }
```
