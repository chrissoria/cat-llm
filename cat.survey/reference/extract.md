# Extract categories from survey responses using LLMs

Wraps the Python `cat_survey.extract()` function. Discovers and returns
a normalised, deduplicated set of categories found in survey response
data.

## Usage

``` r
extract(
  input_data,
  api_key,
  survey_question = "",
  description = "",
  input_type = "text",
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

- input_type:

  Character. Type of input. Default `"text"`.

- max_categories:

  Integer. Maximum final categories. Default `12L`.

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

- mode:

  Character. Processing mode. Default `"text"`.

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

A named list with `counts_df`, `top_categories`, and `raw_top_text`.

## Examples

``` r
if (FALSE) { # \dontrun{
result <- extract(
  input_data      = c("Took a new job in Chicago",
                      "Wanted to be closer to grandkids",
                      "Couldn't afford rent in the Bay Area"),
  survey_question = "Why did you move?",
  api_key         = Sys.getenv("OPENAI_API_KEY"),
  user_model      = "gpt-4o-mini"
)
print(result$top_categories)
} # }
```
