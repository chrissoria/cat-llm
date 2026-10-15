# Explore raw categories in academic paper data

Wraps the Python `catademic.explore()` function. Returns every category
string extracted from every chunk across every iteration – with
duplicates intact.

## Usage

``` r
explore(
  input_data = NULL,
  api_key = NULL,
  description = "",
  journal_issn = NULL,
  journal_name = NULL,
  journal_field = NULL,
  topic_name = NULL,
  topic_id = NULL,
  paper_limit = 50L,
  date_from = NULL,
  date_to = NULL,
  polite_email = NULL,
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

  A character vector, list, or `NULL` to fetch from academic sources.
  Default `NULL`.

- api_key:

  Character or `NULL`. API key for the LLM provider.

- description:

  Character. Context description. Default `""`.

- journal_issn:

  Character or `NULL`. Journal ISSN.

- journal_name:

  Character or `NULL`. Journal name.

- journal_field:

  Character or `NULL`. Academic field.

- topic_name:

  Character or `NULL`. Topic name.

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

- max_categories:

  Integer. Default `12L`.

- categories_per_chunk:

  Integer. Default `10L`.

- divisions:

  Integer. Default `12L`.

- user_model:

  Character. Default `"gpt-4o"`.

- creativity:

  Numeric or `NULL`. Default `NULL`.

- specificity:

  Character. Default `"broad"`.

- research_question:

  Character or `NULL`.

- filename:

  Character or `NULL`.

- model_source:

  Character. Default `"auto"`.

- iterations:

  Integer. Default `8L`.

- random_state:

  Integer or `NULL`.

- focus:

  Character or `NULL`.

- chunk_delay:

  Numeric. Default `0.0`.

## Value

A character vector of every category string extracted.

## Examples

``` r
if (FALSE) { # \dontrun{
raw_cats <- explore(
  input_data = df$abstracts,
  api_key    = Sys.getenv("OPENAI_API_KEY"),
  user_model = "gpt-4o-mini",
  iterations = 4L
)
table(raw_cats)
} # }
```
