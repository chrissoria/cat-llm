# Classify social media content using LLMs

Wraps the Python `catvader.classify()` function. Adds social media
sourcing parameters to the base cat.stack classification engine.

## Usage

``` r
classify(
  input_data = NULL,
  categories = NULL,
  api_key = NULL,
  sm_source = NULL,
  sm_limit = 50L,
  sm_months = NULL,
  sm_days = NULL,
  sm_credentials = NULL,
  sm_handle = NULL,
  sm_timezone = "UTC",
  sm_youtube_content = "video",
  sm_youtube_transcript = FALSE,
  sm_comments_per_video = 20L,
  sm_youtube_transcript_max_chars = 10000L,
  platform = NULL,
  handle = NULL,
  hashtags = NULL,
  post_metadata = NULL,
  description = "",
  feed_question = "",
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
  add_other = "prompt",
  check_verbosity = TRUE
)
```

## Arguments

- input_data:

  A character vector, list, or `data.frame` column, or `NULL` to fetch
  from social media. Default `NULL`.

- categories:

  A character vector of category names, or `"auto"`. Default `NULL`.

- api_key:

  Character or `NULL`. API key for the LLM provider.

- sm_source:

  Character or `NULL`. Social media source (e.g., `"reddit"`,
  `"twitter"`, `"youtube"`).

- sm_limit:

  Integer. Max posts to fetch. Default `50L`.

- sm_months:

  Integer or `NULL`. Fetch posts from last N months.

- sm_days:

  Integer or `NULL`. Fetch posts from last N days.

- sm_credentials:

  Named list or `NULL`. API credentials for the social media platform.

- sm_handle:

  Character or `NULL`. Social media handle to fetch from.

- sm_timezone:

  Character. Timezone for date filtering. Default `"UTC"`.

- sm_youtube_content:

  Character. YouTube content type. Default `"video"`.

- sm_youtube_transcript:

  Logical. Include YouTube transcripts. Default `FALSE`.

- sm_comments_per_video:

  Integer. Comments per YouTube video. Default `20L`.

- sm_youtube_transcript_max_chars:

  Integer. Max transcript chars. Default `10000L`.

- platform:

  Character or `NULL`. Alias for `sm_source`.

- handle:

  Character or `NULL`. Alias for `sm_handle`.

- hashtags:

  Character vector or `NULL`. Hashtags to filter by.

- post_metadata:

  Named list or `NULL`. Additional post metadata.

- description:

  Character. Context description. Default `""`.

- feed_question:

  Character. Feed-specific question context. Default `""`.

- user_model:

  Character. Model name. Default `"gpt-4o"`.

- mode:

  Character. Processing mode. Default `"image"`.

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

  Integer. Default `0L`.

- example1, example2, example3, example4, example5, example6:

  Optional few-shot examples.

- filename:

  Character or `NULL`. Output CSV filename.

- save_directory:

  Character or `NULL`. Output directory.

- model_source:

  Character. Provider hint. Default `"auto"`.

- max_categories:

  Integer. Default `12L`.

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

- add_other:

  Logical or `"prompt"`. Default `"prompt"`.

- check_verbosity:

  Logical. Default `TRUE`.

## Value

A `data.frame` with classification results.

## Examples

``` r
if (FALSE) { # \dontrun{
# Classify your own text directly
results <- classify(
  input_data = c("Just had the best coffee ever!",
                 "Politicians are all the same",
                 "Looking forward to the game tonight"),
  categories = c("Positive", "Negative", "Neutral"),
  api_key    = Sys.getenv("OPENAI_API_KEY"),
  user_model = "gpt-4o-mini"
)
} # }
```
