#' Classify social media content using LLMs
#'
#' Wraps the Python `catvader.classify()` function. Adds social media sourcing
#' parameters to the base cat.stack classification engine.
#'
#' @param input_data A character vector, list, or `data.frame` column, or
#'   `NULL` to fetch from social media. Default `NULL`.
#' @param categories A character vector of category names, or `"auto"`.
#'   Default `NULL`.
#' @param api_key Character or `NULL`. API key for the LLM provider.
#' @param sm_source Character or `NULL`. Social media source (e.g.,
#'   `"reddit"`, `"twitter"`, `"youtube"`).
#' @param sm_limit Integer. Max posts to fetch. Default `50L`.
#' @param sm_months Integer or `NULL`. Fetch posts from last N months.
#' @param sm_days Integer or `NULL`. Fetch posts from last N days.
#' @param sm_credentials Named list or `NULL`. API credentials for the
#'   social media platform.
#' @param sm_handle Character or `NULL`. Social media handle to fetch from.
#' @param sm_timezone Character. Timezone for date filtering. Default `"UTC"`.
#' @param sm_youtube_content Character. YouTube content type. Default `"video"`.
#' @param sm_youtube_transcript Logical. Include YouTube transcripts. Default
#'   `FALSE`.
#' @param sm_comments_per_video Integer. Comments per YouTube video. Default
#'   `20L`.
#' @param sm_youtube_transcript_max_chars Integer. Max transcript chars. Default
#'   `10000L`.
#' @param platform Character or `NULL`. Alias for `sm_source`.
#' @param handle Character or `NULL`. Alias for `sm_handle`.
#' @param hashtags Character vector or `NULL`. Hashtags to filter by.
#' @param post_metadata Named list or `NULL`. Additional post metadata.
#' @param description Character. Context description. Default `""`.
#' @param feed_question Character. Feed-specific question context. Default `""`.
#' @param user_model Character. Model name. Default `"gpt-4o"`.
#' @param mode Character. Processing mode. Default `"image"`.
#' @param creativity Numeric or `NULL`. Temperature. Default `NULL`.
#' @param safety Logical. Save progress after each item. Default `FALSE`.
#' @param chain_of_verification Logical. Default `FALSE`.
#' @param chain_of_thought Logical. Default `FALSE`.
#' @param step_back_prompt Logical. Default `FALSE`.
#' @param context_prompt Logical. Default `FALSE`.
#' @param thinking_budget Integer. Default `0L`.
#' @param example1,example2,example3,example4,example5,example6 Optional
#'   few-shot examples.
#' @param filename Character or `NULL`. Output CSV filename.
#' @param save_directory Character or `NULL`. Output directory.
#' @param model_source Character. Provider hint. Default `"auto"`.
#' @param max_categories Integer. Default `12L`.
#' @param categories_per_chunk Integer. Default `10L`.
#' @param divisions Integer. Default `10L`.
#' @param research_question Character or `NULL`. Optional research context.
#' @param models List of model specs for ensemble mode.
#' @param consensus_threshold Character or numeric. Default `"unanimous"`.
#' @param use_json_schema Logical. Default `TRUE`.
#' @param max_workers Integer or `NULL`. Default `NULL`.
#' @param fail_strategy Character. Default `"partial"`.
#' @param max_retries Integer. Default `5L`.
#' @param batch_retries Integer. Default `2L`.
#' @param retry_delay Numeric. Default `1.0`.
#' @param row_delay Numeric. Default `0.0`.
#' @param pdf_dpi Integer. Default `150L`.
#' @param auto_download Logical. Default `FALSE`.
#' @param add_other Logical or `"prompt"`. Default `"prompt"`.
#' @param check_verbosity Logical. Default `TRUE`.
#'
#' @return A `data.frame` with classification results.
#' @export
classify <- function(
    input_data           = NULL,
    categories           = NULL,
    api_key              = NULL,
    sm_source            = NULL,
    sm_limit             = 50L,
    sm_months            = NULL,
    sm_days              = NULL,
    sm_credentials       = NULL,
    sm_handle            = NULL,
    sm_timezone          = "UTC",
    sm_youtube_content   = "video",
    sm_youtube_transcript = FALSE,
    sm_comments_per_video = 20L,
    sm_youtube_transcript_max_chars = 10000L,
    platform             = NULL,
    handle               = NULL,
    hashtags             = NULL,
    post_metadata        = NULL,
    description          = "",
    feed_question        = "",
    user_model           = "gpt-4o",
    mode                 = "image",
    creativity           = NULL,
    safety               = FALSE,
    chain_of_verification = FALSE,
    chain_of_thought     = FALSE,
    step_back_prompt     = FALSE,
    context_prompt       = FALSE,
    thinking_budget      = 0L,
    example1             = NULL,
    example2             = NULL,
    example3             = NULL,
    example4             = NULL,
    example5             = NULL,
    example6             = NULL,
    filename             = NULL,
    save_directory       = NULL,
    model_source         = "auto",
    max_categories       = 12L,
    categories_per_chunk = 10L,
    divisions            = 10L,
    research_question    = NULL,
    models               = NULL,
    consensus_threshold  = "unanimous",
    use_json_schema      = TRUE,
    max_workers          = NULL,
    fail_strategy        = "partial",
    max_retries          = 5L,
    batch_retries        = 2L,
    retry_delay          = 1.0,
    row_delay            = 0.0,
    pdf_dpi              = 150L,
    auto_download        = FALSE,
    add_other            = "prompt",
    check_verbosity      = TRUE
) {
  mod <- .get_catvader()

  api_key   <- cat.stack:::.strip_quotes(api_key)
  add_other <- cat.stack:::.validate_add_other(add_other)
  if (!is.null(creativity)) creativity <- as.double(creativity)

  py_models <- if (!is.null(models)) cat.stack:::.convert_models(models) else reticulate::py_none()

  # Convert list args to Python dicts
  py_sm_credentials <- if (!is.null(sm_credentials)) reticulate::r_to_py(sm_credentials) else NULL
  py_post_metadata  <- if (!is.null(post_metadata)) reticulate::r_to_py(post_metadata) else NULL
  py_hashtags       <- if (!is.null(hashtags)) reticulate::r_to_py(hashtags) else NULL

  result <- mod$classify(
    input_data            = reticulate::r_to_py(input_data),
    categories            = reticulate::r_to_py(categories),
    api_key               = api_key,
    sm_source             = sm_source,
    sm_limit              = cat.stack:::.as_py_int(sm_limit),
    sm_months             = cat.stack:::.as_py_int(sm_months),
    sm_days               = cat.stack:::.as_py_int(sm_days),
    sm_credentials        = py_sm_credentials,
    sm_handle             = sm_handle,
    sm_timezone           = sm_timezone,
    sm_youtube_content    = sm_youtube_content,
    sm_youtube_transcript = sm_youtube_transcript,
    sm_comments_per_video = cat.stack:::.as_py_int(sm_comments_per_video),
    sm_youtube_transcript_max_chars = cat.stack:::.as_py_int(sm_youtube_transcript_max_chars),
    platform              = platform,
    handle                = handle,
    hashtags              = py_hashtags,
    post_metadata         = py_post_metadata,
    description           = description,
    feed_question         = feed_question,
    user_model            = user_model,
    mode                  = mode,
    creativity            = reticulate::r_to_py(creativity),
    safety                = safety,
    chain_of_verification = chain_of_verification,
    chain_of_thought      = chain_of_thought,
    step_back_prompt      = step_back_prompt,
    context_prompt        = context_prompt,
    thinking_budget       = cat.stack:::.as_py_int(thinking_budget),
    example1              = example1,
    example2              = example2,
    example3              = example3,
    example4              = example4,
    example5              = example5,
    example6              = example6,
    filename              = filename,
    save_directory        = save_directory,
    model_source          = model_source,
    max_categories        = cat.stack:::.as_py_int(max_categories),
    categories_per_chunk  = cat.stack:::.as_py_int(categories_per_chunk),
    divisions             = cat.stack:::.as_py_int(divisions),
    research_question     = research_question,
    progress_callback     = reticulate::py_none(),
    models                = py_models,
    consensus_threshold   = consensus_threshold,
    use_json_schema       = use_json_schema,
    max_workers           = cat.stack:::.as_py_int(max_workers),
    fail_strategy         = fail_strategy,
    max_retries           = cat.stack:::.as_py_int(max_retries),
    batch_retries         = cat.stack:::.as_py_int(batch_retries),
    retry_delay           = as.double(retry_delay),
    row_delay             = as.double(row_delay),
    pdf_dpi               = cat.stack:::.as_py_int(pdf_dpi),
    auto_download         = auto_download,
    add_other             = add_other,
    check_verbosity       = check_verbosity
  )

  reticulate::py_to_r(result)
}
