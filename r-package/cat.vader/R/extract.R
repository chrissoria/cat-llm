#' Extract categories from social media content using LLMs
#'
#' Wraps the Python `catvader.extract()` function. Discovers and returns a
#' normalised, deduplicated set of categories from social media data.
#'
#' @param input_data A character vector, list, or `NULL` to fetch from social
#'   media. Default `NULL`.
#' @param api_key Character or `NULL`. API key for the LLM provider.
#' @param sm_source Character or `NULL`. Social media source.
#' @param sm_limit Integer. Max posts to fetch. Default `50L`.
#' @param sm_months Integer or `NULL`. Fetch posts from last N months.
#' @param sm_credentials Named list or `NULL`. API credentials.
#' @param platform Character or `NULL`. Alias for `sm_source`.
#' @param handle Character or `NULL`. Social media handle.
#' @param hashtags Character vector or `NULL`. Hashtags to filter by.
#' @param post_metadata Named list or `NULL`. Additional post metadata.
#' @param description Character. Context description. Default `""`.
#' @param max_categories Integer. Default `12L`.
#' @param categories_per_chunk Integer. Default `10L`.
#' @param divisions Integer. Default `12L`.
#' @param user_model Character. Default `"gpt-4o"`.
#' @param creativity Numeric or `NULL`. Default `NULL`.
#' @param specificity Character. Default `"broad"`.
#' @param research_question Character or `NULL`.
#' @param mode Character. Default `"text"`.
#' @param filename Character or `NULL`.
#' @param model_source Character. Default `"auto"`.
#' @param iterations Integer. Default `8L`.
#' @param random_state Integer or `NULL`.
#' @param focus Character or `NULL`.
#' @param chunk_delay Numeric. Default `0.0`.
#'
#' @return A named list with `counts_df`, `top_categories`, and `raw_top_text`.
#' @export
extract <- function(
    input_data           = NULL,
    api_key              = NULL,
    sm_source            = NULL,
    sm_limit             = 50L,
    sm_months            = NULL,
    sm_credentials       = NULL,
    platform             = NULL,
    handle               = NULL,
    hashtags             = NULL,
    post_metadata        = NULL,
    description          = "",
    max_categories       = 12L,
    categories_per_chunk = 10L,
    divisions            = 12L,
    user_model           = "gpt-4o",
    creativity           = NULL,
    specificity          = "broad",
    research_question    = NULL,
    mode                 = "text",
    filename             = NULL,
    model_source         = "auto",
    iterations           = 8L,
    random_state         = NULL,
    focus                = NULL,
    chunk_delay          = 0.0
) {
  mod <- .get_catvader()

  api_key <- cat.stack:::.strip_quotes(api_key)
  if (!is.null(creativity)) creativity <- as.double(creativity)

  py_sm_credentials <- if (!is.null(sm_credentials)) reticulate::r_to_py(sm_credentials) else NULL
  py_post_metadata  <- if (!is.null(post_metadata)) reticulate::r_to_py(post_metadata) else NULL
  py_hashtags       <- if (!is.null(hashtags)) reticulate::r_to_py(hashtags) else NULL

  result <- mod$extract(
    input_data           = reticulate::r_to_py(input_data),
    api_key              = api_key,
    sm_source            = sm_source,
    sm_limit             = cat.stack:::.as_py_int(sm_limit),
    sm_months            = cat.stack:::.as_py_int(sm_months),
    sm_credentials       = py_sm_credentials,
    platform             = platform,
    handle               = handle,
    hashtags             = py_hashtags,
    post_metadata        = py_post_metadata,
    description          = description,
    max_categories       = cat.stack:::.as_py_int(max_categories),
    categories_per_chunk = cat.stack:::.as_py_int(categories_per_chunk),
    divisions            = cat.stack:::.as_py_int(divisions),
    user_model           = user_model,
    creativity           = reticulate::r_to_py(creativity),
    specificity          = specificity,
    research_question    = research_question,
    mode                 = mode,
    filename             = filename,
    model_source         = model_source,
    iterations           = cat.stack:::.as_py_int(iterations),
    random_state         = cat.stack:::.as_py_int(random_state),
    focus                = focus,
    progress_callback    = reticulate::py_none(),
    chunk_delay          = as.double(chunk_delay)
  )

  reticulate::py_to_r(result)
}
