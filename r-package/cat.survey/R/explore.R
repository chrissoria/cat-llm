#' Explore raw categories in survey response data
#'
#' Wraps the Python `cat_survey.explore()` function. Returns every category
#' string extracted from every chunk across every iteration -- with duplicates
#' intact. Useful for analysing category stability and saturation.
#'
#' @param input_data A character vector, list, or `data.frame` column of survey
#'   responses.
#' @param api_key Character. API key for the model provider.
#' @param survey_question Character. The survey question text. Default `""`.
#' @param description Character. Additional context. Default `""`.
#' @param max_categories Integer. Max categories per chunk. Default `12L`.
#' @param categories_per_chunk Integer. Default `10L`.
#' @param divisions Integer. Number of data chunks. Default `12L`.
#' @param user_model Character. Model name. Default `"gpt-4o"`.
#' @param creativity Numeric or `NULL`. Temperature. Default `NULL`.
#' @param specificity Character. `"broad"` or `"specific"`. Default `"broad"`.
#' @param research_question Character or `NULL`. Optional research context.
#' @param filename Character or `NULL`. Output CSV filename.
#' @param model_source Character. Provider hint. Default `"auto"`.
#' @param iterations Integer. Number of passes. Default `8L`.
#' @param random_state Integer or `NULL`. Random seed.
#' @param focus Character or `NULL`. Optional focus.
#' @param chunk_delay Numeric. Seconds between API calls. Default `0.0`.
#'
#' @return A character vector of every category string extracted.
#' @export
explore <- function(
    input_data,
    api_key,
    survey_question      = "",
    description          = "",
    max_categories       = 12L,
    categories_per_chunk = 10L,
    divisions            = 12L,
    user_model           = "gpt-4o",
    creativity           = NULL,
    specificity          = "broad",
    research_question    = NULL,
    filename             = NULL,
    model_source         = "auto",
    iterations           = 8L,
    random_state         = NULL,
    focus                = NULL,
    chunk_delay          = 0.0
) {
  mod <- .get_cat_survey()

  api_key <- cat.stack:::.strip_quotes(api_key)
  if (!is.null(creativity)) creativity <- as.double(creativity)

  result <- mod$explore(
    input_data           = reticulate::r_to_py(input_data),
    api_key              = api_key,
    survey_question      = survey_question,
    description          = description,
    max_categories       = cat.stack:::.as_py_int(max_categories),
    categories_per_chunk = cat.stack:::.as_py_int(categories_per_chunk),
    divisions            = cat.stack:::.as_py_int(divisions),
    user_model           = user_model,
    creativity           = reticulate::r_to_py(creativity),
    specificity          = specificity,
    research_question    = research_question,
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
