#' Explore raw categories in web content
#'
#' Wraps the Python `catweb.explore()` function. Returns every category
#' string extracted from every chunk across every iteration -- with
#' duplicates intact.
#'
#' @param input_data A character vector / list of URLs or text. Default `NULL`.
#' @param api_key Character or `NULL`. API key for the LLM provider.
#' @param source_domain Character or `NULL`. Source domain context.
#' @param content_type Character or `NULL`. Content type context.
#' @param web_metadata Named list or `NULL`. Additional metadata.
#' @param description Character. Default `""`.
#' @param timeout Integer. URL fetch timeout (seconds). Default `30L`.
#' @param max_categories Integer. Default `12L`.
#' @param categories_per_chunk Integer. Default `10L`.
#' @param divisions Integer. Default `12L`.
#' @param user_model Character. Default `"gpt-4o"`.
#' @param creativity Numeric or `NULL`. Default `NULL`.
#' @param specificity Character. Default `"broad"`.
#' @param research_question Character or `NULL`.
#' @param filename Character or `NULL`.
#' @param model_source Character. Default `"auto"`.
#' @param iterations Integer. Default `8L`.
#' @param random_state Integer or `NULL`.
#' @param focus Character or `NULL`.
#' @param chunk_delay Numeric. Default `0.0`.
#'
#' @return A character vector of every category string extracted.
#' @export
explore <- function(
    input_data           = NULL,
    api_key              = NULL,
    source_domain        = NULL,
    content_type         = NULL,
    web_metadata         = NULL,
    description          = "",
    timeout              = 30L,
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
  mod <- .get_catweb()

  api_key <- cat.stack::.strip_quotes(api_key)
  if (!is.null(creativity)) creativity <- as.double(creativity)

  py_web_metadata <- if (!is.null(web_metadata)) reticulate::r_to_py(web_metadata) else NULL

  result <- mod$explore(
    input_data           = reticulate::r_to_py(input_data),
    api_key              = api_key,
    source_domain        = source_domain,
    content_type         = content_type,
    web_metadata         = py_web_metadata,
    description          = description,
    timeout              = cat.stack::.as_py_int(timeout),
    max_categories       = cat.stack::.as_py_int(max_categories),
    categories_per_chunk = cat.stack::.as_py_int(categories_per_chunk),
    divisions            = cat.stack::.as_py_int(divisions),
    user_model           = user_model,
    creativity           = reticulate::r_to_py(creativity),
    specificity          = specificity,
    research_question    = research_question,
    filename             = filename,
    model_source         = model_source,
    iterations           = cat.stack::.as_py_int(iterations),
    random_state         = cat.stack::.as_py_int(random_state),
    focus                = focus,
    progress_callback    = reticulate::py_none(),
    chunk_delay          = as.double(chunk_delay)
  )

  reticulate::py_to_r(result)
}
