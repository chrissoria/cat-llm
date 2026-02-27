#' Explore raw categories in text data
#'
#' Wraps the Python `catllm.explore()` function. Returns every category string
#' extracted from every chunk across every iteration — with duplicates intact.
#' Useful for analysing category stability and saturation across repeated
#' extraction runs.
#'
#' Unlike [extract()], which normalises and deduplicates categories, `explore()`
#' returns the raw unprocessed output suitable for frequency and saturation
#' analysis.
#'
#' @param input_data A character vector, list, or `data.frame` column of text
#'   responses.
#' @param api_key Character. API key for the model provider.
#' @param description Character. The survey question or data description.
#'   Default `""`.
#' @param max_categories Integer. Maximum categories per chunk. Default `12L`.
#' @param categories_per_chunk Integer. Categories to extract per chunk.
#'   Default `10L`.
#' @param divisions Integer. Number of data chunks. Default `12L`.
#' @param user_model Character. Model name. Default `"gpt-4o"`.
#' @param creativity Numeric or `NULL`. Temperature setting. `NULL` uses the
#'   provider default. Default `NULL`.
#' @param specificity Character. `"broad"` (default) or `"specific"`.
#' @param research_question Character or `NULL`. Optional research context.
#' @param filename Character or `NULL`. Optional CSV filename to save the raw
#'   category list.
#' @param model_source Character. Provider hint. Default `"auto"`.
#' @param iterations Integer. Number of passes over the data. Default `8L`.
#' @param random_state Integer or `NULL`. Random seed for reproducibility.
#' @param focus Character or `NULL`. Optional focus instruction.
#' @param chunk_delay Numeric. Seconds between API calls. Default `0.0`.
#'
#' @return A character vector of every category string extracted across all
#'   chunks and iterations. Length is approximately
#'   `iterations × divisions × categories_per_chunk`.
#'
#' @examples
#' \dontrun{
#' raw_cats <- explore(
#'   input_data  = df$responses,
#'   description = "Why did you move?",
#'   api_key     = Sys.getenv("OPENAI_API_KEY"),
#'   iterations  = 3L,
#'   divisions   = 5L
#' )
#' length(raw_cats)   # ~150
#' head(raw_cats, 10)
#' }
#'
#' @export
explore <- function(
    input_data,
    api_key,
    description       = "",
    max_categories    = 12L,
    categories_per_chunk = 10L,
    divisions         = 12L,
    user_model        = "gpt-4o",
    creativity        = NULL,
    specificity       = "broad",
    research_question = NULL,
    filename          = NULL,
    model_source      = "auto",
    iterations        = 8L,
    random_state      = NULL,
    focus             = NULL,
    chunk_delay       = 0.0
) {
  cat_py <- .get_catllm()

  api_key <- .strip_quotes(api_key)
  if (!is.null(creativity)) creativity <- as.double(creativity)

  result <- cat_py$explore(
    input_data           = reticulate::r_to_py(input_data),
    api_key              = api_key,
    description          = description,
    max_categories       = .as_py_int(max_categories),
    categories_per_chunk = .as_py_int(categories_per_chunk),
    divisions            = .as_py_int(divisions),
    user_model           = user_model,
    creativity           = reticulate::r_to_py(creativity),
    specificity          = specificity,
    research_question    = research_question,
    filename             = filename,
    model_source         = model_source,
    iterations           = .as_py_int(iterations),
    random_state         = .as_py_int(random_state),
    focus                = focus,
    progress_callback    = reticulate::py_none(),
    chunk_delay          = as.double(chunk_delay)
  )

  reticulate::py_to_r(result)
}
