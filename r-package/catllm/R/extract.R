#' Extract categories from text, images, or PDFs using LLMs
#'
#' Wraps the Python `catllm.extract()` function. Discovers and returns a
#' normalised, deduplicated set of categories found in the input data.
#'
#' @param input_data A character vector, list, or `data.frame` column. For
#'   images/PDFs, a directory path or character vector of file paths.
#' @param api_key Character. API key for the model provider.
#' @param input_type Character. Type of input: `"text"` (default), `"image"`,
#'   or `"pdf"`.
#' @param description Character. The survey question or data description.
#'   Default `""`.
#' @param max_categories Integer. Maximum number of final categories to return.
#'   Default `12L`.
#' @param categories_per_chunk Integer. Categories to extract per data chunk.
#'   Default `10L`.
#' @param divisions Integer. Number of chunks to divide the data into.
#'   Default `12L`.
#' @param user_model Character. Model name. Default `"gpt-4o"`.
#' @param creativity Numeric or `NULL`. Temperature setting. `NULL` uses the
#'   provider default. Default `NULL`.
#' @param specificity Character. Category granularity: `"broad"` (default) or
#'   `"specific"`.
#' @param research_question Character or `NULL`. Optional research context.
#' @param mode Character. Processing mode. For PDFs: `"text"` (default),
#'   `"image"`, or `"both"`. For images: `"image"` (default) or `"both"`.
#' @param filename Character or `NULL`. Optional CSV filename to save results.
#' @param model_source Character. Provider hint: `"auto"`, `"openai"`,
#'   `"anthropic"`, `"google"`, etc. Default `"auto"`.
#' @param iterations Integer. Number of passes over the data. Default `8L`.
#' @param random_state Integer or `NULL`. Random seed for reproducibility.
#' @param focus Character or `NULL`. Optional focus for extraction (e.g.,
#'   `"decisions to move"`).
#' @param chunk_delay Numeric. Seconds between API calls (rate limiting).
#'   Default `0.0`.
#'
#' @return A named list with:
#' \describe{
#'   \item{`counts_df`}{A `data.frame` of discovered categories with counts.}
#'   \item{`top_categories`}{A character vector of the top category names.}
#'   \item{`raw_top_text`}{The raw model output from the final merge step.}
#' }
#'
#' @examples
#' \dontrun{
#' result <- extract(
#'   input_data  = df$responses,
#'   description = "Why did you move to this city?",
#'   api_key     = Sys.getenv("OPENAI_API_KEY")
#' )
#' print(result$top_categories)
#' print(result$counts_df)
#' }
#'
#' @export
extract <- function(
    input_data,
    api_key,
    input_type        = "text",
    description       = "",
    max_categories    = 12L,
    categories_per_chunk = 10L,
    divisions         = 12L,
    user_model        = "gpt-4o",
    creativity        = NULL,
    specificity       = "broad",
    research_question = NULL,
    mode              = "text",
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

  result <- cat_py$extract(
    input_data           = reticulate::r_to_py(input_data),
    api_key              = api_key,
    input_type           = input_type,
    description          = description,
    max_categories       = .as_py_int(max_categories),
    categories_per_chunk = .as_py_int(categories_per_chunk),
    divisions            = .as_py_int(divisions),
    user_model           = user_model,
    creativity           = reticulate::r_to_py(creativity),
    specificity          = specificity,
    research_question    = research_question,
    mode                 = mode,
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
