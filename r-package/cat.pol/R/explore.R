#' Explore raw categories in political and policy documents
#'
#' Wraps the Python `catpol.explore()` function. Returns every category
#' string extracted from every chunk across every iteration -- with
#' duplicates intact.
#'
#' @param input_data A character vector, list, or `NULL` to fetch from a
#'   registered source. Default `NULL`.
#' @param api_key Character or `NULL`. API key for the LLM provider.
#' @param source Character or `NULL`. Registered source name.
#' @param doc_type Character or `NULL`. Filter source by document type.
#' @param since Character or `NULL`. Earliest source row date (YYYY-MM-DD).
#' @param until Character or `NULL`. Latest source row date (YYYY-MM-DD).
#' @param n Integer or `NULL`. Max number of source rows.
#' @param document_context Character. Context about the document. Default `""`.
#' @param description Character. Additional context. Default `""`.
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
#' @examples
#' \dontrun{
#' raw_cats <- explore(
#'   source     = "federal_executive_orders",
#'   since      = "2025-01-01",
#'   n          = 30L,
#'   api_key    = Sys.getenv("OPENAI_API_KEY"),
#'   user_model = "gpt-4o-mini",
#'   iterations = 4L
#' )
#' sort(table(raw_cats), decreasing = TRUE)
#' }
#' @export
explore <- function(
    input_data           = NULL,
    api_key              = NULL,
    source               = NULL,
    doc_type             = NULL,
    since                = NULL,
    until                = NULL,
    n                    = NULL,
    document_context     = "",
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
  mod <- .get_catpol()

  api_key <- cat.stack::.strip_quotes(api_key)
  if (!is.null(creativity)) creativity <- as.double(creativity)

  result <- mod$explore(
    input_data           = reticulate::r_to_py(input_data),
    api_key              = api_key,
    source               = source,
    doc_type             = doc_type,
    since                = since,
    until                = until,
    n                    = cat.stack::.as_py_int(n),
    document_context     = document_context,
    description          = description,
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

  cat.stack::.check_explore_schema(reticulate::py_to_r(result))
}
