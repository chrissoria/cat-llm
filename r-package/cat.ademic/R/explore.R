#' Explore raw categories in academic paper data
#'
#' Wraps the Python `catademic.explore()` function. Returns every category
#' string extracted from every chunk across every iteration -- with duplicates
#' intact.
#'
#' @param input_data A character vector, list, or `NULL` to fetch from academic
#'   sources. Default `NULL`.
#' @param api_key Character or `NULL`. API key for the LLM provider.
#' @param description Character. Context description. Default `""`.
#' @param journal_issn Character or `NULL`. Journal ISSN.
#' @param journal_name Character or `NULL`. Journal name.
#' @param journal_field Character or `NULL`. Academic field.
#' @param topic_name Character or `NULL`. Topic name.
#' @param topic_id Character or `NULL`. OpenAlex topic ID.
#' @param paper_limit Integer. Max papers to fetch. Default `50L`.
#' @param date_from Character or `NULL`. Start date (YYYY-MM-DD).
#' @param date_to Character or `NULL`. End date (YYYY-MM-DD).
#' @param polite_email Character or `NULL`. Email for polite API pool.
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
    description          = "",
    journal_issn         = NULL,
    journal_name         = NULL,
    journal_field        = NULL,
    topic_name           = NULL,
    topic_id             = NULL,
    paper_limit          = 50L,
    date_from            = NULL,
    date_to              = NULL,
    polite_email         = NULL,
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
  mod <- .get_catademic()

  api_key <- cat.stack:::.strip_quotes(api_key)
  if (!is.null(creativity)) creativity <- as.double(creativity)

  result <- mod$explore(
    input_data           = reticulate::r_to_py(input_data),
    api_key              = api_key,
    description          = description,
    journal_issn         = journal_issn,
    journal_name         = journal_name,
    journal_field        = journal_field,
    topic_name           = topic_name,
    topic_id             = topic_id,
    paper_limit          = cat.stack:::.as_py_int(paper_limit),
    date_from            = date_from,
    date_to              = date_to,
    polite_email         = polite_email,
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
