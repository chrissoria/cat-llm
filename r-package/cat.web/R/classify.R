#' Classify web content using LLMs
#'
#' Wraps the Python `catweb.classify()` function. Accepts URLs (auto-fetched
#' to text) or raw text strings. Injects web context (source domain, content
#' type, metadata) into the classification prompt.
#'
#' @param categories A character vector of category names.
#' @param input_data A character vector / list / `data.frame` column of
#'   URLs or text strings. Default `NULL`.
#' @param api_key Character or `NULL`. API key for the LLM provider.
#' @param source_domain Character or `NULL`. Source domain injected into the
#'   prompt as context (e.g. `"nytimes.com"`).
#' @param content_type Character or `NULL`. Content type (e.g. `"news article"`,
#'   `"blog post"`).
#' @param web_metadata Named list or `NULL`. Additional metadata injected
#'   into the prompt.
#' @param description Character. Context description. Default `""`.
#' @param filename Character or `NULL`. Output CSV filename.
#' @param save_directory Character or `NULL`. Output directory.
#' @param timeout Integer. URL fetch timeout (seconds). Default `30L`.
#' @param user_model Character. Model name. Default `"gpt-4o"`.
#' @param mode Character. Processing mode. Default `"image"`.
#' @param creativity Numeric or `NULL`. Temperature. Default `NULL`.
#' @param safety Logical. Default `FALSE`.
#' @param chain_of_verification Logical. Default `FALSE`.
#' @param chain_of_thought Logical. Default `FALSE`.
#' @param step_back_prompt Logical. Default `FALSE`.
#' @param context_prompt Logical. Default `FALSE`.
#' @param thinking_budget Integer. Default `0L`.
#' @param example1,example2,example3,example4,example5,example6 Optional
#'   few-shot examples.
#' @param model_source Character. Default `"auto"`.
#' @param max_categories Integer. Default `12L`.
#' @param categories_per_chunk Integer. Default `10L`.
#' @param divisions Integer. Default `10L`.
#' @param research_question Character or `NULL`.
#' @param models List of model specs for ensemble mode. Default `NULL`.
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
#' @examples
#' \dontrun{
#' # Classify a list of URLs (auto-fetched to text)
#' results <- classify(
#'   categories    = c("News", "Opinion", "Tutorial"),
#'   input_data    = c("https://example.com/article-1",
#'                     "https://example.com/article-2"),
#'   source_domain = "example.com",
#'   content_type  = "blog post",
#'   api_key       = Sys.getenv("OPENAI_API_KEY"),
#'   user_model    = "gpt-4o-mini"
#' )
#'
#' # Or classify raw text (no fetching)
#' results <- classify(
#'   categories = c("News", "Opinion", "Tutorial"),
#'   input_data = df$article_text,
#'   api_key    = Sys.getenv("OPENAI_API_KEY")
#' )
#' }
#' @export
classify <- function(
    categories,
    input_data           = NULL,
    api_key              = NULL,
    source_domain        = NULL,
    content_type         = NULL,
    web_metadata         = NULL,
    description          = "",
    filename             = NULL,
    save_directory       = NULL,
    timeout              = 30L,
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
  mod <- .get_catweb()

  api_key   <- cat.stack::.strip_quotes(api_key)
  add_other <- cat.stack::.validate_add_other(add_other)
  if (!is.null(creativity)) creativity <- as.double(creativity)

  py_models       <- if (!is.null(models)) cat.stack::.convert_models(models) else reticulate::py_none()
  py_web_metadata <- if (!is.null(web_metadata)) reticulate::r_to_py(web_metadata) else NULL

  result <- mod$classify(
    categories            = reticulate::r_to_py(categories),
    input_data            = reticulate::r_to_py(input_data),
    api_key               = api_key,
    source_domain         = source_domain,
    content_type          = content_type,
    web_metadata          = py_web_metadata,
    description           = description,
    filename              = filename,
    save_directory        = save_directory,
    timeout               = cat.stack::.as_py_int(timeout),
    user_model            = user_model,
    mode                  = mode,
    creativity            = reticulate::r_to_py(creativity),
    safety                = safety,
    chain_of_verification = chain_of_verification,
    chain_of_thought      = chain_of_thought,
    step_back_prompt      = step_back_prompt,
    context_prompt        = context_prompt,
    thinking_budget       = cat.stack::.as_py_int(thinking_budget),
    example1              = example1,
    example2              = example2,
    example3              = example3,
    example4              = example4,
    example5              = example5,
    example6              = example6,
    model_source          = model_source,
    max_categories        = cat.stack::.as_py_int(max_categories),
    categories_per_chunk  = cat.stack::.as_py_int(categories_per_chunk),
    divisions             = cat.stack::.as_py_int(divisions),
    research_question     = research_question,
    progress_callback     = reticulate::py_none(),
    models                = py_models,
    consensus_threshold   = consensus_threshold,
    use_json_schema       = use_json_schema,
    max_workers           = cat.stack::.as_py_int(max_workers),
    fail_strategy         = fail_strategy,
    max_retries           = cat.stack::.as_py_int(max_retries),
    batch_retries         = cat.stack::.as_py_int(batch_retries),
    retry_delay           = as.double(retry_delay),
    row_delay             = as.double(row_delay),
    pdf_dpi               = cat.stack::.as_py_int(pdf_dpi),
    auto_download         = auto_download,
    add_other             = add_other,
    check_verbosity       = check_verbosity
  )

  cat.stack::.check_classify_schema(reticulate::py_to_r(result))
}
