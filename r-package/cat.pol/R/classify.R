#' Classify political and policy documents using LLMs
#'
#' Wraps the Python `catpol.classify()` function. Can classify either raw
#' text (via `input_data`) or pull directly from a registered political
#' data source (via `source`). All catstack classification arguments are
#' supported.
#'
#' @param categories A character vector of category names.
#' @param input_data A character vector, list, or `data.frame` column, or
#'   `NULL` to fetch from a registered source. Default `NULL`.
#' @param source Character or `NULL`. Registered source name (e.g.
#'   `"city_san_diego"`, `"federal_laws"`, `"federal_executive_orders"`,
#'   `"social_trump_truth"`). Use [list_sources()] for all options.
#' @param doc_type Character or `NULL`. Filter source by document type
#'   (e.g. `"ordinance"`, `"resolution"`).
#' @param since Character or `NULL`. Earliest source row date (YYYY-MM-DD).
#' @param until Character or `NULL`. Latest source row date (YYYY-MM-DD).
#' @param n Integer or `NULL`. Max number of source rows to classify.
#' @param document_context Character. Context about the policy document
#'   being analyzed. Default `""`.
#' @param description Character. Additional context description. Default `""`.
#' @param api_key Character or `NULL`. API key for the LLM provider.
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
#' # Pull recent San Diego ordinances from a registered source
#' results <- classify(
#'   source     = "city_san_diego",
#'   doc_type   = "ordinance",
#'   since      = "2024-01-01",
#'   n          = 50L,
#'   categories = c("Housing", "Public Safety", "Finance",
#'                  "Infrastructure", "Health"),
#'   api_key    = Sys.getenv("OPENAI_API_KEY"),
#'   user_model = "gpt-4o-mini"
#' )
#'
#' # Or classify your own text directly
#' results <- classify(
#'   input_data = df$bill_text,
#'   categories = c("Housing", "Public Safety", "Finance"),
#'   api_key    = Sys.getenv("OPENAI_API_KEY")
#' )
#' }
#' @export
classify <- function(
    categories,
    input_data           = NULL,
    source               = NULL,
    doc_type             = NULL,
    since                = NULL,
    until                = NULL,
    n                    = NULL,
    document_context     = "",
    description          = "",
    api_key              = NULL,
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
    check_verbosity      = TRUE,
    prompt_tune          = NULL,
    tune_iterations      = 1L,
    tune_ui              = "browser",
    tune_optimize        = "balanced"
) {
  mod <- .get_catpol()

  api_key   <- cat.stack::.strip_quotes(api_key)
  add_other <- cat.stack::.validate_add_other(add_other)
  if (!is.null(creativity)) creativity <- as.double(creativity)

  py_models <- if (!is.null(models)) cat.stack::.convert_models(models) else reticulate::py_none()

  result <- mod$classify(
    input_data            = reticulate::r_to_py(input_data),
    categories            = reticulate::r_to_py(categories),
    source                = source,
    doc_type              = doc_type,
    since                 = since,
    until                 = until,
    n                     = cat.stack::.as_py_int(n),
    document_context      = document_context,
    description           = description,
    add_other             = add_other,
    check_verbosity       = check_verbosity,
    api_key               = api_key,
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
    filename              = filename,
    save_directory        = save_directory,
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
    prompt_tune           = reticulate::r_to_py(prompt_tune),
    tune_iterations       = cat.stack::.as_py_int(tune_iterations),
    tune_ui               = tune_ui,
    tune_optimize         = tune_optimize
  )

  cat.stack::.check_classify_schema(reticulate::py_to_r(result))
}
