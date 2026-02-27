#' Classify text, images, or PDFs using LLMs
#'
#' Wraps the Python `catllm.classify()` function. Supports both single-model
#' and multi-model (ensemble) classification.
#'
#' @param input_data A character vector, list of text strings, or
#'   `data.frame` column containing the items to classify. For image or PDF
#'   classification, a directory path or character vector of file paths.
#' @param categories A character vector of category names, or `"auto"` to
#'   infer categories from the data (requires `survey_question`).
#' @param api_key API key for the model provider (single-model mode).
#'   Not required when `models` is supplied.
#' @param description Character. Context description for the classification
#'   task (e.g., the survey question or image subject). Default `""`.
#' @param user_model Character. Model name to use in single-model mode.
#'   Default `"gpt-4o"`.
#' @param mode Character. PDF processing mode: `"image"` (default), `"text"`,
#'   or `"both"`.
#' @param creativity Numeric or `NULL`. Temperature setting (0–2). `NULL` uses
#'   the provider default. Default `NULL`.
#' @param safety Logical. If `TRUE`, saves progress after each item. Default
#'   `FALSE`.
#' @param chain_of_verification Logical. Enable Chain of Verification.
#'   Empirically degrades accuracy — provided for research only. Default
#'   `FALSE`.
#' @param chain_of_thought Logical. Enable chain-of-thought reasoning. Default
#'   `FALSE`.
#' @param step_back_prompt Logical. Enable step-back prompting. Default
#'   `FALSE`.
#' @param context_prompt Logical. Add expert context to prompts. Default
#'   `FALSE`.
#' @param thinking_budget Integer. Extended thinking token budget (0 = off).
#'   Default `0L`.
#' @param example1,example2,example3,example4,example5,example6 Optional
#'   few-shot example strings. Empirically degrades accuracy — provided for
#'   research only.
#' @param filename Character or `NULL`. Output CSV filename. Default `NULL`.
#' @param save_directory Character or `NULL`. Directory to save results.
#'   Default `NULL`.
#' @param model_source Character. Provider hint for single-model mode:
#'   `"auto"`, `"openai"`, `"anthropic"`, `"google"`, `"mistral"`,
#'   `"huggingface"`, `"xai"`. Default `"auto"`.
#' @param max_categories Integer. Maximum number of categories when
#'   `categories = "auto"`. Default `12L`.
#' @param categories_per_chunk Integer. Categories extracted per chunk when
#'   `categories = "auto"`. Default `10L`.
#' @param divisions Integer. Number of data chunks when `categories = "auto"`.
#'   Default `10L`.
#' @param research_question Character or `NULL`. Optional research context.
#'   Default `NULL`.
#' @param models A list of model specifications for multi-model ensemble mode.
#'   Each element is either a 3-element character vector
#'   `c("model", "provider", "api_key")` or a 4-element list
#'   `list("model", "provider", "api_key", list(creativity = 0.5))`.
#'   When `models` is supplied, `api_key` and `user_model` are ignored.
#' @param consensus_threshold Character or numeric. Agreement threshold for
#'   ensemble mode. Options: `"unanimous"` (default, 100%), `"majority"`
#'   (50%), `"two-thirds"` (67%), or a numeric value between 0 and 1.
#' @param survey_question Character. The survey question text (used when
#'   `categories = "auto"`). Default `""`.
#' @param use_json_schema Logical. Use JSON schema for structured output.
#'   Default `TRUE`.
#' @param max_workers Integer or `NULL`. Max parallel workers. `NULL` = auto.
#'   Default `NULL`.
#' @param fail_strategy Character. How to handle failures: `"partial"`
#'   (default) or `"strict"`.
#' @param max_retries Integer. Max retries per API call. Default `5L`.
#' @param batch_retries Integer. Max retries for batch-level failures.
#'   Default `2L`.
#' @param retry_delay Numeric. Seconds between retries. Default `1.0`.
#' @param row_delay Numeric. Seconds between processing each row (useful for
#'   rate limiting). Default `0.0`.
#' @param pdf_dpi Integer. DPI for PDF page rendering. Default `150L`.
#' @param auto_download Logical. Auto-download Ollama models. Default `FALSE`.
#' @param add_other Logical or `"prompt"`. Controls auto-addition of an
#'   "Other" catch-all category. `"prompt"` (default) asks interactively —
#'   in non-interactive sessions this silently defaults to "no". `TRUE`
#'   silently adds "Other". `FALSE` never adds it.
#' @param check_verbosity Logical. Check whether each category has a
#'   description and examples (1 API call). Default `TRUE`.
#'
#' @return A `data.frame` with one row per input item and classification
#'   columns. In single-model mode the columns are the category names. In
#'   ensemble mode additional `consensus_*` and `agreement_*` columns are
#'   included.
#'
#' @examples
#' \dontrun{
#' # Single-model classification
#' results <- classify(
#'   input_data  = c("I love this!", "Terrible service.", "It was okay."),
#'   categories  = c("Positive", "Negative", "Neutral"),
#'   description = "Customer feedback",
#'   api_key     = Sys.getenv("OPENAI_API_KEY")
#' )
#'
#' # Multi-model ensemble
#' results <- classify(
#'   input_data  = df$responses,
#'   categories  = c("Positive", "Negative", "Neutral"),
#'   models      = list(
#'     c("gpt-4o",              "openai",    Sys.getenv("OPENAI_API_KEY")),
#'     c("claude-sonnet-4-5-20250929", "anthropic", Sys.getenv("ANTHROPIC_API_KEY"))
#'   ),
#'   consensus_threshold = "unanimous"
#' )
#' }
#'
#' @export
classify <- function(
    input_data,
    categories,
    api_key              = NULL,
    description          = "",
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
    survey_question      = "",
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
  cat_py <- .get_catllm()

  api_key   <- .strip_quotes(api_key)
  add_other <- .validate_add_other(add_other)

  # Convert creativity to float when non-NULL
  if (!is.null(creativity)) creativity <- as.double(creativity)

  # Build Python models argument
  py_models <- if (!is.null(models)) .convert_models(models) else reticulate::py_none()

  result <- cat_py$classify(
    input_data            = reticulate::r_to_py(input_data),
    categories            = reticulate::r_to_py(categories),
    api_key               = api_key,
    description           = description,
    user_model            = user_model,
    mode                  = mode,
    creativity            = reticulate::r_to_py(creativity),
    safety                = safety,
    chain_of_verification = chain_of_verification,
    chain_of_thought      = chain_of_thought,
    step_back_prompt      = step_back_prompt,
    context_prompt        = context_prompt,
    thinking_budget       = .as_py_int(thinking_budget),
    example1              = example1,
    example2              = example2,
    example3              = example3,
    example4              = example4,
    example5              = example5,
    example6              = example6,
    filename              = filename,
    save_directory        = save_directory,
    model_source          = model_source,
    max_categories        = .as_py_int(max_categories),
    categories_per_chunk  = .as_py_int(categories_per_chunk),
    divisions             = .as_py_int(divisions),
    research_question     = research_question,
    progress_callback     = reticulate::py_none(),
    models                = py_models,
    consensus_threshold   = consensus_threshold,
    survey_question       = survey_question,
    use_json_schema       = use_json_schema,
    max_workers           = .as_py_int(max_workers),
    fail_strategy         = fail_strategy,
    max_retries           = .as_py_int(max_retries),
    batch_retries         = .as_py_int(batch_retries),
    retry_delay           = as.double(retry_delay),
    row_delay             = as.double(row_delay),
    pdf_dpi               = .as_py_int(pdf_dpi),
    auto_download         = auto_download,
    add_other             = add_other,
    check_verbosity       = check_verbosity
  )

  reticulate::py_to_r(result)
}
