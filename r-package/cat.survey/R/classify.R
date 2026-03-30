#' Classify survey responses using LLMs
#'
#' Wraps the Python `cat_survey.classify()` function. Adds `survey_question`
#' context to the base cat.stack classification engine.
#'
#' @param input_data A character vector, list, or `data.frame` column of survey
#'   responses to classify.
#' @param categories A character vector of category names, or `"auto"` to
#'   infer categories from the data.
#' @param survey_question Character. The survey question text. Default `""`.
#' @param description Character. Additional context for the classification task.
#'   Default `""`.
#' @param add_other Logical or `"prompt"`. Controls addition of an "Other"
#'   category. Default `"prompt"`.
#' @param check_verbosity Logical. Check category descriptions. Default `TRUE`.
#' @param api_key API key for the model provider (single-model mode).
#' @param user_model Character. Model name. Default `"gpt-4o"`.
#' @param mode Character. PDF processing mode. Default `"image"`.
#' @param creativity Numeric or `NULL`. Temperature. Default `NULL`.
#' @param safety Logical. Save progress after each item. Default `FALSE`.
#' @param chain_of_verification Logical. Default `FALSE`.
#' @param chain_of_thought Logical. Default `FALSE`.
#' @param step_back_prompt Logical. Default `FALSE`.
#' @param context_prompt Logical. Default `FALSE`.
#' @param thinking_budget Integer. Extended thinking budget. Default `0L`.
#' @param example1,example2,example3,example4,example5,example6 Optional
#'   few-shot examples.
#' @param filename Character or `NULL`. Output CSV filename.
#' @param save_directory Character or `NULL`. Output directory.
#' @param model_source Character. Provider hint. Default `"auto"`.
#' @param max_categories Integer. Max categories for auto mode. Default `12L`.
#' @param categories_per_chunk Integer. Default `10L`.
#' @param divisions Integer. Default `10L`.
#' @param research_question Character or `NULL`. Optional research context.
#' @param models List of model specs for ensemble mode.
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
#'
#' @return A `data.frame` with classification results.
#' @export
classify <- function(
    input_data,
    categories,
    survey_question      = "",
    description          = "",
    add_other            = "prompt",
    check_verbosity      = TRUE,
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
    auto_download        = FALSE
) {
  mod <- .get_cat_survey()

  api_key   <- cat.stack:::.strip_quotes(api_key)
  add_other <- cat.stack:::.validate_add_other(add_other)
  if (!is.null(creativity)) creativity <- as.double(creativity)

  py_models <- if (!is.null(models)) cat.stack:::.convert_models(models) else reticulate::py_none()

  result <- mod$classify(
    input_data            = reticulate::r_to_py(input_data),
    categories            = reticulate::r_to_py(categories),
    survey_question       = survey_question,
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
    thinking_budget       = cat.stack:::.as_py_int(thinking_budget),
    example1              = example1,
    example2              = example2,
    example3              = example3,
    example4              = example4,
    example5              = example5,
    example6              = example6,
    filename              = filename,
    save_directory        = save_directory,
    model_source          = model_source,
    max_categories        = cat.stack:::.as_py_int(max_categories),
    categories_per_chunk  = cat.stack:::.as_py_int(categories_per_chunk),
    divisions             = cat.stack:::.as_py_int(divisions),
    research_question     = research_question,
    progress_callback     = reticulate::py_none(),
    models                = py_models,
    consensus_threshold   = consensus_threshold,
    use_json_schema       = use_json_schema,
    max_workers           = cat.stack:::.as_py_int(max_workers),
    fail_strategy         = fail_strategy,
    max_retries           = cat.stack:::.as_py_int(max_retries),
    batch_retries         = cat.stack:::.as_py_int(batch_retries),
    retry_delay           = as.double(retry_delay),
    row_delay             = as.double(row_delay),
    pdf_dpi               = cat.stack:::.as_py_int(pdf_dpi),
    auto_download         = auto_download
  )

  reticulate::py_to_r(result)
}
