#' Summarize academic papers using LLMs
#'
#' Wraps the Python `catademic.summarize()` function. Generates summaries of
#' academic paper data. The Python function accepts `input_data` and passes all
#' other arguments through via `**kwargs` to `cat_stack.summarize()`.
#'
#' @param input_data A character vector, list, or `data.frame` column of paper
#'   abstracts or text.
#' @param api_key Character or `NULL`. API key for the model provider.
#' @param description Character. Context description. Default `""`.
#' @param instructions Character. Specific instructions for the summary.
#'   Default `""`.
#' @param format Character. Output format. Default `"paragraph"`.
#' @param max_length Integer or `NULL`. Max summary length. Default `NULL`.
#' @param focus Character or `NULL`. Optional focus. Default `NULL`.
#' @param user_model Character. Model name. Default `"gpt-4o"`.
#' @param model_source Character. Provider hint. Default `"auto"`.
#' @param mode Character. Processing mode. Default `"image"`.
#' @param input_mode Character or `NULL`. Explicit input mode. Default `NULL`.
#' @param input_type Character. Input type. Default `"auto"`.
#' @param pdf_dpi Integer. DPI for PDFs. Default `150L`.
#' @param creativity Numeric or `NULL`. Temperature. Default `NULL`.
#' @param thinking_budget Integer. Default `0L`.
#' @param chain_of_thought Logical. Default `TRUE`.
#' @param context_prompt Logical. Default `FALSE`.
#' @param step_back_prompt Logical. Default `FALSE`.
#' @param filename Character or `NULL`. Output filename.
#' @param save_directory Character or `NULL`. Output directory.
#' @param models List of model specs for ensemble mode. Default `NULL`.
#' @param max_workers Integer or `NULL`. Default `NULL`.
#' @param parallel Logical or `NULL`. Default `NULL`.
#' @param auto_download Logical. Default `FALSE`.
#' @param safety Logical. Default `FALSE`.
#' @param max_retries Integer. Default `5L`.
#' @param batch_retries Integer. Default `2L`.
#' @param retry_delay Numeric. Default `1.0`.
#' @param row_delay Numeric. Default `0.0`.
#' @param fail_strategy Character. Default `"partial"`.
#' @param batch_mode Logical. Default `FALSE`.
#' @param batch_poll_interval Numeric. Default `30.0`.
#' @param batch_timeout Numeric. Default `86400.0`.
#'
#' @return A `data.frame` with summarization results.
#' @export
summarize <- function(
    input_data,
    api_key              = NULL,
    description          = "",
    instructions         = "",
    format               = "paragraph",
    max_length           = NULL,
    focus                = NULL,
    user_model           = "gpt-4o",
    model_source         = "auto",
    mode                 = "image",
    input_mode           = NULL,
    input_type           = "auto",
    pdf_dpi              = 150L,
    creativity           = NULL,
    thinking_budget      = 0L,
    chain_of_thought     = TRUE,
    context_prompt       = FALSE,
    step_back_prompt     = FALSE,
    filename             = NULL,
    save_directory       = NULL,
    models               = NULL,
    max_workers          = NULL,
    parallel             = NULL,
    auto_download        = FALSE,
    safety               = FALSE,
    max_retries          = 5L,
    batch_retries        = 2L,
    retry_delay          = 1.0,
    row_delay            = 0.0,
    fail_strategy        = "partial",
    batch_mode           = FALSE,
    batch_poll_interval  = 30.0,
    batch_timeout        = 86400.0
) {
  mod <- .get_catademic()

  api_key <- cat.stack:::.strip_quotes(api_key)
  if (!is.null(creativity)) creativity <- as.double(creativity)

  py_models <- if (!is.null(models)) cat.stack:::.convert_models(models) else reticulate::py_none()

  result <- mod$summarize(
    input_data          = reticulate::r_to_py(input_data),
    api_key             = api_key,
    description         = description,
    instructions        = instructions,
    format              = format,
    max_length          = cat.stack:::.as_py_int(max_length),
    focus               = focus,
    user_model          = user_model,
    model_source        = model_source,
    mode                = mode,
    input_mode          = input_mode,
    input_type          = input_type,
    pdf_dpi             = cat.stack:::.as_py_int(pdf_dpi),
    creativity          = reticulate::r_to_py(creativity),
    thinking_budget     = cat.stack:::.as_py_int(thinking_budget),
    chain_of_thought    = chain_of_thought,
    context_prompt      = context_prompt,
    step_back_prompt    = step_back_prompt,
    filename            = filename,
    save_directory      = save_directory,
    progress_callback   = reticulate::py_none(),
    models              = py_models,
    max_workers         = cat.stack:::.as_py_int(max_workers),
    parallel            = parallel,
    auto_download       = auto_download,
    safety              = safety,
    max_retries         = cat.stack:::.as_py_int(max_retries),
    batch_retries       = cat.stack:::.as_py_int(batch_retries),
    retry_delay         = as.double(retry_delay),
    row_delay           = as.double(row_delay),
    fail_strategy       = fail_strategy,
    batch_mode          = batch_mode,
    batch_poll_interval = as.double(batch_poll_interval),
    batch_timeout       = as.double(batch_timeout)
  )

  reticulate::py_to_r(result)
}
