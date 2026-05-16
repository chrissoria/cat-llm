#' Summarize web content using LLMs
#'
#' Wraps the Python `catweb.summarize()` function. Accepts URLs (auto-fetched)
#' or raw text. Web context (source domain, content type, metadata) is
#' injected into the summarization prompt.
#'
#' @param input_data Data to summarize: URLs, text, or `data.frame` column.
#' @param source_domain Character or `NULL`. Source domain context.
#' @param content_type Character or `NULL`. Content type context.
#' @param web_metadata Named list or `NULL`. Additional metadata.
#' @param timeout Integer. URL fetch timeout (seconds). Default `30L`.
#' @param api_key Character or `NULL`. API key for the LLM provider.
#' @param description Character. Default `""`.
#' @param instructions Character. Specific instructions for the summary.
#'   Default `""`.
#' @param format Character. Default `"paragraph"`.
#' @param max_length Integer or `NULL`. Default `NULL`.
#' @param focus Character or `NULL`. Default `NULL`.
#' @param user_model Character. Default `"gpt-4o"`.
#' @param model_source Character. Default `"auto"`.
#' @param mode Character. Default `"image"`.
#' @param input_mode Character or `NULL`. Default `NULL`.
#' @param input_type Character. Default `"auto"`.
#' @param pdf_dpi Integer. Default `150L`.
#' @param creativity Numeric or `NULL`. Default `NULL`.
#' @param thinking_budget Integer. Default `0L`.
#' @param chain_of_thought Logical. Default `TRUE`.
#' @param context_prompt Logical. Default `FALSE`.
#' @param step_back_prompt Logical. Default `FALSE`.
#' @param filename Character or `NULL`.
#' @param save_directory Character or `NULL`.
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
    input_data           = NULL,
    source_domain        = NULL,
    content_type         = NULL,
    web_metadata         = NULL,
    timeout              = 30L,
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
  mod <- .get_catweb()

  api_key <- cat.stack::.strip_quotes(api_key)
  if (!is.null(creativity)) creativity <- as.double(creativity)

  py_models       <- if (!is.null(models)) cat.stack::.convert_models(models) else reticulate::py_none()
  py_web_metadata <- if (!is.null(web_metadata)) reticulate::r_to_py(web_metadata) else NULL

  result <- mod$summarize(
    input_data          = reticulate::r_to_py(input_data),
    source_domain       = source_domain,
    content_type        = content_type,
    web_metadata        = py_web_metadata,
    timeout             = cat.stack::.as_py_int(timeout),
    api_key             = api_key,
    description         = description,
    instructions        = instructions,
    format              = format,
    max_length          = cat.stack::.as_py_int(max_length),
    focus               = focus,
    user_model          = user_model,
    model_source        = model_source,
    mode                = mode,
    input_mode          = input_mode,
    input_type          = input_type,
    pdf_dpi             = cat.stack::.as_py_int(pdf_dpi),
    creativity          = reticulate::r_to_py(creativity),
    thinking_budget     = cat.stack::.as_py_int(thinking_budget),
    chain_of_thought    = chain_of_thought,
    context_prompt      = context_prompt,
    step_back_prompt    = step_back_prompt,
    filename            = filename,
    save_directory      = save_directory,
    progress_callback   = reticulate::py_none(),
    models              = py_models,
    max_workers         = cat.stack::.as_py_int(max_workers),
    parallel            = parallel,
    auto_download       = auto_download,
    safety              = safety,
    max_retries         = cat.stack::.as_py_int(max_retries),
    batch_retries       = cat.stack::.as_py_int(batch_retries),
    retry_delay         = as.double(retry_delay),
    row_delay           = as.double(row_delay),
    fail_strategy       = fail_strategy,
    batch_mode          = batch_mode,
    batch_poll_interval = as.double(batch_poll_interval),
    batch_timeout       = as.double(batch_timeout)
  )

  reticulate::py_to_r(result)
}
