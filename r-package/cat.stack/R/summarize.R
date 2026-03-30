#' Summarize text, images, or PDFs using LLMs
#'
#' Wraps the Python `cat_stack.summarize()` function. Generates summaries of
#' input data using one or more LLM models. Supports single-model and
#' multi-model (ensemble) summarization.
#'
#' @param input_data A character vector, list, or `data.frame` column. For
#'   images/PDFs, a directory path or character vector of file paths.
#' @param api_key Character or `NULL`. API key for the model provider
#'   (single-model mode). Not required when `models` is supplied. Default
#'   `NULL`.
#' @param description Character. Context description for the summarization
#'   task. Default `""`.
#' @param instructions Character. Specific instructions for the summary.
#'   Default `""`.
#' @param format Character. Output format: `"paragraph"` (default) or other
#'   supported formats.
#' @param max_length Integer or `NULL`. Maximum length of the summary. `NULL`
#'   uses the model default. Default `NULL`.
#' @param focus Character or `NULL`. Optional focus for the summary. Default
#'   `NULL`.
#' @param user_model Character. Model name. Default `"gpt-4o"`.
#' @param model_source Character. Provider hint: `"auto"`, `"openai"`,
#'   `"anthropic"`, `"google"`, etc. Default `"auto"`.
#' @param mode Character. Processing mode for images/PDFs: `"image"` (default),
#'   `"text"`, or `"both"`.
#' @param input_mode Character or `NULL`. Explicit input mode override. Default
#'   `NULL`.
#' @param input_type Character. Type of input: `"auto"` (default), `"text"`,
#'   `"image"`, or `"pdf"`.
#' @param pdf_dpi Integer. DPI for PDF page rendering. Default `150L`.
#' @param creativity Numeric or `NULL`. Temperature setting. `NULL` uses the
#'   provider default. Default `NULL`.
#' @param thinking_budget Integer. Extended thinking token budget (0 = off).
#'   Default `0L`.
#' @param chain_of_thought Logical. Enable chain-of-thought reasoning. Default
#'   `TRUE`.
#' @param context_prompt Logical. Add expert context to prompts. Default
#'   `FALSE`.
#' @param step_back_prompt Logical. Enable step-back prompting. Default
#'   `FALSE`.
#' @param filename Character or `NULL`. Output filename. Default `NULL`.
#' @param save_directory Character or `NULL`. Directory to save results.
#'   Default `NULL`.
#' @param models A list of model specifications for multi-model ensemble mode.
#'   Each element is either a 3-element character vector
#'   `c("model", "provider", "api_key")` or a 4-element list
#'   `list("model", "provider", "api_key", list(creativity = 0.5))`.
#'   Default `NULL`.
#' @param max_workers Integer or `NULL`. Max parallel workers. `NULL` = auto.
#'   Default `NULL`.
#' @param parallel Logical or `NULL`. Enable parallel processing. Default
#'   `NULL`.
#' @param auto_download Logical. Auto-download Ollama models. Default `FALSE`.
#' @param safety Logical. If `TRUE`, saves progress after each item. Default
#'   `FALSE`.
#' @param max_retries Integer. Max retries per API call. Default `5L`.
#' @param batch_retries Integer. Max retries for batch-level failures.
#'   Default `2L`.
#' @param retry_delay Numeric. Seconds between retries. Default `1.0`.
#' @param row_delay Numeric. Seconds between processing each row. Default
#'   `0.0`.
#' @param fail_strategy Character. How to handle failures: `"partial"`
#'   (default) or `"strict"`.
#' @param batch_mode Logical. Use batch processing mode. Default `FALSE`.
#' @param batch_poll_interval Numeric. Seconds between batch status polls.
#'   Default `30.0`.
#' @param batch_timeout Numeric. Maximum seconds to wait for batch completion.
#'   Default `86400.0`.
#'
#' @return A `data.frame` with summarization results.
#'
#' @examples
#' \dontrun{
#' # Single-model summarization
#' results <- summarize(
#'   input_data   = c("A long article about climate change...",
#'                     "A detailed report on economic trends..."),
#'   description  = "News articles",
#'   instructions = "Provide a 2-sentence summary",
#'   api_key      = Sys.getenv("OPENAI_API_KEY")
#' )
#'
#' # PDF summarization
#' results <- summarize(
#'   input_data = "path/to/documents/",
#'   input_type = "pdf",
#'   api_key    = Sys.getenv("OPENAI_API_KEY")
#' )
#' }
#'
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
  cat_py <- .get_cat_stack()

  api_key <- .strip_quotes(api_key)

  # Convert creativity to float when non-NULL
  if (!is.null(creativity)) creativity <- as.double(creativity)

  # Build Python models argument
  py_models <- if (!is.null(models)) .convert_models(models) else reticulate::py_none()

  result <- cat_py$summarize(
    input_data          = reticulate::r_to_py(input_data),
    api_key             = api_key,
    description         = description,
    instructions        = instructions,
    format              = format,
    max_length          = .as_py_int(max_length),
    focus               = focus,
    user_model          = user_model,
    model_source        = model_source,
    mode                = mode,
    input_mode          = input_mode,
    input_type          = input_type,
    pdf_dpi             = .as_py_int(pdf_dpi),
    creativity          = reticulate::r_to_py(creativity),
    thinking_budget     = .as_py_int(thinking_budget),
    chain_of_thought    = chain_of_thought,
    context_prompt      = context_prompt,
    step_back_prompt    = step_back_prompt,
    filename            = filename,
    save_directory      = save_directory,
    progress_callback   = reticulate::py_none(),
    models              = py_models,
    max_workers         = .as_py_int(max_workers),
    parallel            = parallel,
    auto_download       = auto_download,
    safety              = safety,
    max_retries         = .as_py_int(max_retries),
    batch_retries       = .as_py_int(batch_retries),
    retry_delay         = as.double(retry_delay),
    row_delay           = as.double(row_delay),
    fail_strategy       = fail_strategy,
    batch_mode          = batch_mode,
    batch_poll_interval = as.double(batch_poll_interval),
    batch_timeout       = as.double(batch_timeout)
  )

  reticulate::py_to_r(result)
}
