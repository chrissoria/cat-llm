#' Optimize a classification prompt with human-in-the-loop feedback
#'
#' Wraps the Python `catstack.prompt_tune()` function. Runs a
#' coordinate-descent loop: classifies a small sample, asks you to
#' correct the model's output, then has a meta-LLM rewrite the
#' classification instructions for each category that had errors.
#' Returns the best system prompt found plus per-iteration metrics.
#'
#' This function is **interactive** — you'll be asked to review and
#' correct the model's labels at least once. From an R session, the
#' default `ui = "terminal"` reads your corrections from stdin (works
#' in R, Rscript, and most IDE consoles). `ui = "browser"` opens a
#' local web page with checkboxes; depending on your R setup this may
#' or may not auto-launch the browser, so terminal is the safer
#' default for R users.
#'
#' Use the returned `system_prompt` with [classify()] via the
#' `system_prompt =` argument to apply the tuned instructions.
#'
#' @param input_data A character vector, list, or `data.frame` column
#'   of items to classify during tuning.
#' @param categories A character vector of category names. The labels
#'   themselves are never modified by tuning — only the classification
#'   instructions change.
#' @param api_key Character or `NULL`. API key for the LLM provider.
#' @param user_model Character. Model name. Default `"gpt-4o"`.
#' @param model_source Character. Provider hint. Default `"auto"`.
#' @param models List of model specs for ensemble mode (each
#'   `c(model, provider, api_key)`). Overrides `user_model`/`api_key`/
#'   `model_source` if given. Default `NULL`.
#' @param description Character. Context description. Default `""`.
#' @param survey_question Character. Survey question for context.
#'   Default `""`.
#' @param sample_size Integer. Items to test per iteration. Default `10L`.
#' @param max_iterations Integer. Max instruction attempts per
#'   category. Default `3L`.
#' @param multi_label Logical. Multi-label classification. Default `TRUE`.
#' @param creativity Numeric or `NULL`. Temperature. Default `NULL`.
#' @param use_json_schema Logical. Default `TRUE`.
#' @param consensus_threshold Character or numeric. For ensemble mode.
#'   Default `"unanimous"`.
#' @param max_retries Integer. Default `5L`.
#' @param input_mode Character or `NULL`. Input mode override.
#' @param ui Character. Review interface for corrections.
#'   `"terminal"` (default in R) reads from stdin. `"browser"` opens a
#'   local web page with checkboxes (may not auto-launch from R sessions).
#' @param optimize Character. Which metric to maximize.
#'   `"balanced"` (default), `"precision"`, or `"sensitivity"`.
#' @param add_other Logical or `"prompt"`. Controls auto-addition of an
#'   "Other" catch-all category. Default `"prompt"`.
#' @param thinking_budget Integer. Default `0L`.
#' @param auto_start_ollama Logical. If `TRUE` (default), automatically
#'   call [ensure_ollama_running()] when `model_source = "ollama"` or
#'   any ensemble entry uses the `"ollama"` provider. Set `FALSE` to
#'   skip the check.
#'
#' @return A named list with components:
#'   * `system_prompt` — the optimized system prompt (best found)
#'   * `iterations` — list of per-iteration records (label,
#'     system_prompt, metrics, per_category, total_flips)
#'   * `per_category_summary` — per-category metrics from the
#'     best-scoring iteration
#'
#' @examples
#' \dontrun{
#' result <- prompt_tune(
#'   input_data    = df$open_response,
#'   categories    = c("Positive", "Negative", "Neutral"),
#'   api_key       = Sys.getenv("OPENAI_API_KEY"),
#'   user_model    = "gpt-4o-mini",
#'   sample_size   = 10L,
#'   max_iterations = 3L,
#'   ui            = "terminal"
#' )
#'
#' # Inspect the optimized prompt
#' cat(result$system_prompt)
#'
#' # Use it in classify() via the system_prompt argument
#' results <- classify(
#'   input_data    = df$open_response,
#'   categories    = c("Positive", "Negative", "Neutral"),
#'   api_key       = Sys.getenv("OPENAI_API_KEY"),
#'   user_model    = "gpt-4o-mini",
#'   system_prompt = result$system_prompt
#' )
#' }
#' @export
prompt_tune <- function(
    input_data,
    categories,
    api_key             = NULL,
    user_model          = "gpt-4o",
    model_source        = "auto",
    models              = NULL,
    description         = "",
    survey_question     = "",
    sample_size         = 10L,
    max_iterations      = 3L,
    multi_label         = TRUE,
    creativity          = NULL,
    use_json_schema     = TRUE,
    consensus_threshold = "unanimous",
    max_retries         = 5L,
    input_mode          = NULL,
    ui                  = "terminal",
    optimize            = "balanced",
    add_other           = "prompt",
    thinking_budget     = 0L,
    auto_start_ollama   = TRUE
) {
  mod <- .get_cat_stack()

  .maybe_ensure_ollama(model_source, models, auto = auto_start_ollama)

  api_key   <- .strip_quotes(api_key)
  add_other <- .validate_add_other(add_other)
  if (!is.null(creativity)) creativity <- as.double(creativity)

  py_models <- if (!is.null(models)) .convert_models(models) else reticulate::py_none()

  result <- mod$prompt_tune(
    input_data          = reticulate::r_to_py(input_data),
    categories          = reticulate::r_to_py(categories),
    api_key             = api_key,
    user_model          = user_model,
    model_source        = model_source,
    models              = py_models,
    description         = description,
    survey_question     = survey_question,
    sample_size         = .as_py_int(sample_size),
    max_iterations      = .as_py_int(max_iterations),
    multi_label         = multi_label,
    creativity          = reticulate::r_to_py(creativity),
    use_json_schema     = use_json_schema,
    consensus_threshold = consensus_threshold,
    max_retries         = .as_py_int(max_retries),
    input_mode          = input_mode,
    ui                  = ui,
    optimize            = optimize,
    add_other           = add_other,
    thinking_budget     = .as_py_int(thinking_budget)
  )

  reticulate::py_to_r(result)
}
