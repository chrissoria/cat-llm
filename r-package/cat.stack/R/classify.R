#' Classify text, images, or PDFs using LLMs
#'
#' Wraps the Python `cat_stack.classify()` function. Supports both single-model
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
#' @param creativity Numeric or `NULL`. Temperature setting (0-2). `NULL` uses
#'   the provider default. Default `NULL`.
#' @param safety Logical. If `TRUE`, saves progress after each item. Default
#'   `FALSE`.
#' @param chain_of_verification Logical. Enable Chain of Verification.
#'   Empirically degrades accuracy -- provided for research only. Default
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
#'   few-shot example strings. Empirically degrades accuracy -- provided for
#'   research only.
#' @param filename Character or `NULL`. Output CSV filename. Default `NULL`.
#' @param save_directory Character or `NULL`. Directory to save results.
#'   Default `NULL`.
#' @param model_source Character. Provider hint for single-model mode:
#'   `"auto"`, `"openai"`, `"anthropic"`, `"google"`, `"mistral"`,
#'   `"perplexity"`, `"huggingface"`, `"xai"`, `"ollama"`, or
#'   `"claude-code"`. Default `"auto"` (detects from model name; falls
#'   back to `"huggingface"` for Qwen/Llama/DeepSeek-style names — use
#'   `"ollama"` explicitly to route those to a local Ollama server).
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
#'   ensemble mode. Options:
#'   \itemize{
#'     \item `"unanimous"` (default, 100% — empirically the most accurate)
#'     \item `"majority"` — STRICT majority. More than half of the models
#'       must vote positive. Ties (50/50 splits on even-model ensembles
#'       like 2-2 of 4) resolve to `"0"`. This matches sklearn's
#'       `VotingClassifier` default and standard ensemble literature.
#'       For 2-model ensembles, `"majority"` effectively requires both
#'       models to agree on positive (there's no "more than half" of 2
#'       without being all); use 3+ models for a non-degenerate
#'       majority vote, or pass `0.5` numerically to keep the old
#'       "tie favors positive" semantics.
#'     \item `"two-thirds"` — ~67% agreement, `>=` semantics.
#'     \item numeric between 0 and 1 — evaluated with `>=` semantics
#'       (the user picked a number; they get the literal interpretation).
#'   }
#'   The output `data.frame` for multi-model runs includes
#'   `category_N_agreement` columns (fraction of models that match the
#'   consensus, 0.0-1.0). For even-model ensembles with `"majority"`,
#'   pair with `embedding_tiebreaker = TRUE` to resolve true 50/50 ties
#'   via embedding-centroid similarity instead of the default
#'   "tie → 0"; that adds a `category_N_resolved_by` audit column
#'   (values: `"vote"` or `"centroid"`).
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
#'   Default `1L`. Note: composes multiplicatively with `json_retries` —
#'   a row can hit the LLM up to `(1 + json_retries) * (1 + batch_retries)`
#'   times.
#' @param json_retries Integer. Per-row retries when the LLM returns JSON
#'   that fails schema validation. On each retry the prompt appends
#'   "Respond with ONLY valid JSON". On the final attempt the formatter
#'   fallback (if enabled via `json_formatter`) fires before the row is
#'   marked failed. Default `2L`.
#' @param retry_delay Numeric. Seconds between retries. Default `1.0`.
#' @param row_delay Numeric. Seconds between processing each row (useful for
#'   rate limiting). Default `0.0`.
#' @param pdf_dpi Integer. DPI for PDF page rendering. Default `150L`.
#' @param auto_download Logical. Auto-download Ollama models. Default `FALSE`.
#' @param add_other Logical or `"prompt"`. Controls auto-addition of an
#'   "Other" catch-all category. `"prompt"` (default) asks interactively --
#'   in non-interactive sessions this silently defaults to "no". `TRUE`
#'   silently adds "Other". `FALSE` never adds it.
#' @param check_verbosity Logical. Check whether each category has a
#'   description and examples (1 API call). Default `TRUE`.
#' @param multi_label Logical. If `TRUE` (default), the prompt allows
#'   multiple categories per response. If `FALSE`, the prompt instructs the
#'   model to assign exactly one best-matching category (single-label).
#' @param batch_mode Logical. If `TRUE`, use async batch APIs for ~50%
#'   cost savings and higher rate limits. Supported providers: OpenAI,
#'   Anthropic, Google, Mistral, xAI. HuggingFace / Perplexity / Ollama
#'   fall back to synchronous calls. Incompatible with PDF / image input
#'   and with `embedding_tiebreaker`. Default `FALSE`.
#' @param batch_poll_interval Numeric. Seconds between batch-job status
#'   polls when `batch_mode = TRUE`. Default `30.0`.
#' @param batch_timeout Numeric. Maximum seconds to wait for a batch
#'   job to complete. Default `86400.0` (24 hours).
#' @param json_formatter `TRUE`, `FALSE`, or `NULL`. Three-state control
#'   for the local JSON-repair fallback model that fixes malformed LLM
#'   output before marking rows as failed. Runs only when
#'   `extract_json()` produces invalid output. The model (~1 GB) is
#'   downloaded from HuggingFace Hub on first use; requires
#'   `cat-stack[formatter]`.
#'   \itemize{
#'     \item `TRUE` — eagerly load and use the formatter (implicit
#'       consent for the ~1.5 GB dependency install if needed).
#'     \item `FALSE` — disabled; malformed rows stay as failures.
#'     \item `NULL` (default) — auto-prompt on the first malformed
#'       row. If dependencies are installed, asks
#'       "Use the formatter for this run? (Y/n)"; if not, asks
#'       "Download deps (~1.5 GB) and use the formatter? (Y/n)".
#'       Non-TTY contexts (CI, batch scripts) decline silently and
#'       print a one-time suggestion.
#'   }
#'   Auto-enabled when `two_step_classify = TRUE` or any model uses
#'   the Ollama provider.
#' @param two_step_classify `TRUE`, `FALSE`, or `NULL`. Split
#'   classification into two LLM calls — (1) natural-language reasoning,
#'   (2) JSON formatting. More reliable for weaker models (local
#'   Ollama, lower-tier API models like `gpt-4o-mini`,
#'   `claude-haiku-4-5`, `gemini-2.5-flash`) that struggle to produce
#'   strict per-category JSON in a single shot. Default `NULL`
#'   (auto-enables for Ollama models, disabled otherwise). When
#'   enabled, `json_formatter` is auto-enabled too.
#' @param embedding_tiebreaker Logical. Resolve true ensemble ties
#'   (50/50 splits at the threshold) using embedding centroids built
#'   from unanimously-agreed rows; the closer centroid wins. Companion
#'   for `consensus_threshold = "majority"` on even-model ensembles —
#'   replaces the default `"tie → 0"` with an evidence-based decision.
#'   Adds a `category_N_resolved_by` audit column to the output
#'   (values: `"vote"` or `"centroid"`). Multi-model ensemble + text
#'   input only; not supported in `batch_mode`. Requires
#'   `cat-stack[embeddings]`. Default `FALSE`.
#' @param min_centroid_size Integer. Minimum number of
#'   unanimously-agreed rows needed to build a centroid for a category
#'   when `embedding_tiebreaker = TRUE`. Categories with fewer
#'   confident rows fall back to vote-based consensus. Default `3L`.
#' @param auto_start_ollama Logical. If `TRUE` (default), automatically
#'   call [ensure_ollama_running()] when `model_source = "ollama"` or any
#'   ensemble entry uses the `"ollama"` provider. Set `FALSE` to skip
#'   the check (e.g. on CI runners where you don't want to launch
#'   Ollama).
#' @param system_prompt Character. Custom system-level instruction prepended to
#'   every classification call. Use this to apply a prompt returned by
#'   [prompt_tune()]: `system_prompt = result$system_prompt`. Default `""`.
#' @param prompt_tune Integer or `NULL`. If set, enables Automatic Prompt
#'   Optimization (APO). The value is the number of rows sampled per
#'   correction round. A browser window opens so you can correct
#'   misclassifications; the meta-LLM then rewrites the system prompt and
#'   re-classifies until accuracy converges or `tune_iterations` is reached.
#'   Categories are never modified — only the system prompt changes.
#'   Default `NULL` (disabled).
#' @param tune_iterations Integer. Number of APO optimization passes.
#'   Default `1L`.
#' @param tune_ui Character. Correction UI: `"browser"` (default) opens an
#'   interactive browser window; `"terminal"` uses the console.
#' @param tune_optimize Character. Metric to optimize: `"balanced"` (default,
#'   maximizes average of accuracy, sensitivity, and precision),
#'   `"sensitivity"` (minimize false negatives), or `"precision"` (minimize
#'   false positives).
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
#' # Single-label: force exactly one best-matching category per response
#' # (the prompt asks for the single most appropriate category instead of
#' # all that apply). Use for mutually exclusive coding frames.
#' results <- classify(
#'   input_data  = c("I love this!", "Terrible service.", "It was okay."),
#'   categories  = c("Positive", "Negative", "Neutral"),
#'   description = "Customer feedback",
#'   multi_label = FALSE,
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
#'
#' # Even-model ensemble with strict-majority + embedding tiebreaker
#' # (resolves true 50/50 ties via centroid similarity instead of
#' # the default "tie -> 0"; requires cat-stack[embeddings])
#' results <- classify(
#'   input_data           = df$responses,
#'   categories           = c("Positive", "Negative", "Neutral"),
#'   models               = list(
#'     c("gpt-4o-mini",      "openai",    Sys.getenv("OPENAI_API_KEY")),
#'     c("claude-haiku-4-5", "anthropic", Sys.getenv("ANTHROPIC_API_KEY"))
#'   ),
#'   consensus_threshold  = "majority",
#'   embedding_tiebreaker = TRUE
#' )
#'
#' # Async batch mode (50% cheaper, slower) — OpenAI / Anthropic /
#' # Google / Mistral / xAI only; not yet supported with PDFs/images
#' # or embedding_tiebreaker.
#' results <- classify(
#'   input_data = df$responses,
#'   categories = c("Positive", "Negative", "Neutral"),
#'   api_key    = Sys.getenv("OPENAI_API_KEY"),
#'   batch_mode = TRUE
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
    batch_retries        = 1L,
    json_retries         = 2L,
    retry_delay          = 1.0,
    row_delay            = 0.0,
    pdf_dpi              = 150L,
    auto_download        = FALSE,
    add_other            = "prompt",
    check_verbosity      = TRUE,
    multi_label          = TRUE,
    batch_mode           = FALSE,
    batch_poll_interval  = 30.0,
    batch_timeout        = 86400.0,
    json_formatter       = NULL,
    two_step_classify    = NULL,
    embedding_tiebreaker = FALSE,
    min_centroid_size    = 3L,
    auto_start_ollama    = TRUE,
    system_prompt        = "",
    prompt_tune          = NULL,
    tune_iterations      = 1L,
    tune_ui              = "browser",
    tune_optimize        = "balanced"
) {
  cat_py <- .get_cat_stack()

  .maybe_ensure_ollama(model_source, models, auto = auto_start_ollama)

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
    json_retries          = .as_py_int(json_retries),
    retry_delay           = as.double(retry_delay),
    row_delay             = as.double(row_delay),
    pdf_dpi               = .as_py_int(pdf_dpi),
    auto_download         = auto_download,
    add_other             = add_other,
    check_verbosity       = check_verbosity,
    multi_label           = multi_label,
    batch_mode            = batch_mode,
    batch_poll_interval   = as.double(batch_poll_interval),
    batch_timeout         = as.double(batch_timeout),
    json_formatter        = reticulate::r_to_py(json_formatter),
    two_step_classify     = reticulate::r_to_py(two_step_classify),
    embedding_tiebreaker  = embedding_tiebreaker,
    min_centroid_size     = .as_py_int(min_centroid_size),
    system_prompt         = system_prompt,
    prompt_tune           = reticulate::r_to_py(prompt_tune),
    tune_iterations       = .as_py_int(tune_iterations),
    tune_ui               = tune_ui,
    tune_optimize         = tune_optimize
  )

  .check_classify_schema(reticulate::py_to_r(result))
}
