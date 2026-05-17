#' Convert R models list to Python tuples
#'
#' Converts each model entry in an R list to a Python tuple as expected by
#' `cat_stack`'s `normalize_model_input()`. Handles both 3-element character
#' vectors and 4-element lists (with options dict).
#'
#' @param models An R list where each element is either:
#'   - A 3-element character vector: `c("gpt-4o", "openai", "sk-...")`
#'   - A 4-element list: `list("gpt-4o", "openai", "sk-...", list(creativity = 0.5))`
#'   - A plain character vector (single model, shorthand): `c("gpt-4o", "openai", "sk-...")`
#'
#' @return A Python list of tuples.
#' @keywords internal
#' @export
.convert_models <- function(models) {
  # Handle the case where a single model is passed as a plain character vector
  # (not wrapped in a list). Detect: if models is atomic (not a list), wrap it.
  if (is.character(models) && !is.list(models)) {
    models <- list(models)
  }

  py_tuples <- lapply(models, function(entry) {
    if (is.character(entry)) {
      # c("model", "provider", "api_key")
      if (length(entry) < 3L) {
        stop(
          "Each model entry must have at least 3 elements: ",
          "c(\"model_name\", \"provider\", \"api_key\").",
          call. = FALSE
        )
      }
      reticulate::tuple(entry[[1L]], entry[[2L]], .strip_quotes(entry[[3L]]))
    } else if (is.list(entry)) {
      if (length(entry) < 3L) {
        stop(
          "Each model entry must have at least 3 elements: ",
          "list(\"model_name\", \"provider\", \"api_key\").",
          call. = FALSE
        )
      }
      if (length(entry) >= 4L) {
        opts <- entry[[4L]]
        if (!is.list(opts)) {
          stop("The 4th element of a model entry must be a named list of options.",
               call. = FALSE)
        }
        py_opts <- reticulate::py_dict(
          keys   = as.list(names(opts)),
          values = as.list(opts)
        )
        reticulate::tuple(entry[[1L]], entry[[2L]], .strip_quotes(entry[[3L]]), py_opts)
      } else {
        reticulate::tuple(entry[[1L]], entry[[2L]], .strip_quotes(entry[[3L]]))
      }
    } else {
      stop(
        "Each element of 'models' must be a character vector or list.",
        call. = FALSE
      )
    }
  })

  reticulate::r_to_py(py_tuples)
}


#' Coerce a value to Python int (or NULL/None)
#'
#' R does not distinguish integers from doubles. This helper ensures Python
#' receives an actual `int` for parameters like `max_workers`, `max_retries`,
#' `thinking_budget`, etc.
#'
#' @param x A scalar numeric or `NULL`.
#' @return Python `int` or Python `None`.
#' @keywords internal
#' @export
.as_py_int <- function(x) {
  if (is.null(x)) return(reticulate::py_none())
  reticulate::r_to_py(as.integer(x))
}


#' Strip surrounding quotes from a string
#'
#' Many `.env` files wrap values in single or double quotes
#' (e.g. `OPENAI_API_KEY="sk-..."`). Python's `dotenv` strips these
#' automatically, but R users who read `.env` files manually may
#' inadvertently pass the quotes through. This helper removes them.
#'
#' @param x A character scalar.
#' @return `x` with leading/trailing matching quotes removed.
#' @keywords internal
#' @export
.strip_quotes <- function(x) {
  if (is.null(x) || !is.character(x)) return(x)
  gsub("^['\"]|['\"]$", "", x)
}


#' Auto-start a local Ollama server when needed
#'
#' Internal helper: if `model_source = "ollama"` or any model spec in the
#' ensemble `models` list has `"ollama"` as the provider, call
#' [ensure_ollama_running()] silently so the user doesn't see a
#' `ConnectionError: OLLAMA NOT RUNNING` from the Python side.
#'
#' @param model_source Character or `NULL` (single-model mode).
#' @param models List of model specs (ensemble mode), or `NULL`.
#' @param auto Logical. `FALSE` skips the check entirely (user opt-out).
#' @return Invisibly `NULL`.
#' @keywords internal
#' @export
.maybe_ensure_ollama <- function(model_source = NULL, models = NULL, auto = TRUE) {
  if (!isTRUE(auto)) return(invisible(NULL))

  ms <- if (is.null(model_source)) "" else tolower(as.character(model_source))
  single_ollama <- identical(ms, "ollama")

  ensemble_ollama <- FALSE
  if (!is.null(models)) {
    ensemble_ollama <- any(vapply(models, function(m) {
      provider <- NULL
      if (is.character(m) && length(m) >= 2L) provider <- m[[2L]]
      else if (is.list(m) && length(m) >= 2L) provider <- m[[2L]]
      !is.null(provider) && identical(tolower(as.character(provider)), "ollama")
    }, logical(1L)))
  }

  if (single_ollama || ensemble_ollama) {
    ensure_ollama_running(verbose = FALSE)
  }
  invisible(NULL)
}


#' Validate and normalise the `add_other` argument
#'
#' Python's default for `add_other` is `"prompt"`, which calls `input()`.
#' In non-interactive sessions, Python's `input()` raises `EOFError`, which
#' the Python code catches and treats as "no". The R default matches Python's
#' default (`"prompt"`). This helper validates the user's input.
#'
#' @param add_other `FALSE`, `TRUE`, or `"prompt"`.
#' @return The validated value (unchanged if already valid).
#' @keywords internal
#' @export
.validate_add_other <- function(add_other) {
  valid <- c(FALSE, TRUE, "prompt")
  if (!identical(add_other, FALSE) &&
      !identical(add_other, TRUE) &&
      !identical(add_other, "prompt")) {
    stop(
      "'add_other' must be FALSE, TRUE, or \"prompt\".",
      call. = FALSE
    )
  }
  add_other
}


# ---- Schema canaries ---------------------------------------------------------
#
# Each verb wrapper calls .check_<verb>_schema() on the py_to_r-converted
# result. If cat-stack ever changes its output shape (e.g. renames the
# category_N columns, drops top_categories from extract's return list,
# or changes summarize's column name), the canary emits a clear warning
# pointing at a likely schema-version mismatch -- so downstream R code
# that assumes a specific shape fails fast with a useful message rather
# than producing silently-wrong data.
#
# The canaries do NOT alter the result. They return the input unchanged
# so they can chain with the rest of the call.

.canary_msg <- function(fn, expected, got) {
  paste0(
    "Unexpected return shape from cat_stack$", fn, "(): ", expected,
    " Got: ", got, ". This usually means cat-stack changed its output ",
    "schema -- pin to a known-good version or report at ",
    "https://github.com/chrissoria/cat-llm/issues."
  )
}

#' Schema canary for classify() output
#' @keywords internal
#' @export
.check_classify_schema <- function(result) {
  if (!is.data.frame(result)) {
    warning(.canary_msg(
      "classify",
      "expected a data.frame.",
      paste(class(result), collapse = "/")
    ), call. = FALSE)
    return(result)
  }
  pat <- "^category_[0-9]+(_consensus)?$"
  if (!any(grepl(pat, names(result)))) {
    warning(.canary_msg(
      "classify",
      "no category_N or category_N_consensus columns found.",
      paste(names(result), collapse = ", ")
    ), call. = FALSE)
  }
  result
}

#' Schema canary for extract() output
#' @keywords internal
#' @export
.check_extract_schema <- function(result) {
  if (!is.list(result) || !"top_categories" %in% names(result)) {
    warning(.canary_msg(
      "extract",
      "expected a named list with a 'top_categories' element.",
      paste(class(result), collapse = "/")
    ), call. = FALSE)
  }
  result
}

#' Schema canary for explore() output
#' @keywords internal
#' @export
.check_explore_schema <- function(result) {
  if (is.null(result) || !is.character(result)) {
    warning(.canary_msg(
      "explore",
      "expected a character vector of category strings.",
      paste(class(result), collapse = "/")
    ), call. = FALSE)
  }
  result
}

#' Schema canary for summarize() output
#' @keywords internal
#' @export
.check_summarize_schema <- function(result) {
  if (!is.data.frame(result)) {
    warning(.canary_msg(
      "summarize",
      "expected a data.frame.",
      paste(class(result), collapse = "/")
    ), call. = FALSE)
    return(result)
  }
  meta <- c("input_index", "input_data", "processing_status",
            "failed_models", "pdf_path", "page_index")
  has_summary_col <- "summary" %in% names(result) ||
                     any(!names(result) %in% meta)
  if (!has_summary_col) {
    warning(.canary_msg(
      "summarize",
      "no 'summary' column and no fallback non-metadata column.",
      paste(names(result), collapse = ", ")
    ), call. = FALSE)
  }
  result
}
