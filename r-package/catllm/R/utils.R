#' Convert R models list to Python tuples
#'
#' Converts each model entry in an R list to a Python tuple as expected by
#' `catllm`'s `normalize_model_input()`. Handles both 3-element character
#' vectors and 4-element lists (with options dict).
#'
#' @param models An R list where each element is either:
#'   - A 3-element character vector: `c("gpt-4o", "openai", "sk-...")`
#'   - A 4-element list: `list("gpt-4o", "openai", "sk-...", list(creativity = 0.5))`
#'   - A plain character vector (single model, shorthand): `c("gpt-4o", "openai", "sk-...")`
#'
#' @return A Python list of tuples.
#' @noRd
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
#' @noRd
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
#' @noRd
.strip_quotes <- function(x) {
  if (is.null(x) || !is.character(x)) return(x)
  gsub("^['\"]|['\"]$", "", x)
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
#' @noRd
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
