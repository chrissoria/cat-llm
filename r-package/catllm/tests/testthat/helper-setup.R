# Test helpers shared across all test files
# ============================================================

#' Skip a test if the catllm Python package is not available
#'
#' Checks both that Python is reachable via reticulate and that the
#' `catllm` module can be imported.
skip_if_no_catllm <- function() {
  if (!reticulate::py_available(initialize = FALSE)) {
    testthat::skip("Python is not available")
  }
  result <- tryCatch(
    {
      reticulate::import("catllm")
      TRUE
    },
    error = function(e) FALSE
  )
  if (!isTRUE(result)) {
    testthat::skip("catllm Python package is not installed")
  }
}


#' Skip a test if a required environment variable (API key) is not set
#'
#' @param env_var Character. Name of the environment variable to check
#'   (e.g. `"OPENAI_API_KEY"`).
skip_if_no_api_key <- function(env_var) {
  key <- Sys.getenv(env_var, unset = "")
  if (nchar(key) == 0L) {
    testthat::skip(paste0(env_var, " is not set"))
  }
}
