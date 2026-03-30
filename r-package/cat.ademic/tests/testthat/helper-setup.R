skip_if_no_catademic <- function() {
  if (!reticulate::py_available(initialize = TRUE)) skip("Python not available")
  tryCatch(
    reticulate::import("catademic"),
    error = function(e) skip("catademic Python package not installed")
  )
}

skip_if_no_api_key <- function(env_var) {
  key <- Sys.getenv(env_var, "")
  if (nchar(key) == 0L) skip(paste0(env_var, " not set"))
}
