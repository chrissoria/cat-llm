skip_if_no_cat_stack <- function() {
  if (!reticulate::py_available(initialize = TRUE)) skip("Python not available")
  tryCatch(
    reticulate::import("cat_stack"),
    error = function(e) skip("cat_stack Python package not installed")
  )
}

skip_if_no_api_key <- function(env_var) {
  key <- Sys.getenv(env_var, "")
  if (nchar(key) == 0L) skip(paste0(env_var, " not set"))
}
