.catvader_env <- new.env(parent = emptyenv())

.onLoad <- function(libname, pkgname) {}

.get_catvader <- function() {
  if (is.null(.catvader_env$mod)) {
    if (!reticulate::py_available(initialize = TRUE))
      stop("Python not available. Run cat.stack::install_cat_stack()", call. = FALSE)
    tryCatch(
      .catvader_env$mod <- reticulate::import("catvader"),
      error = function(e) stop(
        "Python package 'catvader' is not installed.\nRun: pip install catvader\n(",
        conditionMessage(e), ")", call. = FALSE
      )
    )
  }
  .catvader_env$mod
}
