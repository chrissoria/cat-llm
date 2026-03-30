.catademic_env <- new.env(parent = emptyenv())

.onLoad <- function(libname, pkgname) {}

.get_catademic <- function() {
  if (is.null(.catademic_env$mod)) {
    if (!reticulate::py_available(initialize = TRUE))
      stop("Python not available. Run cat.stack::install_cat_stack()", call. = FALSE)
    tryCatch(
      .catademic_env$mod <- reticulate::import("catademic"),
      error = function(e) stop(
        "Python package 'catademic' is not installed.\nRun: pip install catademic\n(",
        conditionMessage(e), ")", call. = FALSE
      )
    )
  }
  .catademic_env$mod
}
