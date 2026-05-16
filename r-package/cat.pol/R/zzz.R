.catpol_env <- new.env(parent = emptyenv())

.onLoad <- function(libname, pkgname) {}

.get_catpol <- function() {
  if (is.null(.catpol_env$mod)) {
    if (!reticulate::py_available(initialize = TRUE))
      stop("Python not available. Run cat.stack::install_cat_stack()", call. = FALSE)
    tryCatch(
      .catpol_env$mod <- reticulate::import("catpol"),
      error = function(e) stop(
        "Python package 'catpol' is not installed.\nRun: pip install cat-pol\n(",
        conditionMessage(e), ")", call. = FALSE
      )
    )
  }
  .catpol_env$mod
}
