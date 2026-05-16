.catweb_env <- new.env(parent = emptyenv())

.onLoad <- function(libname, pkgname) {}

.get_catweb <- function() {
  if (is.null(.catweb_env$mod)) {
    if (!reticulate::py_available(initialize = TRUE))
      stop("Python not available. Run cat.stack::install_cat_stack()", call. = FALSE)
    tryCatch(
      .catweb_env$mod <- reticulate::import("catweb"),
      error = function(e) stop(
        "Python package 'catweb' is not installed.\nRun: pip install cat-web\n(",
        conditionMessage(e), ")", call. = FALSE
      )
    )
  }
  .catweb_env$mod
}
