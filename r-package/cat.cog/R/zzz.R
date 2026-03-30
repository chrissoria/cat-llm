.catcog_env <- new.env(parent = emptyenv())

.onLoad <- function(libname, pkgname) {}

.get_cat_cog <- function() {
  if (is.null(.catcog_env$mod)) {
    if (!reticulate::py_available(initialize = TRUE))
      stop("Python not available. Run cat.stack::install_cat_stack()", call. = FALSE)
    tryCatch(
      .catcog_env$mod <- reticulate::import("cat_cog"),
      error = function(e) stop(
        "Python package 'cat-cog' is not installed.\nRun: pip install cat-cog\n(",
        conditionMessage(e), ")", call. = FALSE
      )
    )
  }
  .catcog_env$mod
}
