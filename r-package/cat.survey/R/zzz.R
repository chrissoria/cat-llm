.catsurvey_env <- new.env(parent = emptyenv())

.onLoad <- function(libname, pkgname) {}

.get_cat_survey <- function() {
  if (is.null(.catsurvey_env$mod)) {
    if (!reticulate::py_available(initialize = TRUE))
      stop("Python not available. Run cat.stack::install_cat_stack()", call. = FALSE)
    tryCatch(
      .catsurvey_env$mod <- reticulate::import("cat_survey"),
      error = function(e) stop(
        "Python package 'cat-survey' is not installed.\nRun: pip install cat-survey\n(",
        conditionMessage(e), ")", call. = FALSE
      )
    )
  }
  .catsurvey_env$mod
}
