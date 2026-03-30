.catstack_env <- new.env(parent = emptyenv())

.onLoad <- function(libname, pkgname) {}

.get_cat_stack <- function() {
  if (is.null(.catstack_env$mod)) {
    if (!reticulate::py_available(initialize = TRUE)) {
      stop("Python is not available. Install Python and configure reticulate:\n  reticulate::install_python()\nThen install cat-stack:\n  cat.stack::install_cat_stack()", call. = FALSE)
    }
    tryCatch(
      .catstack_env$mod <- reticulate::import("cat_stack"),
      error = function(e) {
        stop("The 'cat-stack' Python package is not installed.\nRun: cat.stack::install_cat_stack()\n(Original error: ", conditionMessage(e), ")", call. = FALSE)
      }
    )
  }
  .catstack_env$mod
}
