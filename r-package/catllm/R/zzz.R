# Package-level environment to hold the cached Python module.
# Using a dedicated environment avoids polluting the package namespace.
.catllm_env <- new.env(parent = emptyenv())

.onLoad <- function(libname, pkgname) {
  # Do NOT import the Python module at load time — importing here causes

  # R CMD check to fail on machines without Python.  The lazy import in
  # .get_catllm() handles everything when a wrapper function is first called.
}

#' Get (or lazily import) the catllm Python module
#'
#' Called at the top of every wrapper function. Imports the module on the
#' first call and returns the cached module on all subsequent calls.
#'
#' @return The catllm Python module object.
#' @noRd
.get_catllm <- function() {
  if (is.null(.catllm_env$mod)) {
    if (!reticulate::py_available(initialize = TRUE)) {
      stop(
        "Python is not available. Install Python and configure reticulate:\n",
        "  reticulate::install_python()\n",
        "  reticulate::use_virtualenv('r-catllm')\n",
        "Then install catllm:\n",
        "  catllm::install_catllm()",
        call. = FALSE
      )
    }
    tryCatch(
      .catllm_env$mod <- reticulate::import("catllm"),
      error = function(e) {
        stop(
          "The 'catllm' Python package is not installed.\n",
          "Run: catllm::install_catllm()\n",
          "(Original error: ", conditionMessage(e), ")",
          call. = FALSE
        )
      }
    )
  }
  .catllm_env$mod
}
