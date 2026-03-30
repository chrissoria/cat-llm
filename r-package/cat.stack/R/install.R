#' Install the cat-stack Python package
#'
#' Installs the `cat-stack` Python package into the Python environment used by
#' reticulate. Optionally installs PDF extras.
#'
#' @param method Installation method passed to [reticulate::py_install()].
#'   Default `"auto"`.
#' @param conda Conda environment name. Default `"auto"`.
#' @param pdf Logical. If `TRUE`, installs `cat-stack[pdf]` with PDF extras.
#'   Default `FALSE`.
#' @param upgrade Logical. If `TRUE`, upgrades an existing installation.
#'   Default `FALSE`.
#' @param ... Additional arguments passed to [reticulate::py_install()].
#'
#' @return Invisibly `NULL`.
#' @export
install_cat_stack <- function(method = "auto", conda = "auto", pdf = FALSE,
                              upgrade = FALSE, ...) {
  pkg <- if (isTRUE(pdf)) "cat-stack[pdf]" else "cat-stack"
  pip_opts <- character(0)
  if (isTRUE(upgrade)) pip_opts <- c(pip_opts, "--upgrade")
  message("Installing Python package: ", pkg)
  reticulate::py_install(
    packages    = pkg,
    method      = method,
    conda       = conda,
    pip         = TRUE,
    pip_options = pip_opts,
    ...
  )
  message("cat-stack installed successfully. Restart R before using the package.")
  invisible(NULL)
}
