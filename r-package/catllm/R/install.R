#' Install the catllm Python package
#'
#' A convenience wrapper around [reticulate::py_install()] that installs the
#' `catllm` Python package (and optional PDF-support extras) into the active
#' Python environment.
#'
#' @param method Installation method: `"auto"` (default), `"pip"`, or
#'   `"conda"`.
#' @param conda Name of the conda environment to install into (only relevant
#'   when `method = "conda"`).
#' @param pdf Logical. If `TRUE`, also installs the optional PDF-processing
#'   dependencies (`catllm[pdf]`). Default `FALSE`.
#' @param upgrade Logical. If `TRUE`, upgrades an existing `catllm`
#'   installation to the latest version. Default `FALSE`.
#' @param ... Additional arguments passed to [reticulate::py_install()].
#'
#' @return Invisibly returns `NULL`. Called for its side effect.
#'
#' @examples
#' \dontrun{
#' # Basic install
#' install_catllm()
#'
#' # Install with PDF support
#' install_catllm(pdf = TRUE)
#'
#' # Upgrade to latest
#' install_catllm(upgrade = TRUE)
#' }
#'
#' @export
install_catllm <- function(method = "auto",
                           conda  = "auto",
                           pdf    = FALSE,
                           upgrade = FALSE,
                           ...) {
  pkg <- if (isTRUE(pdf)) "catllm[pdf]" else "catllm"

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

  message("catllm installed successfully. Restart R before using the package.")
  invisible(NULL)
}
