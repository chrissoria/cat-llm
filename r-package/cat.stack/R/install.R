#' Install the cat-stack Python package
#'
#' Installs the `cat-stack` Python package into the Python environment used by
#' reticulate. Optionally installs PDF extras.
#'
#' The version floor is pinned to `cat-stack >= 1.6.0` — that release adds
#' strict-majority consensus, embedding tiebreaker, async batch mode, and the
#' JSON-formatter auto-consent flow that the R wrappers now expose. Older
#' Python installs work, but R users will hit "unexpected keyword argument"
#' errors from `reticulate` when the new parameters get forwarded.
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
#' @examples
#' \dontrun{
#' # Standard install
#' install_cat_stack()
#'
#' # With PDF support (installs cat-stack[pdf])
#' install_cat_stack(pdf = TRUE)
#'
#' # Upgrade an existing install
#' install_cat_stack(upgrade = TRUE)
#' }
#' @export
install_cat_stack <- function(method = "auto", conda = "auto", pdf = FALSE,
                              upgrade = FALSE, ...) {
  # Minimum Python cat-stack version required for the new R wrapper params
  # (embedding_tiebreaker, json_formatter, batch_mode, etc.). Bump this
  # alongside the R package version when adding new Python passthroughs.
  pkg <- if (isTRUE(pdf)) "cat-stack[pdf]>=1.6.0" else "cat-stack>=1.6.0"
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
