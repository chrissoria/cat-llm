#' Install the cat-stack Python package
#'
#' Installs the `cat-stack` Python package into the Python environment used by
#' reticulate. Optionally installs PDF extras.
#'
#' The version floor is pinned to `cat-stack >= 2.0.1` — the stable 2.0 line
#' centralizes provider parameter handling (current Anthropic models no
#' longer 400 on `creativity` / `thinking_budget`), grades `thinking_budget`
#' consistently across providers, and fixes `description=` context routing
#' in `classify()` / `prompt_tune()`. Older Python installs work for old
#' models, but silently degrade on the newest Anthropic generation.
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
  # Minimum Python cat-stack version required by the R wrappers. Bump this
  # alongside the R package version when adding new Python passthroughs or
  # when the engine ships fixes the wrappers rely on.
  pkg <- if (isTRUE(pdf)) "cat-stack[pdf]>=2.0.1" else "cat-stack>=2.0.1"
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
