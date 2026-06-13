# Install the cat-stack Python package

Installs the `cat-stack` Python package into the Python environment used
by reticulate. Optionally installs PDF extras.

## Usage

``` r
install_cat_stack(
  method = "auto",
  conda = "auto",
  pdf = FALSE,
  upgrade = FALSE,
  ...
)
```

## Arguments

- method:

  Installation method passed to
  [`reticulate::py_install()`](https://rstudio.github.io/reticulate/reference/py_install.html).
  Default `"auto"`.

- conda:

  Conda environment name. Default `"auto"`.

- pdf:

  Logical. If `TRUE`, installs `cat-stack[pdf]` with PDF extras. Default
  `FALSE`.

- upgrade:

  Logical. If `TRUE`, upgrades an existing installation. Default
  `FALSE`.

- ...:

  Additional arguments passed to
  [`reticulate::py_install()`](https://rstudio.github.io/reticulate/reference/py_install.html).

## Value

Invisibly `NULL`.

## Details

The version floor is pinned to `cat-stack >= 1.6.0` — that release adds
strict-majority consensus, embedding tiebreaker, async batch mode, and
the JSON-formatter auto-consent flow that the R wrappers now expose.
Older Python installs work, but R users will hit "unexpected keyword
argument" errors from `reticulate` when the new parameters get
forwarded.

## Examples

``` r
if (FALSE) { # \dontrun{
# Standard install
install_cat_stack()

# With PDF support (installs cat-stack[pdf])
install_cat_stack(pdf = TRUE)

# Upgrade an existing install
install_cat_stack(upgrade = TRUE)
} # }
```
