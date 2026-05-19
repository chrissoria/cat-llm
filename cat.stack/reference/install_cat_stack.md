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
