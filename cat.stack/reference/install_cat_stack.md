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

The version floor is pinned to `cat-stack >= 2.0.1` — the stable 2.0
line centralizes provider parameter handling (current Anthropic models
no longer 400 on `creativity` / `thinking_budget`), grades
`thinking_budget` consistently across providers, and fixes
`description=` context routing in
[`classify()`](https://christophersoria.com/cat-llm/cat.stack/reference/classify.md)
/
[`prompt_tune()`](https://christophersoria.com/cat-llm/cat.stack/reference/prompt_tune.md).
Older Python installs work for old models, but silently degrade on the
newest Anthropic generation.

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
