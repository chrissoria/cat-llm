# Attach all CatLLM ecosystem packages

Explicitly loads all domain packages. Normally this happens
automatically when
[`library(cat.llm)`](https://christophersoria.com/cat-llm/cat.llm/) is
called, but this function can be used to reload after detaching.

## Usage

``` r
catllm_attach()
```

## Value

Invisibly returns a character vector of attached package names.

## Examples

``` r
if (FALSE) { # \dontrun{
# Normally this happens automatically on `library(cat.llm)`.
# Call manually to re-attach after detaching:
catllm_attach()
} # }
```
