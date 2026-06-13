# Convert R models list to Python tuples

Converts each model entry in an R list to a Python tuple as expected by
`cat_stack`'s `normalize_model_input()`. Handles both 3-element
character vectors and 4-element lists (with options dict).

## Usage

``` r
.convert_models(models)
```

## Arguments

- models:

  An R list where each element is either:

  - A 3-element character vector: `c("gpt-4o", "openai", "sk-...")`

  - A 4-element list:
    `list("gpt-4o", "openai", "sk-...", list(creativity = 0.5))`

  - A plain character vector (single model, shorthand):
    `c("gpt-4o", "openai", "sk-...")`

## Value

A Python list of tuples.
