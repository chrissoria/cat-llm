# Validate and normalise the `add_other` argument

Python's default for `add_other` is `"prompt"`, which calls `input()`.
In non-interactive sessions, Python's `input()` raises `EOFError`, which
the Python code catches and treats as "no". The R default matches
Python's default (`"prompt"`). This helper validates the user's input.

## Usage

``` r
.validate_add_other(add_other)
```

## Arguments

- add_other:

  `FALSE`, `TRUE`, or `"prompt"`.

## Value

The validated value (unchanged if already valid).
