# Coerce a value to Python int (or NULL/None)

R does not distinguish integers from doubles. This helper ensures Python
receives an actual `int` for parameters like `max_workers`,
`max_retries`, `thinking_budget`, etc.

## Usage

``` r
.as_py_int(x)
```

## Arguments

- x:

  A scalar numeric or `NULL`.

## Value

Python `int` or Python `None`.
