# Strip surrounding quotes from a string

Many `.env` files wrap values in single or double quotes (e.g.
`OPENAI_API_KEY="sk-..."`). Python's `dotenv` strips these
automatically, but R users who read `.env` files manually may
inadvertently pass the quotes through. This helper removes them.

## Usage

``` r
.strip_quotes(x)
```

## Arguments

- x:

  A character scalar.

## Value

`x` with leading/trailing matching quotes removed.
