# Check whether a specific Ollama model is installed locally

Returns `TRUE` if the named model is available in your local Ollama
installation, `FALSE` otherwise. Partial name matching is supported
(e.g. `"llama3.2"` matches `"llama3.2:latest"`).

## Usage

``` r
check_ollama_model(model, host = "localhost", port = 11434L)
```

## Arguments

- model:

  Character. Model name to look for (e.g. `"qwen2.5:7b"`).

- host:

  Character. Hostname Ollama is reachable on. Default `"localhost"`.

- port:

  Integer. Port Ollama is reachable on. Default `11434L`.

## Value

Logical scalar.

## Examples

``` r
if (FALSE) { # \dontrun{
check_ollama_model("qwen2.5:7b")
} # }
```
