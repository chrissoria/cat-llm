# List locally installed Ollama models

Returns the names of all models already downloaded to your local Ollama
installation. Requires Ollama to be running (call
[`ensure_ollama_running()`](https://christophersoria.com/cat-llm/cat.stack/reference/ensure_ollama_running.md)
first, or start it manually with `ollama serve`).

## Usage

``` r
list_ollama_models(host = "localhost", port = 11434L)
```

## Arguments

- host:

  Character. Hostname Ollama is reachable on. Default `"localhost"`.

- port:

  Integer. Port Ollama is reachable on. Default `11434L`.

## Value

A character vector of model names (e.g.
`c("qwen2.5:7b", "mistral:7b")`), or an empty character vector if Ollama
is not running.

## Examples

``` r
if (FALSE) { # \dontrun{
ensure_ollama_running()
list_ollama_models()
} # }
```
