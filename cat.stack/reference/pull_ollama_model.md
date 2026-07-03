# Pull (download) an Ollama model

Downloads the named model into your local Ollama installation. Prints
the estimated model size and a resource check before downloading. Set
`auto_confirm = TRUE` to skip the interactive confirmation prompt —
useful in scripts and RMarkdown documents.

## Usage

``` r
pull_ollama_model(
  model,
  host = "localhost",
  port = 11434L,
  auto_confirm = FALSE
)
```

## Arguments

- model:

  Character. Model name to download (e.g. `"llama3.2"`, `"qwen2.5:7b"`).

- host:

  Character. Hostname Ollama is reachable on. Default `"localhost"`.

- port:

  Integer. Port Ollama is reachable on. Default `11434L`.

- auto_confirm:

  Logical. Skip the confirmation prompt. Default `FALSE`.

## Value

Invisibly returns `TRUE` on success, `FALSE` on failure.

## Examples

``` r
if (FALSE) { # \dontrun{
pull_ollama_model("llama3.2", auto_confirm = TRUE)
} # }
```
