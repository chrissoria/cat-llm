# Auto-start a local Ollama server when needed

Internal helper: if `model_source = "ollama"` or any model spec in the
ensemble `models` list has `"ollama"` as the provider, call
[`ensure_ollama_running()`](https://christophersoria.com/cat-llm/cat.stack/reference/ensure_ollama_running.md)
silently so the user doesn't see a `ConnectionError: OLLAMA NOT RUNNING`
from the Python side.

## Usage

``` r
.maybe_ensure_ollama(model_source = NULL, models = NULL, auto = TRUE)
```

## Arguments

- model_source:

  Character or `NULL` (single-model mode).

- models:

  List of model specs (ensemble mode), or `NULL`.

- auto:

  Logical. `FALSE` skips the check entirely (user opt-out).

## Value

Invisibly `NULL`.
