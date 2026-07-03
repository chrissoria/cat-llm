# Ensure a local Ollama server is running

Checks whether an Ollama server is reachable at `host:port`. If not,
attempts to start it using the platform-appropriate command and polls
until the server responds (or `timeout` is reached). Call this once at
the top of an R session before classifying with
`model_source = "ollama"`.

## Usage

``` r
ensure_ollama_running(
  auto_start = TRUE,
  timeout = 30,
  host = "localhost",
  port = 11434L,
  verbose = TRUE
)
```

## Arguments

- auto_start:

  Logical. If `TRUE` (default), attempt to launch Ollama when not
  running. If `FALSE`, just check and error if not running.

- timeout:

  Numeric. Seconds to wait for Ollama to become ready after
  `auto_start`. Default `30`.

- host:

  Character. Hostname Ollama is reachable on. Default `"localhost"`.

- port:

  Integer. Port Ollama is reachable on. Default `11434L`.

- verbose:

  Logical. Print status messages. Default `TRUE`.

## Value

Invisibly returns `TRUE` when Ollama is running.

## Details

Platform start commands:

- **macOS** — `open -a Ollama` (launches the Ollama.app daemon). Falls
  back to `ollama serve` if the app is not installed.

- **Linux** — `ollama serve` (run in a detached process).

- **Windows** — `ollama serve`.

If Ollama is not installed, the function returns a clear error message
linking to <https://ollama.com>.

## Examples

``` r
if (FALSE) { # \dontrun{
# Ensure Ollama is up before classifying with a local model
ensure_ollama_running()

results <- classify(
  input_data   = c("text 1", "text 2"),
  categories   = c("Positive", "Negative", "Neutral"),
  user_model   = "qwen2.5:7b",
  model_source = "ollama"
)

# Just check without auto-starting
ensure_ollama_running(auto_start = FALSE)
} # }
```
