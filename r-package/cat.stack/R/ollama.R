#' Ensure a local Ollama server is running
#'
#' Checks whether an Ollama server is reachable at `host:port`. If not,
#' attempts to start it using the platform-appropriate command and
#' polls until the server responds (or `timeout` is reached). Call
#' this once at the top of an R session before classifying with
#' `model_source = "ollama"`.
#'
#' Platform start commands:
#' * **macOS** — `open -a Ollama` (launches the Ollama.app daemon).
#'   Falls back to `ollama serve` if the app is not installed.
#' * **Linux** — `ollama serve` (run in a detached process).
#' * **Windows** — `ollama serve`.
#'
#' If Ollama is not installed, the function returns a clear error
#' message linking to <https://ollama.com>.
#'
#' @param auto_start Logical. If `TRUE` (default), attempt to launch
#'   Ollama when not running. If `FALSE`, just check and error if not
#'   running.
#' @param timeout Numeric. Seconds to wait for Ollama to become ready
#'   after `auto_start`. Default `30`.
#' @param host Character. Hostname Ollama is reachable on.
#'   Default `"localhost"`.
#' @param port Integer. Port Ollama is reachable on. Default `11434L`.
#' @param verbose Logical. Print status messages. Default `TRUE`.
#'
#' @return Invisibly returns `TRUE` when Ollama is running.
#'
#' @examples
#' \dontrun{
#' # Ensure Ollama is up before classifying with a local model
#' ensure_ollama_running()
#'
#' results <- classify(
#'   input_data   = c("text 1", "text 2"),
#'   categories   = c("Positive", "Negative", "Neutral"),
#'   user_model   = "qwen2.5:7b",
#'   model_source = "ollama"
#' )
#'
#' # Just check without auto-starting
#' ensure_ollama_running(auto_start = FALSE)
#' }
#' @export
ensure_ollama_running <- function(auto_start = TRUE,
                                  timeout    = 30,
                                  host       = "localhost",
                                  port       = 11434L,
                                  verbose    = TRUE) {
  providers <- tryCatch(
    reticulate::import("cat_stack._providers"),
    error = function(e) stop(
      "Could not load cat_stack Python module. ",
      "Run cat.stack::install_cat_stack() first. (",
      conditionMessage(e), ")", call. = FALSE
    )
  )

  is_running <- function() {
    isTRUE(providers$check_ollama_running(host = host, port = .as_py_int(port)))
  }

  if (is_running()) {
    if (verbose) message("Ollama is already running at ", host, ":", port, ".")
    return(invisible(TRUE))
  }

  if (!auto_start) {
    stop(
      "Ollama is not running at ", host, ":", port, ".\n",
      "  Start it (e.g. 'ollama serve' in a terminal, or open the Ollama app)\n",
      "  and try again. Install from https://ollama.com if needed.",
      call. = FALSE
    )
  }

  os <- Sys.info()[["sysname"]]
  started <- FALSE

  if (os == "Darwin") {
    # Prefer the macOS Ollama.app (which manages its own daemon).
    if (verbose) message("Starting Ollama (macOS app)...")
    rc <- suppressWarnings(system2(
      "open", c("-a", "Ollama"),
      stdout = FALSE, stderr = FALSE, wait = TRUE
    ))
    started <- (rc == 0L)
    if (!started && nzchar(Sys.which("ollama"))) {
      if (verbose) message("Ollama.app not found; falling back to 'ollama serve'.")
      system2("ollama", "serve", wait = FALSE, stdout = FALSE, stderr = FALSE)
      started <- TRUE
    }
  } else if (os == "Linux" || os == "Windows") {
    if (!nzchar(Sys.which("ollama"))) {
      stop(
        "ollama CLI not found in PATH. ",
        "Install from https://ollama.com and try again.", call. = FALSE
      )
    }
    if (verbose) message("Starting Ollama via 'ollama serve'...")
    system2("ollama", "serve", wait = FALSE, stdout = FALSE, stderr = FALSE)
    started <- TRUE
  } else {
    stop("Unsupported OS for auto-starting Ollama: ", os,
         ". Start it manually and try again.", call. = FALSE)
  }

  if (!started) {
    stop(
      "Could not start Ollama automatically. ",
      "Start it manually (e.g. 'ollama serve') and try again. ",
      "Install from https://ollama.com if needed.", call. = FALSE
    )
  }

  # Poll until ready
  if (verbose) message("Waiting for Ollama to become ready (up to ", timeout, "s)...")
  deadline <- Sys.time() + timeout
  while (Sys.time() < deadline) {
    if (is_running()) {
      if (verbose) message("Ollama is now running.")
      return(invisible(TRUE))
    }
    Sys.sleep(0.5)
  }

  stop(
    "Ollama did not become ready within ", timeout, " seconds. ",
    "Check that it started correctly (e.g. 'ollama serve' in a terminal) ",
    "and try again.", call. = FALSE
  )
}

#' List locally installed Ollama models
#'
#' Returns the names of all models already downloaded to your local Ollama
#' installation. Requires Ollama to be running (call `ensure_ollama_running()`
#' first, or start it manually with `ollama serve`).
#'
#' @param host Character. Hostname Ollama is reachable on. Default `"localhost"`.
#' @param port Integer. Port Ollama is reachable on. Default `11434L`.
#'
#' @return A character vector of model names (e.g. `c("qwen2.5:7b", "mistral:7b")`),
#'   or an empty character vector if Ollama is not running.
#'
#' @examples
#' \dontrun{
#' ensure_ollama_running()
#' list_ollama_models()
#' }
#' @export
list_ollama_models <- function(host = "localhost", port = 11434L) {
  providers <- tryCatch(
    reticulate::import("cat_stack._providers"),
    error = function(e) stop(
      "Could not load cat_stack Python module. ",
      "Run cat.stack::install_cat_stack() first. (",
      conditionMessage(e), ")", call. = FALSE
    )
  )
  as.character(reticulate::py_to_r(
    providers$list_ollama_models(host = host, port = .as_py_int(port))
  ))
}

#' Check whether a specific Ollama model is installed locally
#'
#' Returns `TRUE` if the named model is available in your local Ollama
#' installation, `FALSE` otherwise. Partial name matching is supported
#' (e.g. `"llama3.2"` matches `"llama3.2:latest"`).
#'
#' @param model Character. Model name to look for (e.g. `"qwen2.5:7b"`).
#' @param host Character. Hostname Ollama is reachable on. Default `"localhost"`.
#' @param port Integer. Port Ollama is reachable on. Default `11434L`.
#'
#' @return Logical scalar.
#'
#' @examples
#' \dontrun{
#' check_ollama_model("qwen2.5:7b")
#' }
#' @export
check_ollama_model <- function(model, host = "localhost", port = 11434L) {
  providers <- tryCatch(
    reticulate::import("cat_stack._providers"),
    error = function(e) stop(
      "Could not load cat_stack Python module. ",
      "Run cat.stack::install_cat_stack() first. (",
      conditionMessage(e), ")", call. = FALSE
    )
  )
  isTRUE(reticulate::py_to_r(
    providers$check_ollama_model(model = model, host = host, port = .as_py_int(port))
  ))
}

#' Pull (download) an Ollama model
#'
#' Downloads the named model into your local Ollama installation. Prints the
#' estimated model size and a resource check before downloading. Set
#' `auto_confirm = TRUE` to skip the interactive confirmation prompt — useful
#' in scripts and RMarkdown documents.
#'
#' @param model Character. Model name to download (e.g. `"llama3.2"`,
#'   `"qwen2.5:7b"`).
#' @param host Character. Hostname Ollama is reachable on. Default `"localhost"`.
#' @param port Integer. Port Ollama is reachable on. Default `11434L`.
#' @param auto_confirm Logical. Skip the confirmation prompt. Default `FALSE`.
#'
#' @return Invisibly returns `TRUE` on success, `FALSE` on failure.
#'
#' @examples
#' \dontrun{
#' pull_ollama_model("llama3.2", auto_confirm = TRUE)
#' }
#' @export
pull_ollama_model <- function(model,
                               host         = "localhost",
                               port         = 11434L,
                               auto_confirm = FALSE) {
  providers <- tryCatch(
    reticulate::import("cat_stack._providers"),
    error = function(e) stop(
      "Could not load cat_stack Python module. ",
      "Run cat.stack::install_cat_stack() first. (",
      conditionMessage(e), ")", call. = FALSE
    )
  )
  result <- reticulate::py_to_r(
    providers$pull_ollama_model(
      model        = model,
      host         = host,
      port         = .as_py_int(port),
      auto_confirm = auto_confirm
    )
  )
  invisible(isTRUE(result))
}
