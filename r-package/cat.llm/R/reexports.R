#' Re-exported functions from cat.stack
#'
#' The `cat.llm` meta-package re-exports the core functions from `cat.stack`
#' so they are available immediately after `library(cat.llm)`. Behavior and
#' arguments are identical to the originals --- the links below point at
#' the canonical documentation in `cat.stack`.
#'
#' \describe{
#'   \item{\code{\link[cat.stack]{classify}}}{Classify text, images, or PDFs into predefined categories.}
#'   \item{\code{\link[cat.stack]{extract}}}{Discover and extract categories from data when no scheme is predefined.}
#'   \item{\code{\link[cat.stack]{explore}}}{Raw category extraction for saturation analysis.}
#'   \item{\code{\link[cat.stack]{summarize}}}{Summarize text, images, or PDFs in several formats.}
#'   \item{\code{\link[cat.stack]{prompt_tune}}}{Interactive, human-in-the-loop tuning of the classification system prompt.}
#'   \item{\code{\link[cat.stack]{install_cat_stack}}}{One-time installer for the Python `catstack` backend.}
#'   \item{\code{\link[cat.stack]{ensure_ollama_running}}}{Check (and optionally start) a local Ollama server before running local-model classification.}
#'   \item{\code{\link[cat.stack]{list_ollama_models}}}{List locally installed Ollama models.}
#'   \item{\code{\link[cat.stack]{check_ollama_model}}}{Check whether a specific Ollama model is installed locally.}
#'   \item{\code{\link[cat.stack]{pull_ollama_model}}}{Download an Ollama model to the local installation.}
#' }
#'
#' @name reexports
#' @aliases classify extract explore summarize prompt_tune install_cat_stack ensure_ollama_running list_ollama_models check_ollama_model pull_ollama_model
#' @keywords internal
NULL
