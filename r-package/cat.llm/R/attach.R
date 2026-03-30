#' Attach all CatLLM ecosystem packages
#'
#' Explicitly loads all domain packages. Normally this happens automatically
#' when `library(cat.llm)` is called, but this function can be used to
#' reload after detaching.
#'
#' @return Invisibly returns a character vector of attached package names.
#' @export
catllm_attach <- function() {
  pkgs <- c("cat.stack", "cat.survey", "cat.vader", "cat.ademic", "cat.cog")
  for (pkg in pkgs) {
    if (!paste0("package:", pkg) %in% search()) {
      suppressPackageStartupMessages(
        library(pkg, character.only = TRUE)
      )
    }
  }
  invisible(pkgs)
}
