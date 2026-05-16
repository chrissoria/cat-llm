#' List registered political data sources
#'
#' Returns the names of all data sources registered with the Python `catpol`
#' package (city ordinances, federal laws, executive orders, presidential
#' speeches, social media archives, etc.).
#'
#' @return A character vector of source names.
#' @export
list_sources <- function() {
  mod <- .get_catpol()
  reticulate::py_to_r(mod$list_sources())
}
