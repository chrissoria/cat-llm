#' List registered political data sources
#'
#' Returns the names of all data sources registered with the Python `catpol`
#' package (city ordinances, federal laws, executive orders, presidential
#' speeches, social media archives, etc.).
#'
#' @return A character vector of source names.
#' @examples
#' \dontrun{
#' list_sources()
#' #> [1] "city_san_diego"           "city_san_francisco"
#' #> [3] "federal_laws"             "federal_executive_orders"
#' #> [5] "social_trump_truth"       ...
#' }
#' @export
list_sources <- function() {
  mod <- .get_catpol()
  reticulate::py_to_r(mod$list_sources())
}
