#' @importFrom cat.stack classify extract explore summarize install_cat_stack
NULL

.onAttach <- function(libname, pkgname) {
  packageStartupMessage(
    "-- Attaching cat.llm ecosystem --\n",
    "v ", utils::packageVersion("cat.stack"),  " cat.stack\n",
    "v ", utils::packageVersion("cat.survey"), " cat.survey\n",
    "v ", utils::packageVersion("cat.vader"),  " cat.vader\n",
    "v ", utils::packageVersion("cat.ademic"), " cat.ademic\n",
    "v ", utils::packageVersion("cat.cog"),    " cat.cog"
  )
}
