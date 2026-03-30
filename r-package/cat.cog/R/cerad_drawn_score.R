#' Score CERAD constructional praxis drawings using LLMs
#'
#' Wraps the Python `cat_cog.cerad_drawn_score()` function. Scores drawn shapes
#' (circle, diamond, rectangles, cube) from the CERAD constructional praxis
#' assessment using vision-capable LLMs.
#'
#' @param shape Character. The shape being scored: `"circle"`, `"diamond"`,
#'   `"rectangles"`, or `"cube"`.
#' @param image_input Character. Path to the image file or directory of images.
#' @param api_key Character. API key for the model provider.
#' @param user_model Character. Model name. Default `"gpt-4o"`.
#' @param creativity Numeric or `NULL`. Temperature setting. Default `NULL`.
#' @param safety Logical. Save progress after each item. Default `FALSE`.
#' @param chain_of_thought Logical. Enable chain-of-thought reasoning. Default
#'   `TRUE`.
#' @param filename Character or `NULL`. Output CSV filename. Default `NULL`.
#' @param save_directory Character or `NULL`. Directory to save results. Default
#'   `NULL`.
#' @param model_source Character. Provider hint: `"auto"`, `"openai"`,
#'   `"anthropic"`, `"google"`, etc. Default `"auto"`.
#' @param ... Additional arguments passed to the Python function.
#'
#' @return A `data.frame` with scoring results.
#'
#' @examples
#' \dontrun{
#' # Score a single circle drawing
#' result <- cerad_drawn_score(
#'   shape       = "circle",
#'   image_input = "path/to/circle_drawing.png",
#'   api_key     = Sys.getenv("OPENAI_API_KEY")
#' )
#'
#' # Score a directory of cube drawings
#' results <- cerad_drawn_score(
#'   shape       = "cube",
#'   image_input = "path/to/cube_drawings/",
#'   api_key     = Sys.getenv("OPENAI_API_KEY"),
#'   user_model  = "claude-sonnet-4-5-20250929",
#'   model_source = "anthropic"
#' )
#' }
#'
#' @export
cerad_drawn_score <- function(
    shape,
    image_input,
    api_key,
    user_model       = "gpt-4o",
    creativity       = NULL,
    safety           = FALSE,
    chain_of_thought = TRUE,
    filename         = NULL,
    save_directory   = NULL,
    model_source     = "auto",
    ...
) {
  mod <- .get_cat_cog()

  api_key <- cat.stack:::.strip_quotes(api_key)
  if (!is.null(creativity)) creativity <- as.double(creativity)

  result <- mod$cerad_drawn_score(
    shape            = shape,
    image_input      = image_input,
    api_key          = api_key,
    user_model       = user_model,
    creativity       = reticulate::r_to_py(creativity),
    safety           = safety,
    chain_of_thought = chain_of_thought,
    filename         = filename,
    save_directory   = save_directory,
    model_source     = model_source,
    ...
  )

  reticulate::py_to_r(result)
}
