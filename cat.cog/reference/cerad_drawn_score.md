# Score CERAD constructional praxis drawings using LLMs

Wraps the Python `cat_cog.cerad_drawn_score()` function. Scores drawn
shapes (circle, diamond, rectangles, cube) from the CERAD constructional
praxis assessment using vision-capable LLMs.

## Usage

``` r
cerad_drawn_score(
  shape,
  image_input,
  api_key,
  user_model = "gpt-4o",
  creativity = NULL,
  safety = FALSE,
  chain_of_thought = TRUE,
  filename = NULL,
  save_directory = NULL,
  model_source = "auto",
  ...
)
```

## Arguments

- shape:

  Character. The shape being scored: `"circle"`, `"diamond"`,
  `"rectangles"`, or `"cube"`.

- image_input:

  Character. Path to the image file or directory of images.

- api_key:

  Character. API key for the model provider.

- user_model:

  Character. Model name. Default `"gpt-4o"`.

- creativity:

  Numeric or `NULL`. Temperature setting. Default `NULL`.

- safety:

  Logical. Save progress after each item. Default `FALSE`.

- chain_of_thought:

  Logical. Enable chain-of-thought reasoning. Default `TRUE`.

- filename:

  Character or `NULL`. Output CSV filename. Default `NULL`.

- save_directory:

  Character or `NULL`. Directory to save results. Default `NULL`.

- model_source:

  Character. Provider hint: `"auto"`, `"openai"`, `"anthropic"`,
  `"google"`, etc. Default `"auto"`.

- ...:

  Additional arguments passed to the Python function.

## Value

A `data.frame` with scoring results.

## Examples

``` r
if (FALSE) { # \dontrun{
# Score a single circle drawing
result <- cerad_drawn_score(
  shape       = "circle",
  image_input = "path/to/circle_drawing.png",
  api_key     = Sys.getenv("OPENAI_API_KEY")
)

# Score a directory of cube drawings
results <- cerad_drawn_score(
  shape       = "cube",
  image_input = "path/to/cube_drawings/",
  api_key     = Sys.getenv("OPENAI_API_KEY"),
  user_model  = "claude-sonnet-4-5-20250929",
  model_source = "anthropic"
)
} # }
```
