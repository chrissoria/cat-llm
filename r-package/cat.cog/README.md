# cat.cog

Cognitive assessment scoring with LLMs. A domain wrapper around
[cat.stack](../cat.stack/) that scores CERAD constructional praxis (drawn
shape) assessments using vision-capable large language models.

`cat.cog` wraps the Python [cat-cog](https://pypi.org/project/cat-cog/)
package via [reticulate](https://rstudio.github.io/reticulate/).

## Installation

```r
devtools::install("path/to/cat.stack")
devtools::install("path/to/cat.cog")

# Install the Python backend
pip install cat-cog
```

## Quick Start

### Score a single drawing

```r
library(cat.cog)

result <- cerad_drawn_score(
  shape       = "circle",
  image_input = "path/to/circle_drawing.png",
  api_key     = Sys.getenv("OPENAI_API_KEY")
)
```

### Score a directory of drawings

```r
results <- cerad_drawn_score(
  shape       = "cube",
  image_input = "path/to/cube_drawings/",
  api_key     = Sys.getenv("OPENAI_API_KEY"),
  user_model  = "claude-sonnet-4-5-20250929",
  model_source = "anthropic"
)
```

## Functions

| Function | Description |
|----------|-------------|
| `cerad_drawn_score()` | Score CERAD constructional praxis drawings |

## Supported Shapes

- `"circle"` - Circle drawing assessment
- `"diamond"` - Diamond drawing assessment
- `"rectangles"` - Overlapping rectangles assessment
- `"cube"` - 3D cube drawing assessment

## License

MIT
