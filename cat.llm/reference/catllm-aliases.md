# Domain-suffixed aliases for the CatLLM ecosystem

These functions provide convenient domain-suffixed names so users can
tab-complete to find the right function. Each is a thin re-export from
the corresponding domain package.

## Usage

``` r
classify_survey(...)

extract_survey(...)

explore_survey(...)

classify_social(...)

extract_social(...)

explore_social(...)

classify_academic(...)

extract_academic(...)

explore_academic(...)

summarize_academic(...)

cerad_drawn_score(...)

classify_political(...)

extract_political(...)

explore_political(...)

summarize_political(...)

classify_web(...)

extract_web(...)

explore_web(...)

summarize_web(...)
```

## Arguments

- ...:

  Additional arguments passed to the Python function.

## Examples

``` r
if (FALSE) { # \dontrun{
library(cat.llm)

# Survey classification (re-export of cat.survey::classify)
classify_survey(
  input_data      = df$responses,
  categories      = c("Cost", "Quality", "Service", "Other"),
  survey_question = "Why did you choose us?",
  api_key         = Sys.getenv("OPENAI_API_KEY")
)

# Political documents (re-export of cat.pol::classify)
classify_political(
  source     = "city_san_diego",
  doc_type   = "ordinance",
  n          = 50L,
  categories = c("Housing", "Public Safety", "Finance"),
  api_key    = Sys.getenv("OPENAI_API_KEY")
)

# Web content (re-export of cat.web::classify)
classify_web(
  input_data    = c("https://example.com/article-1",
                    "https://example.com/article-2"),
  categories    = c("News", "Opinion", "Tutorial"),
  source_domain = "example.com",
  api_key       = Sys.getenv("OPENAI_API_KEY")
)

# CERAD cognitive scoring (re-export of cat.cog::cerad_drawn_score)
cerad_drawn_score(
  shape       = "circle",
  image_input = "./drawings/",
  api_key     = Sys.getenv("OPENAI_API_KEY")
)
} # }
```
