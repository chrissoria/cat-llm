# Classifying Open-Ended Survey Responses

## What `cat.survey` adds

`cat.survey` is a thin domain wrapper around `cat.stack` that injects
**survey-question context** into every prompt. When you call
`classify(input_data = ..., survey_question = "Why did you move?")`, the
LLM sees:

> “A respondent was asked: *Why did you move?* Their answer was: *…*”

That framing measurably improves accuracy on open-ended survey data
versus generic text classification, because the model uses the question
to disambiguate short or context-dependent responses.

Everything else — supported models, output format, ensemble voting,
batch mode — is identical to `cat.stack`.

## Install

``` r

install.packages(
  "cat.survey",
  repos = c("https://chrissoria.r-universe.dev",
            "https://cloud.r-project.org")
)
library(cat.survey)
```

## Quick classification

``` r

responses <- c(
  "Took a new job in Chicago",
  "Wanted to be closer to grandkids",
  "Couldn't afford rent in the Bay Area",
  "Job market collapsed after the layoffs",
  "Family pressure to move home"
)

# Verbose category descriptions classify better than short labels.
verbose_cats <- c(
  "Job/school: A change in employment, education, or career, including transfers and retirement.",
  "Family: Relationship changes, having children, supporting relatives, or relocating to be near family.",
  "Cost of living: Housing affordability, cost of goods, or general economic pressure.",
  "Other: The response does not fit any of the above categories."
)

results <- classify(
  input_data      = responses,
  categories      = verbose_cats,
  survey_question = "Why did you move to a new city?",
  api_key         = Sys.getenv("OPENAI_API_KEY"),
  user_model      = "gpt-4o-mini"
)
```

## Multi-label survey responses

Many survey responses fit more than one category (“I moved for a new job
and to be closer to family”). The default classifier is multi-label —
`results` will have one 0/1 column per category, and a row can have
multiple 1s.

To force single-label, set `add_other = FALSE` and shrink the category
list. To make multi-label explicit in your analysis, use the binary
columns directly:

``` r

# Example downstream summary:
library(dplyr)
results %>%
  dplyr::summarize(
    pct_job    = mean(`Job/school`),
    pct_family = mean(Family),
    pct_cost   = mean(`Cost of living`),
    pct_other  = mean(Other)
  )
```

## Discovering a category scheme when you don’t have one

If you don’t already have a coding scheme, use
[`extract()`](https://christophersoria.com/cat-llm/cat.survey/reference/extract.md)
to discover one from the responses themselves, then pass the result to
[`classify()`](https://christophersoria.com/cat-llm/cat.survey/reference/classify.md):

``` r

cats <- extract(
  input_data      = responses,
  survey_question = "Why did you move to a new city?",
  max_categories  = 8L,
  api_key         = Sys.getenv("OPENAI_API_KEY")
)
cats$top_categories

# Optionally rewrite the labels to be more verbose, then classify:
results <- classify(
  input_data      = responses,
  categories      = cats$top_categories,
  survey_question = "Why did you move to a new city?",
  api_key         = Sys.getenv("OPENAI_API_KEY")
)
```

See also `extract.Rmd` in the `r-package/examples/` directory for a
deeper walkthrough of category discovery.

## Recommendations for survey work

1.  **Always set `survey_question`** — it’s the whole point of using
    `cat.survey` over `cat.stack`. Without it you might as well use
    [`cat.stack::classify()`](https://christophersoria.com/cat-llm/cat.stack/reference/classify.html)
    directly.
2.  **Write verbose category descriptions.** A label like
    `"Family: relocating to be near family, having a child, divorce..."`
    classifies several percentage points more accurately than just
    `"Family"`. This is the single biggest accuracy lever.
3.  **Include an “Other” category.** Prevents the model from forcing
    ambiguous responses into ill-fitting boxes. `cat.survey` will prompt
    to add one if you forget (`add_other = "prompt"` is the default).
4.  **Validate on a hand-coded subsample.** For published research,
    never trust classifications without spot-checking against human
    coding on at least 50–100 responses.

## Where to learn more

- Full Getting Started guide:
  [`vignette("getting-started", package = "cat.llm")`](https://christophersoria.com/cat-llm/cat.llm/articles/getting-started.html)
- Per-function reference:
  [`?cat.survey::classify`](https://christophersoria.com/cat-llm/cat.survey/reference/classify.md),
  [`?cat.survey::extract`](https://christophersoria.com/cat-llm/cat.survey/reference/extract.md),
  [`?cat.survey::explore`](https://christophersoria.com/cat-llm/cat.survey/reference/explore.md)
- Empirical best-practices research (incl. why verbose labels help) is
  in the project [Python
  README](https://github.com/chrissoria/cat-llm#best-practices-for-classification).
