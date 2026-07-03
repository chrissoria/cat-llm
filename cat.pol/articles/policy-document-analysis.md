# Classifying Policy Documents and Political Text

## What `cat.pol` adds

`cat.pol` is a thin domain wrapper around `cat.stack` that adds:

1.  **A registered-source fetcher** — 17 built-in political data sources
    (municipal ordinances, federal laws, executive orders, presidential
    speeches, Truth Social posts, etc.) accessible via a single
    `source=` argument. The data lives on HuggingFace and is refreshed
    weekly.
2.  **Policy-document prompt framing** — context like
    `"This is a policy document; identify what it does and who it affects"`
    injected automatically.

Everything else — supported models, output format, ensemble voting — is
identical to `cat.stack`.

## Install

``` r

install.packages(
  "cat.pol",
  repos = c("https://chrissoria.r-universe.dev",
            "https://cloud.r-project.org")
)
library(cat.pol)
```

## See available data sources

``` r

list_sources()
#> [1] "city_san_diego"            "city_san_francisco"
#> [3] "city_los_angeles"          "federal_laws"
#> [5] "federal_executive_orders"  "social_trump_truth"
#> ...
```

Each source maps to a curated HuggingFace dataset with weekly updates.
See the [Python catpol README](https://pypi.org/project/cat-pol/) for
the current full list and the schema of each source.

## Fetch and classify ordinances

``` r

results <- classify(
  source     = "city_san_diego",
  doc_type   = "ordinance",
  since      = "2024-01-01",
  n          = 50L,
  categories = c("Housing", "Public Safety", "Finance",
                 "Infrastructure", "Health", "Other"),
  api_key    = Sys.getenv("OPENAI_API_KEY"),
  user_model = "gpt-4o-mini"
)
```

The returned `data.frame` has one row per ordinance with the original
text, the date, the URL/ID, and one 0/1 column per category.

## Filter by date range and document type

``` r

# Resolutions only, between two dates:
results <- classify(
  source     = "city_san_francisco",
  doc_type   = "resolution",
  since      = "2024-06-01",
  until      = "2024-12-31",
  n          = 200L,
  categories = c("Climate", "Housing", "Transportation",
                 "Police accountability", "Other"),
  api_key    = Sys.getenv("OPENAI_API_KEY")
)
```

## Classify your own policy text

If you have policy documents not in the registered sources (state
legislation, agency rules, advocacy white papers), pass them as
`input_data`:

``` r

results <- classify(
  input_data       = df$bill_text,
  document_context = "California state Senate bills, 2024 session",
  categories       = c("Housing", "Public Safety", "Education",
                       "Healthcare", "Environment", "Other"),
  api_key          = Sys.getenv("OPENAI_API_KEY"),
  user_model       = "gpt-4o-mini"
)
```

`document_context` is `cat.pol`’s analog of `cat.survey`’s
`survey_question` — it gives the model framing for the documents being
analyzed.

## Summarize before classifying

Long ordinances (10–20k words) can blow past context limits and cost a
lot in tokens. Summarize first, then classify the summaries:

``` r

summaries <- summarize(
  source     = "city_san_diego",
  doc_type   = "ordinance",
  since      = "2024-01-01",
  n          = 50L,
  format     = "paragraph",
  tone       = "eli5",                       # plain-language summary
  api_key    = Sys.getenv("OPENAI_API_KEY"),
  user_model = "gpt-4o-mini"
)

results <- classify(
  input_data = summaries$summary,
  categories = c("Housing", "Public Safety", "Finance", "Other"),
  api_key    = Sys.getenv("OPENAI_API_KEY")
)
```

The `tone` parameter is specific to
[`cat.pol::summarize()`](https://christophersoria.com/cat-llm/cat.pol/reference/summarize.md);
options include `"eli5"` (plain language), `"neutral"` (technical), and
`"academic"` (formal). Useful for downstream readability or for
generating press-friendly summaries alongside the analytic
classification.

## Tips for political-text work

1.  **Be specific about category boundaries.** Policy domains overlap —
    a housing ordinance might also touch finance and zoning. Either use
    multi-label (default) or write categories with explicit exclusion
    criteria.
2.  **Watch for ideological priors.** LLMs have political biases. For
    research where the political lean of the classifier matters, use a
    multi-model ensemble (see the meta-package vignette,
    [`vignette("getting-started", "cat.llm")`](https://christophersoria.com/cat-llm/cat.llm/articles/getting-started.html)).
3.  **Cite the data source.** Each `source=` value corresponds to a
    public HuggingFace dataset that has its own preferred citation.

## Where to learn more

- Full Getting Started guide:
  [`vignette("getting-started", package = "cat.llm")`](https://christophersoria.com/cat-llm/cat.llm/articles/getting-started.html)
- Per-function reference:
  [`?cat.pol::classify`](https://christophersoria.com/cat-llm/cat.pol/reference/classify.md),
  [`?cat.pol::extract`](https://christophersoria.com/cat-llm/cat.pol/reference/extract.md),
  [`?cat.pol::explore`](https://christophersoria.com/cat-llm/cat.pol/reference/explore.md),
  [`?cat.pol::summarize`](https://christophersoria.com/cat-llm/cat.pol/reference/summarize.md),
  [`?cat.pol::list_sources`](https://christophersoria.com/cat-llm/cat.pol/reference/list_sources.md)
- Built-in source schemas:
  <https://github.com/chrissoria/cat-pol#built-in-sources>
