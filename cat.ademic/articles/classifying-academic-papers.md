# Classifying Academic Papers

## What `cat.ademic` adds

`cat.ademic` is a thin domain wrapper around `cat.stack` that adds
**OpenAlex-based paper fetching** plus academic prompt framing. You can:

1.  **Fetch papers from a journal or topic** via OpenAlex
    (`journal_name`, `journal_issn`, `journal_field`, `topic_name`,
    `topic_id`) and classify them in one call.
2.  **Classify text you already have** (abstracts, full text, or PDFs)
    as a plain character vector or file directory.

Everything else — supported models, output format, ensemble voting,
batch mode — is identical to `cat.stack`.

## Install

``` r

install.packages(
  "cat.ademic",
  repos = c("https://chrissoria.r-universe.dev",
            "https://cloud.r-project.org")
)
library(cat.ademic)
```

## Classify abstracts you already have

``` r

abstracts <- c(
  "We use mixed-methods to study labor market outcomes for...",
  "This paper develops a formal model of bargaining under...",
  "A systematic review of 47 studies on educational interventions..."
)

results <- classify(
  categories = c("Empirical-quantitative",
                 "Empirical-qualitative",
                 "Theoretical-formal",
                 "Review/meta-analysis",
                 "Other"),
  input_data = abstracts,
  mode       = "text",
  api_key    = Sys.getenv("OPENAI_API_KEY"),
  user_model = "gpt-4o-mini"
)
```

## Fetch papers from a journal

`cat.ademic` connects to [OpenAlex](https://openalex.org) — a free, open
scholarly database — to fetch papers by journal, field, or topic. Set
`polite_email` (your email) for higher rate limits.

``` r

results <- classify(
  categories   = c("Quantitative", "Qualitative", "Mixed Methods"),
  journal_name = "American Sociological Review",
  paper_limit  = 100L,
  date_from    = "2024-01-01",
  polite_email = "you@university.edu",
  api_key      = Sys.getenv("OPENAI_API_KEY")
)
```

Or by ISSN for unambiguous journal identification:

``` r

results <- classify(
  categories   = c("Empirical", "Theoretical", "Review"),
  journal_issn = "0003-1224",                # AJS
  paper_limit  = 50L,
  polite_email = "you@university.edu",
  api_key      = Sys.getenv("OPENAI_API_KEY")
)
```

## Fetch papers by topic

OpenAlex auto-tags papers with research topics. You can pull all papers
on a topic across journals:

``` r

results <- classify(
  categories   = c("Causal-identification", "Descriptive",
                   "Theoretical", "Other"),
  topic_name   = "climate change adaptation",
  paper_limit  = 200L,
  date_from    = "2023-01-01",
  polite_email = "you@university.edu",
  api_key      = Sys.getenv("OPENAI_API_KEY")
)
```

## Classify full-text PDFs

Pass a directory or a vector of file paths. `cat.ademic` extracts the
text (or renders pages as images for vision models) and classifies:

``` r

# One-time: install PDF extras
# cat.stack::install_cat_stack(pdf = TRUE)

results <- classify(
  categories = c("Has-DGP-assumption", "No-DGP-assumption",
                 "Unclear", "Other"),
  input_data = "./papers/",                  # directory of PDFs
  mode       = "image",                      # rendered-page vision mode
  api_key    = Sys.getenv("OPENAI_API_KEY"),
  user_model = "gpt-4o"                      # vision-capable model
)
```

## Summarize before classifying

For long full-text inputs, summarizing first can improve downstream
classification quality (and reduce token cost):

``` r

summaries <- summarize(
  input_data   = "./papers/",
  description  = "Sociology articles",
  instructions = "Summarize methodology and key findings in 3 sentences",
  format       = "paragraph",
  api_key      = Sys.getenv("OPENAI_API_KEY"),
  user_model   = "gpt-4o-mini"
)

results <- classify(
  categories = c("Causal", "Descriptive", "Theoretical", "Other"),
  input_data = summaries$summary,
  api_key    = Sys.getenv("OPENAI_API_KEY")
)
```

## Tips for academic work

1.  **Always set `polite_email`** when fetching from OpenAlex — without
    it you’re throttled to a low rate limit.
2.  **Abstract vs. full text.** Abstracts are cheap and fast; full-text
    classification (PDF input) is more accurate for methodological
    categories but costs more. Use abstracts for screening, full text
    for in-depth coding.
3.  **Cite OpenAlex** if you publish on data fetched through it — see
    [openalex.org](https://openalex.org) for citation guidance.

## Where to learn more

- Full Getting Started guide:
  [`vignette("getting-started", package = "cat.llm")`](https://christophersoria.com/cat-llm/cat.llm/articles/getting-started.html)
- Per-function reference:
  [`?cat.ademic::classify`](https://christophersoria.com/cat-llm/cat.ademic/reference/classify.md),
  [`?cat.ademic::extract`](https://christophersoria.com/cat-llm/cat.ademic/reference/extract.md),
  [`?cat.ademic::explore`](https://christophersoria.com/cat-llm/cat.ademic/reference/explore.md),
  [`?cat.ademic::summarize`](https://christophersoria.com/cat-llm/cat.ademic/reference/summarize.md)
- OpenAlex docs: <https://docs.openalex.org>
