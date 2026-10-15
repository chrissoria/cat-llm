# Classifying Social Media Posts

## What `cat.vader` adds

`cat.vader` is a thin domain wrapper around `cat.stack` that adds
**social-media platform connectors** plus prompt framing tuned for
short, informal, online language. You can use it two ways:

1.  **Pull posts directly from a platform** (Threads, Reddit, Bluesky,
    Mastodon, YouTube, etc.) using the `sm_source`, `sm_handle`,
    `sm_credentials` arguments, then classify.
2.  **Classify text you already have** as a plain character vector —
    identical to
    [`cat.stack::classify()`](https://christophersoria.com/cat-llm/cat.stack/reference/classify.html)
    but with social-media-aware prompt context.

Everything else — supported models, output format, ensemble voting — is
identical to `cat.stack`.

## Install

``` r

install.packages(
  "cat.vader",
  repos = c("https://chrissoria.r-universe.dev",
            "https://cloud.r-project.org")
)
library(cat.vader)
```

## Classify text you already have

The simplest path — pass a character vector of post text:

``` r

posts <- c(
  "Just had the best coffee ever! Highly recommend the new place downtown.",
  "Politicians are all the same. Nothing ever changes.",
  "Looking forward to the game tonight!",
  "This new policy is going to ruin small businesses.",
  "Anyone know a good vet in the area?"
)

results <- classify(
  input_data = posts,
  categories = c("Positive sentiment", "Negative sentiment",
                 "Question/request", "Other"),
  api_key    = Sys.getenv("OPENAI_API_KEY"),
  user_model = "gpt-4o-mini"
)
```

## Pull and classify in one call

If you want to analyse your own posting history or a public account,
`cat.vader` can pull posts directly and feed them through the
classifier:

``` r

# Authenticate once with your Threads API credentials, then:
results <- classify(
  sm_source      = "threads",
  sm_handle      = "your_username",
  sm_months      = 6L,                      # last 6 months
  sm_credentials = Sys.getenv("THREADS_TOKEN"),
  categories     = c("Personal", "Political", "Promotional",
                     "Question", "Other"),
  api_key        = Sys.getenv("OPENAI_API_KEY"),
  user_model     = "gpt-4o-mini"
)
```

The returned `data.frame` includes the original text **and** platform
engagement metrics (likes, replies, reposts, etc.) so you can correlate
content categories with reach.

Each platform has its own authentication setup; check the [cat-vader
Python docs](https://pypi.org/project/cat-vader/) for the current
credential formats.

## Discovering categories from a feed

Before classifying, see what themes are actually present:

``` r

cats <- extract(
  input_data     = df$posts,
  max_categories = 10L,
  api_key        = Sys.getenv("OPENAI_API_KEY"),
  user_model     = "gpt-4o-mini"
)
cats$top_categories
```

Then iterate — drop noise categories, merge similar ones, and re-run
[`classify()`](https://christophersoria.com/cat-llm/cat.vader/reference/classify.md)
with the cleaned scheme.

## Tips for social-media data

1.  **Short text is harder.** Tweets/posts often lack context (`"lol"`,
    `"this"`). Including the platform name and a brief description in
    `description=` helps the model interpret short posts.
2.  **Emoji and slang.** Frontier models handle these well; older or
    smaller models less so. If you’re seeing weird classifications on
    highly informal text, try `gpt-4o` or `claude-3-5-sonnet` over the
    mini/haiku tier.
3.  **Engagement isn’t ground truth.** A high-engagement post isn’t
    necessarily *correctly* classified — engagement reflects audience,
    not category. Validate on a hand-coded subsample.

## Where to learn more

- Full Getting Started guide:
  [`vignette("getting-started", package = "cat.llm")`](https://christophersoria.com/cat-llm/cat.llm/articles/getting-started.html)
- Per-function reference:
  [`?cat.vader::classify`](https://christophersoria.com/cat-llm/cat.vader/reference/classify.md),
  [`?cat.vader::extract`](https://christophersoria.com/cat-llm/cat.vader/reference/extract.md),
  [`?cat.vader::explore`](https://christophersoria.com/cat-llm/cat.vader/reference/explore.md)
- Platform-connector docs are maintained in the [Python catvader
  package](https://pypi.org/project/cat-vader/).
