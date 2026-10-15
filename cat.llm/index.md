# cat.llm

**cat.llm** is the meta-package for the CatLLM R ecosystem — the
tidyverse of LLM-powered text classification for social science.
Installing `cat.llm` brings in the full family of domain-specific
packages.

## Installation

``` r

# Install the meta-package from R-universe (brings everything)
install.packages("cat.llm",
                 repos = c("https://chrissoria.r-universe.dev",
                          "https://cloud.r-project.org"))

# Or install individual packages
install.packages(c("cat.stack", "cat.survey", "cat.vader",
                   "cat.ademic", "cat.cog", "cat.pol", "cat.web"),
                 repos = c("https://chrissoria.r-universe.dev",
                          "https://cloud.r-project.org"))
```

Then install the Python backend (one-time):

``` r

library(cat.llm)
install_cat_stack()
```

## The Ecosystem

| Package | Domain | Key Functions |
|----|----|----|
| **cat.stack** | General-purpose | [`classify()`](https://christophersoria.com/cat-llm/cat.stack/reference/classify.html), [`extract()`](https://christophersoria.com/cat-llm/cat.stack/reference/extract.html), [`explore()`](https://christophersoria.com/cat-llm/cat.stack/reference/explore.html), [`summarize()`](https://christophersoria.com/cat-llm/cat.stack/reference/summarize.html) |
| **cat.survey** | Survey responses | [`classify()`](https://christophersoria.com/cat-llm/cat.stack/reference/classify.html), [`extract()`](https://christophersoria.com/cat-llm/cat.stack/reference/extract.html), [`explore()`](https://christophersoria.com/cat-llm/cat.stack/reference/explore.html) with `survey_question` |
| **cat.vader** | Social media | [`classify()`](https://christophersoria.com/cat-llm/cat.stack/reference/classify.html), [`extract()`](https://christophersoria.com/cat-llm/cat.stack/reference/extract.html), [`explore()`](https://christophersoria.com/cat-llm/cat.stack/reference/explore.html) with platform metadata |
| **cat.ademic** | Academic papers | [`classify()`](https://christophersoria.com/cat-llm/cat.stack/reference/classify.html), [`extract()`](https://christophersoria.com/cat-llm/cat.stack/reference/extract.html), [`explore()`](https://christophersoria.com/cat-llm/cat.stack/reference/explore.html), [`summarize()`](https://christophersoria.com/cat-llm/cat.stack/reference/summarize.html) with OpenAlex |
| **cat.cog** | Cognitive assessment | [`cerad_drawn_score()`](https://christophersoria.com/cat-llm/cat.llm/reference/catllm-aliases.md) for CERAD drawings |
| **cat.pol** | Policy documents | [`classify()`](https://christophersoria.com/cat-llm/cat.stack/reference/classify.html), [`extract()`](https://christophersoria.com/cat-llm/cat.stack/reference/extract.html), [`explore()`](https://christophersoria.com/cat-llm/cat.stack/reference/explore.html), [`summarize()`](https://christophersoria.com/cat-llm/cat.stack/reference/summarize.html), [`list_sources()`](https://christophersoria.com/cat-llm/cat.pol/reference/list_sources.html) for ordinances, federal laws, executive orders, political speech |
| **cat.web** | Web content | [`classify()`](https://christophersoria.com/cat-llm/cat.stack/reference/classify.html), [`extract()`](https://christophersoria.com/cat-llm/cat.stack/reference/extract.html), [`explore()`](https://christophersoria.com/cat-llm/cat.stack/reference/explore.html), [`summarize()`](https://christophersoria.com/cat-llm/cat.stack/reference/summarize.html) with automatic URL fetching |
| **cat.llm** | Meta-package | Re-exports everything with domain-suffixed aliases |

## Usage

``` r

library(cat.llm)
# -- Attaching cat.llm ecosystem --
# v 0.1.0 cat.stack
# v 0.1.0 cat.survey
# v 0.1.0 cat.vader
# v 0.1.0 cat.ademic
# v 0.1.0 cat.cog
# v 0.1.0 cat.pol
# v 0.1.0 cat.web

# Base classification (from cat.stack)
results <- classify(
  input_data = c("Great product!", "Terrible experience."),
  categories = c("Positive", "Negative"),
  api_key    = Sys.getenv("OPENAI_API_KEY")
)

# Survey classification (domain-suffixed alias)
results <- classify_survey(
  input_data      = df$responses,
  categories      = c("Satisfied", "Dissatisfied", "Neutral"),
  survey_question = "How satisfied are you with our service?",
  api_key         = Sys.getenv("OPENAI_API_KEY")
)

# Social media classification
results <- classify_social(
  sm_source = "reddit",
  sm_handle = "technology",
  sm_limit  = 100,
  categories = c("Opinion", "News", "Question"),
  api_key    = Sys.getenv("OPENAI_API_KEY")
)

# Or use domain packages directly
library(cat.survey)
cat.survey::classify(...)
```

## License

MIT
