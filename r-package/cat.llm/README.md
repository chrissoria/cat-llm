# cat.llm

**cat.llm** is the meta-package for the CatLLM R ecosystem — the tidyverse of LLM-powered text classification for social science. Installing `cat.llm` brings in the full family of domain-specific packages.

## Installation

```r
# Install the meta-package (brings everything)
install.packages("cat.llm")

# Or install individual packages
install.packages("cat.stack")    # General-purpose engine
install.packages("cat.survey")   # Survey responses
install.packages("cat.vader")    # Social media
install.packages("cat.ademic")   # Academic papers
install.packages("cat.cog")      # Cognitive assessment (CERAD)
```

Then install the Python backend:

```r
library(cat.llm)
install_cat_stack()
```

## The Ecosystem

| Package | Domain | Key Functions |
|---------|--------|---------------|
| **cat.stack** | General-purpose | `classify()`, `extract()`, `explore()`, `summarize()` |
| **cat.survey** | Survey responses | `classify()`, `extract()`, `explore()` with `survey_question` |
| **cat.vader** | Social media | `classify()`, `extract()`, `explore()` with platform metadata |
| **cat.ademic** | Academic papers | `classify()`, `extract()`, `explore()`, `summarize()` with OpenAlex |
| **cat.cog** | Cognitive assessment | `cerad_drawn_score()` for CERAD drawings |
| **cat.llm** | Meta-package | Re-exports everything with domain-suffixed aliases |

## Usage

```r
library(cat.llm)
# -- Attaching cat.llm ecosystem --
# v 0.1.0 cat.stack
# v 0.1.0 cat.survey
# v 0.1.0 cat.vader
# v 0.1.0 cat.ademic
# v 0.1.0 cat.cog

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
