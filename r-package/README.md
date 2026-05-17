# CatLLM R Ecosystem

This directory contains the R interface to the [CatLLM](https://catllm.com)
Python ecosystem — eight R packages that wrap their Python counterparts via
[reticulate](https://rstudio.github.io/reticulate/) so R users can do
LLM-powered text classification with the same API as Python users.

## Install

The whole ecosystem is published to **R-universe**:

```r
# Install the meta-package — pulls everything in
install.packages("cat.llm",
                 repos = c("https://chrissoria.r-universe.dev",
                          "https://cloud.r-project.org"))

# Install the Python backend (one-time setup)
library(cat.llm)
install_cat_stack()
```

Or install a single domain package for a lighter footprint:

```r
install.packages(c("cat.stack", "cat.survey"),
                 repos = c("https://chrissoria.r-universe.dev",
                          "https://cloud.r-project.org"))
```

## The packages

| Package        | Domain                        | Python backend  | Wraps                                                       |
|----------------|-------------------------------|-----------------|-------------------------------------------------------------|
| **cat.stack**  | General-purpose classification base | `catstack`      | `classify`, `extract`, `explore`, `summarize`               |
| **cat.survey** | Open-ended survey responses   | `catsurvey`     | `classify`, `extract`, `explore` (with `survey_question=`)  |
| **cat.vader**  | Social media posts            | `catvader`      | `classify`, `extract`, `explore` (platform connectors)      |
| **cat.ademic** | Academic papers (OpenAlex)    | `catademic`     | `classify`, `extract`, `explore`, `summarize`               |
| **cat.cog**    | Cognitive assessment scoring  | `catcog`        | `cerad_drawn_score`                                          |
| **cat.pol**    | Policy documents              | `catpol`        | `classify`, `extract`, `explore`, `summarize`, `list_sources` |
| **cat.web**    | Web content (URL fetching)    | `catweb`        | `classify`, `extract`, `explore`, `summarize`               |
| **cat.llm**    | Meta-package (installs all)   | —               | Re-exports + domain-suffixed aliases (`classify_survey()`, `classify_political()`, etc.) |

Each domain package depends on `cat.stack`, which holds the shared
classification engine.

## Quick example

```r
library(cat.survey)

results <- classify(
  input_data      = df$open_ended_response,
  categories      = c("Cost", "Quality", "Service", "Other"),
  survey_question = "Why did you choose us?",
  api_key         = Sys.getenv("OPENAI_API_KEY"),
  user_model      = "gpt-4o-mini"
)

write.csv(results, "coded.csv", row.names = FALSE)
```

The output is a regular `data.frame` ready for downstream analysis in R,
Stata, or Python.

## Source layout

```
r-package/
├── cat.stack/           ← General-purpose engine
├── cat.survey/          ← Survey responses
├── cat.vader/           ← Social media
├── cat.ademic/          ← Academic papers
├── cat.cog/             ← Cognitive assessment (CERAD)
├── cat.pol/             ← Policy documents
├── cat.web/             ← Web content
├── cat.llm/             ← Meta-package
└── test-all-packages.R  ← End-to-end smoke test (8/8 PASS expected)
```

## Verifying a local install

After cloning the parent `cat-llm` repo:

```bash
OPENAI_API_KEY=sk-... Rscript r-package/test-all-packages.R
```

This installs all 8 R packages from local source, installs the Python
backends via reticulate, and runs a minimal classification call against
each. Expected output: `8 / 8 passed (0 failed, 0 skipped)`.

## Reproducible research citation

See the [parent README](../README.md#academic-research) for the BibTeX
citation. The R wrappers are versioned alongside the Python packages, so
results obtained via the R interface are bitwise-equivalent to those from
Python (the R layer is a thin reticulate shim).

## License

GPL-3.0-or-later
