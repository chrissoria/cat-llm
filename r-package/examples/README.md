# CatLLM R Examples

R Markdown walkthroughs mirroring the [Python `examples/` notebooks](../../examples/). Each `.Rmd` is a self-contained example covering one use case — install the relevant package, set your API key, run.

| File | What it covers |
|---|---|
| [01-classifying-text-with-ollama.Rmd](01-classifying-text-with-ollama.Rmd) | Classify text against a local Ollama model — zero API cost, fully local |
| [02-validate-hybrid-ensemble.Rmd](02-validate-hybrid-ensemble.Rmd) | Smoke-test the ensemble pipeline end-to-end (Ollama auto-start, JSON-formatter auto-enable, expected output columns) |
| [03-ensemble-classification-cloud-and-local.Rmd](03-ensemble-classification-cloud-and-local.Rmd) | Combine predictions from multiple LLMs with unanimous voting |
| [04-exploring-categories-with-explore.Rmd](04-exploring-categories-with-explore.Rmd) | Use `explore()` to surface raw categories for saturation analysis |
| [05-extracting-categories-from-columns.Rmd](05-extracting-categories-from-columns.Rmd) | End-to-end: extract a category scheme from a data.frame column, then classify |
| [06-extracting-categories-with-extract.Rmd](06-extracting-categories-with-extract.Rmd) | Discover and de-duplicate categories with `extract()` when you have no predefined scheme |
| [07-summarizing-text-and-pdf.Rmd](07-summarizing-text-and-pdf.Rmd) | Use `summarize()` on text and PDF inputs |

## Running an example

Each `.Rmd` is meant to be **read on GitHub** (it renders inline) or **knit locally** if you want to actually execute it. To run, install the packages, set your `OPENAI_API_KEY`, and source the chunks.

```r
# One-time install
install.packages("cat.llm",
                 repos = c("https://chrissoria.r-universe.dev",
                          "https://cloud.r-project.org"))
library(cat.llm)
install_cat_stack()

# Run an example
rmarkdown::render("01-classifying-text-with-ollama.Rmd")
```

For the canonical R user guide, see the **Getting Started vignette**:
`vignette("getting-started", package = "cat.llm")` or
[chrissoria.r-universe.dev/cat.llm](https://chrissoria.r-universe.dev/cat.llm).
