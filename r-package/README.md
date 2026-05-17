# CatLLM R Ecosystem

This directory contains the R interface to the [CatLLM](https://catllm.com)
Python ecosystem — eight R packages that wrap their Python counterparts via
[reticulate](https://rstudio.github.io/reticulate/) so R users can do
LLM-powered text classification with the same API as Python users.

> **📖 Docs site:** every package has its own pkgdown site at
> **<https://christophersoria.com/cat-llm/>** with reference docs,
> vignettes, examples, and changelog. The cat.llm Getting Started
> vignette is the canonical long-form guide; in-R access:
>
> ```r
> vignette("getting-started", package = "cat.llm")
> ```

## Install

```r
install.packages(
  "cat.llm",
  repos = c("https://chrissoria.r-universe.dev",
            "https://cloud.r-project.org")
)

# One-time Python backend setup
library(cat.llm)
install_cat_stack()
```

## The packages

| Package        | Domain                              | Python backend  | Docs                                                       |
|----------------|-------------------------------------|-----------------|------------------------------------------------------------|
| **cat.stack**  | General-purpose classification base | `catstack`      | [cat.stack/](https://christophersoria.com/cat-llm/cat.stack/)   |
| **cat.survey** | Open-ended survey responses         | `catsurvey`     | [cat.survey/](https://christophersoria.com/cat-llm/cat.survey/) |
| **cat.vader**  | Social media posts                  | `catvader`      | [cat.vader/](https://christophersoria.com/cat-llm/cat.vader/)   |
| **cat.ademic** | Academic papers (OpenAlex)          | `catademic`     | [cat.ademic/](https://christophersoria.com/cat-llm/cat.ademic/) |
| **cat.cog**    | Cognitive assessment scoring        | `catcog`        | [cat.cog/](https://christophersoria.com/cat-llm/cat.cog/)       |
| **cat.pol**    | Policy documents                    | `catpol`        | [cat.pol/](https://christophersoria.com/cat-llm/cat.pol/)       |
| **cat.web**    | Web content (URL fetching)          | `catweb`        | [cat.web/](https://christophersoria.com/cat-llm/cat.web/)       |
| **cat.llm**    | Meta-package (installs all 7)       | —               | [cat.llm/](https://christophersoria.com/cat-llm/cat.llm/)       |

Each domain package depends on `cat.stack` (the shared classification engine).
Core functions are `classify()`, `extract()`, `explore()`, `summarize()`, and
`prompt_tune()` — all available from any package after loading.

## Vignettes

Each package ships a domain-specific vignette in its own pkgdown site:

- **cat.llm**: [Getting Started](https://christophersoria.com/cat-llm/cat.llm/articles/getting-started.html) — the canonical R user guide
- **cat.survey**: [Classifying Open-Ended Survey Responses](https://christophersoria.com/cat-llm/cat.survey/articles/classifying-survey-responses.html)
- **cat.vader**: [Classifying Social Media Posts](https://christophersoria.com/cat-llm/cat.vader/articles/classifying-social-media.html)
- **cat.ademic**: [Classifying Academic Papers](https://christophersoria.com/cat-llm/cat.ademic/articles/classifying-academic-papers.html)
- **cat.cog**: [CERAD Constructional Praxis Scoring](https://christophersoria.com/cat-llm/cat.cog/articles/cerad-scoring.html)
- **cat.pol**: [Classifying Policy Documents](https://christophersoria.com/cat-llm/cat.pol/articles/policy-document-analysis.html)
- **cat.web**: [Classifying Web Content](https://christophersoria.com/cat-llm/cat.web/articles/web-content-classification.html)

## Worked examples

R Markdown walkthroughs covering common use cases live in [`examples/`](examples/) — Ollama-local classification, multi-model ensembles, category discovery with `extract()`, saturation analysis with `explore()`, end-to-end column → classified-dataset flows, and PDF/text summarization. Mirror the [Python `examples/`](../examples/) notebooks.

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
├── cat.llm/             ← Meta-package + Getting Started vignette
├── examples/            ← R Markdown worked examples
└── test-all-packages.R  ← End-to-end smoke test (8/8 PASS expected)
```

## Verifying a local install

After cloning the parent `cat-llm` repo:

```bash
OPENAI_API_KEY=sk-... Rscript r-package/test-all-packages.R
```

Installs all 8 R packages from local source, installs the Python backends
via reticulate, and runs a minimal classification per package. Expected
output: `8 / 8 passed (0 failed, 0 skipped)`.

## Citation

If you use CatLLM in published research, see the [parent
README](../README.md#academic-research) for the BibTeX. R results are
bitwise-equivalent to Python results — the R layer is a thin reticulate
shim with no language-specific logic.

## License

GPL-3.0-or-later
