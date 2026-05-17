# CatLLM R Ecosystem

This directory contains the R interface to the [CatLLM](https://catllm.com)
Python ecosystem — eight R packages that wrap their Python counterparts via
[reticulate](https://rstudio.github.io/reticulate/) so R users can do
LLM-powered text classification with the same API as Python users.

> **📖 Full guide:** the canonical R user guide is the **Getting Started
> vignette** in `cat.llm`. Read it online at
> <https://chrissoria.r-universe.dev/cat.llm> (Vignettes tab), or inside R
> after install:
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

| Package        | Domain                              | Python backend  |
|----------------|-------------------------------------|-----------------|
| **cat.stack**  | General-purpose classification base | `catstack`      |
| **cat.survey** | Open-ended survey responses         | `catsurvey`     |
| **cat.vader**  | Social media posts                  | `catvader`      |
| **cat.ademic** | Academic papers (OpenAlex)          | `catademic`     |
| **cat.cog**    | Cognitive assessment scoring        | `catcog`        |
| **cat.pol**    | Policy documents                    | `catpol`        |
| **cat.web**    | Web content (URL fetching)          | `catweb`        |
| **cat.llm**    | Meta-package (installs all 7)       | —               |

Each domain package depends on `cat.stack` (the shared classification engine).

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
