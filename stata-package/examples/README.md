# CatLLM Stata Examples

Stata `.do` walkthroughs mirroring the [Python `examples/` notebooks](../../examples/) and [R `examples/` Rmd files](../../r-package/examples/). Each `.do` file is a self-contained example covering one use case — install the relevant package, set your API key, run.

| File | What it covers |
|---|---|
| [01-classifying-text-with-ollama.do](01-classifying-text-with-ollama.do) | Classify text against a local Ollama model — zero API cost, fully local |
| [02-validate-hybrid-ensemble.do](02-validate-hybrid-ensemble.do) | Smoke-test the ensemble pipeline end-to-end (Ollama auto-start, JSON-formatter auto-enable, expected output columns) |
| [03-ensemble-classification-cloud-and-local.do](03-ensemble-classification-cloud-and-local.do) | Combine predictions from multiple LLMs with unanimous voting |
| [04-exploring-categories-with-explore.do](04-exploring-categories-with-explore.do) | Use `catllm explore` to surface raw categories for saturation analysis |
| [05-extracting-categories-from-columns.do](05-extracting-categories-from-columns.do) | End-to-end: extract a category scheme from a Stata variable, then classify |
| [06-extracting-categories-with-extract.do](06-extracting-categories-with-extract.do) | Discover and de-duplicate categories with `catllm extract` when you have no predefined scheme |
| [07-summarizing-text-and-pdf.do](07-summarizing-text-and-pdf.do) | Use `catllm summarize` on text and (optionally) PDF inputs |
| [08-extracting-and-exploring-with-ollama.do](08-extracting-and-exploring-with-ollama.do) | Run `catllm extract` and `catllm explore` entirely on a local Ollama model — no API keys, saturation analysis included |

The regression-test file [`test_stata_package.do`](test_stata_package.do) exercises the whole package surface in one run (error paths, `domain()`, `pyoptions()`, etc.) and is the smoke-test we use before tagging a release.

## Prerequisites

- Stata 16+ with Python integration. Check with `python query`.
- A one-time install of the Python backend:

  ```stata
  catllm setup                  // base cat-stack
  catllm setup, domain(all)     // base + all 6 domain sub-packages
  catllm setup, check           // probe what's installed
  ```

- An API key from a supported provider, exported in your shell or stored in `profile.do`:

  ```stata
  global OPENAI_API_KEY    "sk-..."
  global ANTHROPIC_API_KEY "sk-ant-..."
  global GOOGLE_API_KEY    "AI..."
  ```

  Or read from the environment at the top of a `.do` file:

  ```stata
  global OPENAI_API_KEY : env OPENAI_API_KEY
  ```

## Running an example

From the `stata-package/examples/` directory:

```bash
stata-se -b do 02-validate-hybrid-ensemble.do
```

Or in an interactive Stata session:

```stata
do 02-validate-hybrid-ensemble.do
```

If the `catllm` package is installed via `ssc` or `net install`, the `adopath +` line at the top of each example is unnecessary. The examples include it so you can run them straight from a git checkout without first installing.

## When to use which command

| Command | Returns | Use when |
|---|---|---|
| `catllm classify` | New variable with one category label per row | You already have a category scheme |
| `catllm extract`  | r() macros listing top categories | You want the model to discover a scheme |
| `catllm explore`  | r() macros listing every raw category (with duplicates) | You're doing saturation analysis |
| `catllm summarize` | New `strL` variable with a summary per row | You want digests rather than labels |
| `catllm cerad`    | Score variables for CERAD drawn-shape recall | You're scoring cognitive test drawings |

## Verbose categories classify more accurately

The single biggest accuracy improvement is writing **verbose, definition-style** category labels rather than one-word labels. Each category must be **one quoted string** — Stata splits on whitespace between quote pairs, so if you wrap a single definition across two strings it will be parsed as two categories.

```stata
* Less accurate
catllm classify response,                                       ///
    categories("Job" "Family" "Cost" "Other") ...

* More accurate -- one definition per quoted string
catllm classify response,                                                                    ///
    categories(                                                                              ///
        "Job/school: A change in employment, education, or career, including retirement."   ///
        "Family: Relationship changes, having children, or relocating to be near family."   ///
        "Cost of living: Housing affordability, cost of goods, or general economic pressure." ///
        "Other: The response does not fit any of the above categories.")                     ///
    ...
```

For more, see the Python README's "Best Practices for Classification" section.
