# catllm (Stata)

A Stata interface to the [`cat-stack`](https://pypi.org/project/cat-stack/) Python
package for LLM-powered classification, extraction, exploration, and
summarization of open-ended text. Mirrors the
[Python](../examples/) and [R](../r-package/) workflows so you can stay in Stata.

## Installation

```stata
net install catllm, ///
    from("https://raw.githubusercontent.com/chrissoria/cat-llm/main/stata-package/") ///
    replace

catllm setup                  // one-time: install the Python backend
catllm setup, domain(all)     // optional: + the 6 domain sub-packages
catllm setup, check           // probe what's installed
```

Requires Stata 16+ with Python integration (`python query` to verify), Python 3.8+,
and an API key from a supported provider (or a local Ollama install for no-API
workflows). After SSC submission, `ssc install catllm` will also work.

## Commands

| Command | Returns | Use when |
|---|---|---|
| [`catllm classify`](catllm_classify.sthlp) | new variable with one category label per row | you already have a category scheme |
| [`catllm extract`](catllm_extract.sthlp) | `r()` macros listing top categories | you want the model to discover a scheme |
| [`catllm explore`](catllm_explore.sthlp) | `r()` macros listing every raw category | you're doing saturation analysis |
| [`catllm summarize`](catllm_summarize.sthlp) | new `strL` variable with a summary per row | you want digests rather than labels |
| [`catllm cerad`](catllm_cerad.sthlp) | scored variables for drawn-shape recall | you're scoring CERAD cognitive test images |
| [`catllm setup`](catllm_setup.sthlp) | `r(status)` | (re)install or probe the Python backend |

## Quickstart

```stata
* 1. Set your key (or read from the environment)
global OPENAI_API_KEY : env OPENAI_API_KEY

* 2. Classify a column of free text
catllm classify response,                                          ///
    categories(                                                    ///
        "Positive: The respondent expresses satisfaction or approval." ///
        "Negative: The respondent expresses frustration or criticism." ///
        "Neutral: The response is factual or ambivalent."          ///
        "Other: The response does not fit any category.")          ///
    apikey($OPENAI_API_KEY)                                        ///
    model("gpt-4o-mini")                                           ///
    generate(sentiment)

list response sentiment, separator(0)
```

For zero-cost local-model workflows, use Ollama (`provider("ollama")`,
`model("qwen2.5:14b")`, any non-empty placeholder for `apikey`).

## Domain-specific prompts

Six sub-packages ship prompts tuned for particular text types. Select via
the `domain()` option:

```stata
catllm classify response, categories("Yes" "No" "Unclear")  ///
    apikey($OPENAI_API_KEY) domain(survey)
```

| `domain(...)` | Backend Python package | Tuned for |
|---|---|---|
| `pol` | `cat-pol` | political opinion text |
| `vader` | `cat-vader` | sentiment (with VADER comparison) |
| `ademic` | `cat-ademic` | academic abstracts and methods |
| `survey` | `cat-survey` | survey free-text responses |
| `cog` | `cat-cog` | cognitive-assessment drawings (CERAD) |
| `web` | `cat-web` | web-scraped content |

Install a domain package with `catllm setup, domain(name)`.

## Multi-model ensembles

Run several models in parallel and take a consensus vote:

```stata
local ens "gpt-4o-mini openai $OPENAI_API_KEY; claude-haiku-4-5-20251001 anthropic $ANTHROPIC_API_KEY; gemini-2.5-flash google $GOOGLE_API_KEY"

catllm classify response, categories("A" "B" "C") apikey($OPENAI_API_KEY) ///
    models("`ens'") consensus("two-thirds") generate(label)
```

Mix cloud and local models freely. See [`examples/03-ensemble-classification-cloud-and-local.do`](examples/03-ensemble-classification-cloud-and-local.do).

## Automatic prompt optimization (APO)

Iteratively rewrite the classification prompt based on a small sample you
hand-correct via a browser UI:

```stata
catllm classify response, categories(`cats') apikey($OPENAI_API_KEY) ///
    prompttune(10) tuneiterations(3) tuneui(browser)
```

See [`examples/test_prompt_tune.do`](examples/test_prompt_tune.do).

## Pass-through to cat-stack

Anything in the underlying Python signature that isn't a first-class Stata
option is reachable via `pyoptions()`:

```stata
catllm classify response, categories(`cats') apikey($OPENAI_API_KEY) ///
    pyoptions("max_retries=3, retry_delay=0.5, embeddings=True")
```

Values are parsed with Python's `ast.literal_eval` so numbers, booleans,
strings, and lists all work naturally.

## Examples

The [`examples/`](examples/) folder has 8 numbered walkthroughs mirroring the
R and Python example sets, plus interactive verification scripts. Start with
`01-classifying-text-with-ollama.do` if you have Ollama installed, or
`03-ensemble-classification-cloud-and-local.do` if you have cloud API keys.

## Tips for accuracy

- **Verbose category labels classify several percentage points better than
  one-word labels** on every model we've tested. Write each category as a
  one-sentence definition: `"Positive: The respondent expresses satisfaction
  or approval."` not just `"Positive"`.
- **For weak local models** (qwen2.5:7b, llama3.2): two-step classification
  with the fine-tuned JSON formatter is auto-enabled for Ollama. To force
  the same path on lower-tier cloud models (gpt-4o-mini, claude-haiku),
  pass `twostepclassify(true)`.
- **For ambiguous text** the model can't classify cleanly: add an explicit
  `"Other: The response does not fit any of the above categories."` category.

## Author / License / Issues

- Author: Christopher Soria (`chrissoria@berkeley.edu`), University of California, Berkeley
- License: GPL-3.0 (see [`LICENSE`](../LICENSE) in the repo root)
- Issues: [github.com/chrissoria/cat-llm/issues](https://github.com/chrissoria/cat-llm/issues)
- Python backend source: [github.com/chrissoria/cat-stack](https://github.com/chrissoria/cat-stack)
