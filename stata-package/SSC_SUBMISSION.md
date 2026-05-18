# SSC submission: catllm

**Draft cover note. Last revised: 2026-05-18.**

Send to **kit.baum@bc.edu** with the subject line:

> SSC submission: catllm

Attach a zip of the package's runnable contents (the `.pkg`, `.toc`,
`.ado`, and `.sthlp` files — not the `examples/` subdirectory).
Distribution-Date in `catllm.pkg` and `stata.toc` should match the day
you actually send the email; update both if the send date slips.

---

## Cover note

Dear Kit,

I would like to submit a new package, **`catllm`**, for inclusion in SSC.

**Purpose.** `catllm` is a Stata interface to the `cat-stack` Python package
for automating the categorization of open-ended text using large language
models (LLMs). It exposes four verbs — `catllm classify`, `catllm extract`,
`catllm explore`, `catllm summarize` — plus `catllm setup` to install the
Python backend and `catllm cerad` for scoring CERAD drawn-shape recall
tests. Domain-specific prompts (political opinion, sentiment, academic,
survey, cognitive, web) are selectable via a `domain()` option. Supports
9 LLM providers including OpenAI, Anthropic, Google, Mistral, xAI, and
local models via Ollama.

**Why it is useful.** Coding open-ended survey responses by hand is one of
the most time-consuming tasks in applied social-science research. `catllm`
brings the same LLM-assisted classification pipeline that is already
established in Python and R workflows into Stata, so analysts can stay in
their native environment and reproduce the same coding scheme across all
three languages. The package handles category-scheme discovery, single-
and multi-model classification (with consensus voting), and text
summarization. A two-step natural-language-then-format prompting path with
a fine-tuned JSON-formatter fallback makes it usable with small local
models (e.g. `qwen2.5:7b` via Ollama) for cost-free or privacy-sensitive
workflows.

**Output shape.** `catllm classify` produces one byte indicator variable
per category, named `<prefix>_<short_label>` (0/1), matching the wide
DataFrame shape Python and R users see from `cat-stack`'s `classify()`.
Multi-label classification is first-class: a single response can have
multiple indicators set to 1. Standard Stata verbs (`tab1`, `summarize`,
`regress`, `egen`) work directly on the resulting variables.

**Originality.** I have searched the SSC archive and other Stata package
repositories and am not aware of any existing package named `catllm` or
providing equivalent LLM-classification functionality. The closest
existing tools (`chatgpt`, `gpt2stata`, etc.) wrap individual API calls;
`catllm` provides a full classification/extraction pipeline with consensus
voting, JSON-schema validation, automatic prompt verbosity checking, and
optional domain-specific prompts.

**Requirements.** Stata 16+ with Python integration (`python query` to
verify), Python 3.8+, and an API key from a supported provider (or a local
Ollama install for no-API-key workflows). The Python backend is
[`cat-stack`](https://pypi.org/project/cat-stack/) **≥ 1.4.0**; the user
runs `catllm setup` once after installation to install it from PyPI. The
Stata wrapper is intentionally thin — string parsing, domain resolution,
output shaping, and schema validation all live server-side in `cat-stack`,
which means new Python features become usable from Stata without a
package update.

**Package contents.** 7 `.ado` files and 7 matching `.sthlp` files, plus
`catllm.pkg` and `stata.toc`. License: GPL-3.0 (LICENSE file in the
source repository).

**Source repository and documentation.**
- GitHub: https://github.com/chrissoria/cat-llm
- Python backend (`cat-stack`): https://pypi.org/project/cat-stack/
- Worked examples (mirroring Python and R): `stata-package/examples/`

Please let me know if you need any changes to the format or contents.

Best regards,
Christopher Soria
University of California, Berkeley
chrissoria@berkeley.edu

---

## Zip contents (what to attach)

From the repository root:

```bash
cd stata-package
zip -r /tmp/catllm-ssc.zip \
    catllm.pkg stata.toc \
    catllm*.ado catllm*.sthlp
```

This produces a flat archive of:
- `stata.toc`, `catllm.pkg`
- `catllm.ado`, `catllm_classify.ado`, `catllm_extract.ado`,
  `catllm_explore.ado`, `catllm_summarize.ado`, `catllm_setup.ado`,
  `catllm_cerad.ado`
- `catllm.sthlp`, `catllm_classify.sthlp`, `catllm_extract.sthlp`,
  `catllm_explore.sthlp`, `catllm_summarize.sthlp`, `catllm_setup.sthlp`,
  `catllm_cerad.sthlp`

The `examples/`, `tests/`, and `worktrees/` subdirectories are NOT included
— SSC archives only ship the runnable package, and examples live in the
GitHub repo.
