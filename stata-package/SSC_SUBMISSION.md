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

Submitting a new package, catllm, for SSC. It's a Stata wrapper around the
cat-stack Python package that lets users classify, summarize, and discover
themes in open-ended text using LLMs (OpenAI, Anthropic, Google, plus
local models via Ollama). The use case I built it for is coding survey
free-text without moving the data out of Stata.

I checked the SSC archive and don't see anything overlapping in name or
scope. License is GPL-3.0. It needs Stata 16+, Python 3.8+, and the
cat-stack Python backend (>= 1.6.0), which the included catllm setup
command installs from PyPI.

Source and worked examples are at https://github.com/chrissoria/cat-llm.
The attached zip has 7 .ado files, 7 .sthlp files, catllm.pkg, and
stata.toc. Let me know if anything needs reformatting.

Thanks,
Chris Soria
UC Berkeley
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
