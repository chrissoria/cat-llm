# Stata Conference — poster abstract

**Status:** draft, ready to submit. Registration and submission have to be done
by hand at the conference site (see "Before you submit" at the bottom).

---

## Title

Coding Open-Ended Survey Text Without Leaving Stata: the `catllm` Package for
LLM-Assisted Content Coding

## Presenter

Chris Soria, Department of Demography, University of California, Berkeley
(chrissoria@berkeley.edu)

## Presentation type

Poster

---

## Abstract (~250 words)

Open-ended survey items capture what closed-ended items cannot, and are
routinely dropped from analysis because hand-coding them does not scale. A
single free-text item in a 3,000-respondent survey is weeks of research-assistant
time, and researchers who turn to large language models generally leave their
statistical environment to do it — exporting to CSV, scripting in Python, and
merging results back, losing the audit trail that makes coding reproducible.

`catllm` puts the whole loop inside Stata. It is a thin `.ado` layer over a
Python backend (`cat-stack`), exposing five verbs — `classify`, `extract`,
`explore`, `summarize`, and `setup` — and supporting OpenAI, Anthropic,
and Google models as well as local open-weight models through Ollama, so that
restricted-use data need never leave the analyst's machine. `catllm classify`
takes a string variable and a set of category definitions and returns ordinary
Stata indicator variables, multi-label by construction and immediately usable in
`tabulate` or `regress`.

This poster reports validation against human coders on survey free text:
proprietary models agree with human annotators on 97% of straightforward items
and 88–91% of complex interpretive ones, with open-weight models 1–2 points
behind; a three-model ensemble reaches 98% agreement with human consensus.
Across eight models and 25,664 classifications, cost for an identical job varied
by a factor of 73 and wall-clock time by a factor of 18, with accuracy only
loosely coupled to price. Because models disagree most on the responses humans
find hardest, `catllm` reports vote-level agreement rather than hiding it behind
a single label.

A self-contained do-file reproducing every result is distributed with the poster.

---

## Shorter variant (~120 words, if the submission form caps length)

Open-ended survey items are routinely dropped from analysis because hand-coding
them does not scale. `catllm` codes free text with large language models from
inside Stata, returning ordinary indicator variables ready for `tabulate` and
`regress`. It wraps a Python backend behind six verbs and supports OpenAI,
Anthropic, and Google models as well as local open-weight models through Ollama,
so restricted-use data need not leave the analyst's machine. Validated against
human coders, proprietary models agree on 97% of straightforward items and 88–91%
of complex interpretive ones; a three-model ensemble reaches 98% agreement with
human consensus. Cost for an identical job varies 73-fold across vendors with
accuracy only loosely coupled to price. A self-contained do-file reproducing
every result is distributed with the poster.

---

## Keywords

open-ended survey data · content coding · large language models · text
classification · reproducibility · Python integration

---

## Before you submit

Two things I could not do from here, both of which need your account:

1. **Register** at the conference site — this needs payment details and cannot
   be delegated.
2. **Check the submission requirements against the live call for papers.** The
   Stata site is blocked by this environment's network policy, so I could not
   confirm the current deadline, the word limit, whether posters are a separate
   submission track this year, or whether a short bio is required. Both abstract
   lengths above are provided so you can pick whichever fits the form.

Also worth confirming before printing: the poster is laid out at **48 × 36 in
landscape**, inherited from the CADAS dissertation template. If the conference
specifies different dimensions, change `@page { size: ... }` and the
`.poster-shell` width/height in `poster/poster.css` — everything else is
proportional and will reflow.
