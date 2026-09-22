# Stata Conference poster — a worked Stata example

Conference materials for the `catllm` Stata package: a poster, a submission
abstract, and the reproducible do-file that goes with them. Kept here in
`academic_examples/` rather than in `stata-package/` because it is an example of
using the package, not part of what ships to SSC.

```
stata_conference_poster/
├── abstract.md                     submission-ready abstract (two lengths)
├── catllm_stata_conference.do      the handout — runs everything on the poster
└── poster/
    ├── poster.qmd                  poster source (Quarto)
    ├── poster.css                  48 x 36 in landscape stylesheet
    ├── make_figures.py             regenerates figures/ from source numbers
    ├── unanimous_ensemble.R        unanimous-vote ensemble F1s for fig4 (needs an
    │                                 external companion benchmark -- see its header)
    ├── build.py                    pandoc-only fallback if you lack Quarto
    ├── render_pdf.py               poster.html -> print-ready PDF
    ├── assets/                     QR code, logo
    └── figures/                    generated — do not edit by hand
```

## Building the poster

```bash
cd poster
python make_figures.py      # only needed if the numbers changed
quarto render poster.qmd
python render_pdf.py        # -> poster_poster.pdf, 48 x 36 in
```

The committed `poster.html` and `poster_poster.pdf` were built this way and fit
the 36 in height exactly, with the three columns balanced to within 25 px.

Note the `theme: none` and `minimal: true` in the YAML header: without them
Quarto loads Bootstrap, which constrains the body width and collapses the
48 in layout. Leave them in.

If you don't have Quarto, `build.py` runs the same body through pandoc and
produces an equivalent `poster.html`:

```bash
python build.py && python render_pdf.py
```

`render_pdf.py` calls `pdfcrop` at the end to trim trailing whitespace. That step
is optional — without `pdfcrop` the PDF is left at the full 48 × 36 in, which is
what you want for print anyway. If Playwright can't find a browser, point it at
one with `CHROMIUM_PATH=/path/to/chrome python render_pdf.py`.

### Changing the poster size

Conference boards vary. Edit two places in `poster.css` — `@page { size: ... }`
and the `.poster-shell` width/height — then re-render and check the reported
height; the columns are balanced for the current dimensions and will need
rebalancing if the aspect ratio changes much.

## The do-file

`catllm_stata_conference.do` is the takeaway for attendees, and doubles as a
worked example of the Stata package end to end. It is self-contained: the data is
embedded with `input`, so there are no downloads and no file paths to fix. Parts
3, 4, and 6 are guarded and skip themselves with a message when the key or local
model they need is missing, so the file runs start to finish with a single API
key — or with none at all if Ollama is installed.

```stata
net install catllm, ///
  from("https://raw.githubusercontent.com/chrissoria/cat-llm/main/stata-package/") ///
  replace

catllm setup
do catllm_stata_conference.do
```

Six parts: preflight, the data, the core classify pipeline into `tabulate` and
`regress`, a three-provider comparison with `kap`, ensemble voting with
agreement scores, category discovery with `extract`, and the zero-cost local
route. The narrower per-feature examples live in
[`stata-package/examples/`](../../stata-package/examples/).

## Where the numbers come from

Every figure traces to a source in this repository; `make_figures.py` documents
the mapping in its module docstring.

| Poster claim | Source |
|---|---|
| 97% straightforward / 88–91% complex ("Tested on multi-label, multi-class items..." in the Abstract) | [`../README.md`](../README.md) (UCNETS validation) |
| 98% vs. human consensus | [root `README.md`](../../README.md) |
| 8 models, 3,208 responses, 25,664 classifications, 100% valid structured output | [`../paper.md`](../paper.md) |
| Cloud-tier and local-model macro F1 (fig2, fig3), top-3 unanimous-ensemble precision and recall (fig4) | an unpublished companion benchmark, not included in this repository — see `make_figures.py`'s module docstring and `unanimous_ensemble.R`'s header for the exact computation |

`fig1_pipeline` is a schematic and encodes no data.

Note: the Abstract's headline numbers and fig2/fig3/fig4 come from two different
studies (different metric too — raw agreement vs. macro F1) that happen to sit
next to each other on the poster. They're not in tension, but a reader comparing
them closely may notice the numbers don't obviously reconcile.
