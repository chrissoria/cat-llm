# Stata Conference materials

Poster, abstract, and the reproducible do-file that goes with them.

```
conference/
├── abstract.md                     submission-ready abstract (two lengths)
├── catllm_stata_conference.do      the handout — runs everything on the poster
└── poster/
    ├── poster.qmd                  poster source (Quarto)
    ├── poster.css                  48 x 36 in landscape stylesheet
    ├── make_figures.py             regenerates figures/ from source numbers
    ├── build.py                    pandoc-only fallback if you lack Quarto
    ├── render_pdf.py               poster.html -> print-ready PDF
    ├── assets/                     QR code, logo
    └── figures/                    generated — do not edit by hand
```

## Building the poster

With Quarto (the normal path):

```bash
cd poster
python make_figures.py      # only needed if the numbers changed
quarto render poster.qmd
python render_pdf.py        # -> poster_poster.pdf, 48 x 36 in
```

Without Quarto, `build.py` runs the same body through pandoc and produces an
equivalent `poster.html`:

```bash
python build.py && python render_pdf.py
```

`render_pdf.py` calls `pdfcrop` at the end to trim trailing whitespace. That step
is optional — if `pdfcrop` isn't installed, the PDF is left at the full
48 × 36 in, which is what you want for print anyway.

If Playwright can't find a browser, point it at one:

```bash
CHROMIUM_PATH=/path/to/chrome python render_pdf.py
```

### Changing the poster size

Conference boards vary. Edit two places in `poster.css` — `@page { size: ... }`
and the `.poster-shell` width/height — and everything else reflows.

## The do-file

`catllm_stata_conference.do` is the takeaway for attendees. It is self-contained:
the data is embedded with `input`, so there are no downloads and no file paths to
fix. Parts 3, 4, and 6 are guarded and skip themselves with a message when the
key or local model they need is missing, so the file runs start to finish with a
single API key — or with none at all if Ollama is installed.

```stata
net install catllm, ///
  from("https://raw.githubusercontent.com/chrissoria/cat-llm/main/stata-package/") ///
  replace

catllm setup
do catllm_stata_conference.do
```

Six parts: preflight, the data, the core classify pipeline, a three-provider
comparison, ensemble voting with agreement scores, category discovery with
`extract`, and the zero-cost local route.

## Where the numbers come from

Every figure on the poster traces to a source in this repository. `make_figures.py`
documents the mapping in its module docstring:

| Poster claim | Source |
|---|---|
| 97% straightforward / 88–91% complex; 95–96% and 87% open-weight; brevity effect | `academic_examples/README.md` (UCNETS validation) |
| 98% vs. human consensus | `README.md` |
| 8 models, 3,208 responses, 25,664 classifications, 100% valid structured output, $0.38–$27.85, 23 min–7 hr | `academic_examples/paper.md` |

Two figures (`fig1_pipeline`, `fig4_ensemble`) are schematics and encode no data.

The cost and runtime figure plots only the published endpoints, because only the
endpoints are published. If you have the full per-model benchmark table, a
per-model scatter of cost against accuracy would be a stronger panel than the
range bars currently there — that is the one place on the poster where better
data would visibly improve it.
