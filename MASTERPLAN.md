# CatLLM Ecosystem — Master Plan

## Vision

**cat-llm** keeps its brand and identity as the meta-package — the tidyverse of
LLM-powered text analysis for social science. Installing `cat-llm` pulls in a family
of domain-specific sub-packages, each with its own PyPI release, versioning, and focus.

The shared infrastructure and general-purpose base lives in **`cat-stack`**: a
standalone package for classifying any text column, with no domain assumptions.
Domain-specific packages (`cat-survey`, `cat-vader`, etc.) extend `cat-stack`
with domain-tuned prompts, metadata injection, and workflow helpers.

```
pip install cat-llm
# brings in:
#   cat-stack   general-purpose text column classification (the base)
#   cat-survey    survey response classification & extraction
#   cat-vader     social media (Reddit, Twitter/X, forums)
#   cat-ademic    academic papers, PDFs, citations
#   cat-pol       political text (manifestos, speeches, legislation)
```

**Dependency graph** (modeled on tidyverse / rlang):

```
cat-stack                    ← general base + shared infra (like rlang)
                                  independently useful; no domain assumptions
    ↑
cat-survey  cat-vader  cat-ademic  cat-pol    ← domain packages, each depends on cat-stack
    ↑           ↑          ↑          ↑
                    cat-llm                    ← pure meta-package, keeps the brand
                                               (depends on all domain packages + cat-stack)
```

Every package is independently installable:
```
pip install cat-stack   # general text classification, no domain framing
pip install cat-survey    # survey-specific, pulls in cat-stack automatically
pip install cat-llm       # everything
```

---

## The Packages

### `cat-stack` *(new — the base)*
General-purpose text column classification and extraction with no domain assumptions.
This replaces the need for a separate invisible `catcore` — it is both the shared
infrastructure layer AND independently useful for researchers who just have a text
column and don't fit neatly into a domain.

**Scope:**
- `classify()`, `extract()`, `explore()`, `summarize()` for any text input
- `UnifiedLLMClient`, `PROVIDER_CONFIG`, `detect_provider` — all provider infrastructure
- `_batch.py` — async batch API logic
- Shared text utilities: `build_json_schema`, `extract_json`, `validate_classification_json`
- No survey framing, no social media framing, no PDF-first assumptions

**Status:** Does not exist yet. Code lives inside `cat-llm/src/catllm/` and needs to be
extracted. This is the prerequisite for all other phases.

---

### `cat-survey` *(extract from current cat-llm)*
Survey response classification — the current heart of `cat-llm`, spun out as its
own package. Keeps the paper methodology, R/Stata wrappers, and survey-specific defaults.

**Scope:**
- `classify()` / `extract()` / `explore()` / `summarize()` with survey-tuned prompts
- All prompt strategies from the paper: CoT, CoVe, step-back, few-shot examples
- Defaults calibrated for survey data (temperature, verbosity checks, etc.)
- R and Stata wrappers
- HuggingFace Space: `CatLLM/survey-classifier`

**Status:** Code exists inside `cat-llm`. Needs to be extracted into its own repo
and published, then wired back into `cat-llm` as a dependency.

---

### `cat-vader` *(already separate)*
Social media text classification with platform-aware context injection.

**Scope:**
- `classify()` / `extract()` / `explore()` with social media metadata fields
  (platform, handle, hashtags, engagement metrics injected into prompt)
- Reddit thread classification (nested comment structure)
- HuggingFace Space: `CatLLM/CatVader`

**Status:** Separate repo and PyPI package. Needs to be updated to depend on
`cat-stack` instead of carrying its own provider copy, then wired into `cat-llm`.

---

### `cat-ademic` *(planned)*
Classification and extraction for academic and long-form documents.

**Scope:**
- PDF-first `classify()` / `extract()` — builds on current `pdf_functions.py` logic
- Per-page and whole-document classification modes
- Citation and abstract extraction
- CERAD scoring (natural home here given image + PDF focus)
- Designed for systematic reviews, coding codebooks, content analysis of papers

**Status:** Not started. Core PDF machinery in `cat-llm/src/catllm/pdf_functions.py`
and `image_functions.py` becomes the seed.

---

### `cat-pol` *(future)*
Political text analysis — manifestos, speeches, legislation, news.

**Scope:**
- Domain-tuned prompts for political science categories
  (ideology, policy area, sentiment toward government, actor identification)
- Multi-language support (EU parliament, UN documents)
- Manifesto coding scheme compatibility (MARPOR/CMP)
- Time-series classification across document corpora

**Status:** Concept only. No code yet.

---

### `cat-llm` *(stays as meta-package)*
Keeps its brand identity as the survey-analysis flagship and the entry point for the
full ecosystem. Marketing and documentation continues to emphasize survey use cases,
but installing it gives you everything.

**Scope:**
- `pyproject.toml` lists all sub-packages as dependencies
- `__init__.py` re-exports all sub-package public APIs
- README, academic citation, HuggingFace org hub
- No domain logic lives here

---

## Target Package Structure

```
cat-stack/                  ← general base + shared infra (like rlang)
├── src/catstack/
│   ├── _providers.py         ← UnifiedLLMClient, PROVIDER_CONFIG
│   ├── _batch.py             ← batch API logic
│   ├── text_functions.py     ← shared text utilities
│   ├── classify.py           ← domain-agnostic classify()
│   ├── extract.py            ← domain-agnostic extract()
│   ├── explore.py            ← domain-agnostic explore()
│   └── summarize.py          ← domain-agnostic summarize()

cat-survey/                   ← survey package (depends on cat-stack)
│   survey-tuned classify(), extract(), explore(), summarize()
│   R + Stata wrappers

cat-vader/                    ← social media (depends on cat-stack)
│   classify(), extract(), explore() with social metadata

cat-ademic/                   ← academic/PDF (depends on cat-stack)
│   classify(), extract() for PDFs, CERAD scoring

cat-pol/                      ← political text (depends on cat-stack)
│   classify(), extract() with political science prompt library

cat-llm/                      ← pure meta-package, keeps the brand
│   depends on: cat-stack, cat-survey, cat-vader, cat-ademic, cat-pol
│   src/catllm/__init__.py re-exports all sub-package public APIs
│   No domain logic lives here
```

---

## Migration Path

### Phase 1 — Stabilize (now)
- Ship features in `cat-llm` as-is; no structural changes yet.
- Draw a clear internal boundary between general-purpose code (→ `cat-stack`)
  and survey-specific code (→ `cat-survey`). This is the design prerequisite.

### Phase 2 — Create `cat-stack`
- Extract `_providers.py`, `_batch.py`, `text_functions.py`, and the four core
  entry-point functions (`classify`, `extract`, `explore`, `summarize`) into a new repo.
- Publish `cat-stack` to PyPI.
- Update `cat-vader` to depend on `cat-stack`.

### Phase 3 — Extract `cat-survey`
- Move survey-specific code (prompt strategies, defaults, R/Stata wrappers) into a
  new `cat-survey` repo; add `cat-stack` as a dependency.
- Publish `cat-survey` to PyPI independently.
- Slim `cat-llm` to a meta-package: `pyproject.toml` lists sub-packages as deps,
  `__init__.py` re-exports their public APIs.
- Version bump `cat-llm`; update README.

### Phase 4 — Build `cat-ademic`
- Seed from `pdf_functions.py`, `image_functions.py`, CERAD functions currently in cat-llm.
- Publish `cat-ademic`; add to `cat-llm` dependencies.

### Phase 5 — `cat-pol` (when ready)
- Develop domain-tuned prompt library for political science.
- Publish and wire into `cat-llm`.

---

## API Consistency Across Packages

All packages expose the same four verbs — modeled on tidyverse's consistent grammar:

| Function | Purpose |
|----------|---------|
| `classify()` | Assign predefined categories to documents |
| `extract()` | Discover categories from a corpus |
| `explore()` | Raw extraction for saturation analysis |
| `summarize()` | Summarize documents |

Domain-specific behavior is injected through keyword arguments
(`platform=`, `metadata=` in cat-vader; `page_mode=` in cat-ademic),
never by changing function names. Learn once, apply anywhere.

The `pyproject.toml` for `cat-llm` will look like:

```toml
[project]
name = "cat-llm"
dependencies = [
    "cat-stack~=1.0",
    "cat-survey~=1.0",
    "cat-vader~=1.0",
    "cat-ademic~=1.0",
]
```

---

## Open Questions

- **Namespace**: Flat import names (`import catstack`, `import catsurvey`) matching
  the tidyverse pattern. All importable by their own names whether installed directly
  or via `cat-llm`.
- **Versioning**: Compatible-release (`~=`) constraints so patch updates propagate
  automatically but breaking changes require an explicit `cat-llm` bump.
- **R/Stata wrappers**: Live in `cat-survey`. A future R meta-package could mirror
  the Python ecosystem.
- **HuggingFace Spaces**: One Space per domain package (current approach). A unified
  multi-tab Space under `CatLLM/` is a long-term goal once 2+ Spaces exist.
- **`cat-stack` marketing**: Position it as "classify any text column" —
  useful for researchers outside the survey/social-media domains who want the
  LLM pipeline without domain framing.
