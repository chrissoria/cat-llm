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
#   cat-stack     general-purpose classification engine (the base)
#   cat-survey    survey response classification & extraction
#   cat-vader     social media (Reddit, Twitter/X, forums)
#   cat-ademic    academic papers, PDFs, citations
#   cat-search    web content classification & fact-checking
#   cat-pol       political text (manifestos, speeches, legislation)
#   cat-cog       cognitive assessment & visual scoring (CERAD, drawing tests)
```

**Dependency graph** (modeled on tidyverse / rlang):

```
cat-stack                                      ← general base + shared infra (like rlang)
                                                    independently useful; no domain assumptions
    ↑
cat-survey  cat-vader  cat-ademic  cat-search  cat-pol  cat-cog  ← domain packages, each depends on cat-stack
    ↑           ↑          ↑          ↑        ↑
                         cat-llm                     ← pure meta-package, keeps the brand
                                                      (depends on all domain packages + cat-stack)
```

Every package is independently installable:
```
pip install cat-stack    # general classification engine, no domain framing
pip install cat-survey   # survey-specific, pulls in cat-stack automatically
pip install cat-search   # web classification + fact-checking, pulls in cat-stack
pip install cat-cog      # cognitive assessment, pulls in cat-stack automatically
pip install cat-llm      # everything
```

---

## The Packages

### `cat-stack` *(new — the base)*
General-purpose text column classification and extraction with no domain assumptions.
This replaces the need for a separate invisible `catcore` — it is both the shared
infrastructure layer AND independently useful for researchers who just have a text
column and don't fit neatly into a domain.

**Scope:**
- `classify()`, `extract()`, `explore()`, `summarize()` for text, image, PDF, and URL input
- `UnifiedLLMClient`, `PROVIDER_CONFIG`, `detect_provider` — all provider infrastructure
- `_batch.py` — async batch API logic
- Image and PDF processing: `image_functions.py`, `pdf_functions.py`
- Shared text utilities: `build_json_schema`, `extract_json`, `validate_classification_json`
- No survey framing, no social media framing, no cognitive-test assumptions
- Namespace: `import cat_stack`

**Status:** ✅ Created. Repo at `cat-stack/`, namespace `cat_stack`, v0.1.0. PyPI pending.

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

**Status:** ✅ Created as thin wrapper on cat-stack, v0.1.0. PyPI pending.

---

### `cat-vader` *(already separate)*
Social media text classification with platform-aware context injection.

**Scope:**
- `classify()` / `extract()` / `explore()` with social media metadata fields
  (platform, handle, hashtags, engagement metrics injected into prompt)
- Reddit thread classification (nested comment structure)
- HuggingFace Space: `CatLLM/CatVader`

**Status:** ✅ Refactored to thin wrapper on cat-stack, v1.13.0. PyPI pending.

---

### `cat-ademic` *(already separate)*
Classification and extraction for academic and long-form documents.

**Scope:**
- PDF-first `classify()` / `extract()` — builds on cat-stack's PDF/image processing
- Per-page and whole-document classification modes
- OpenAlex API integration for fetching academic papers by journal, field, or topic
- Citation and abstract extraction
- Designed for systematic reviews, coding codebooks, content analysis of papers

**Status:** ✅ Refactored to thin wrapper on cat-stack, v0.1.1. PyPI pending.

---

### `cat-cog` *(extract from current cat-llm)*
Cognitive assessment and visual scoring — LLM-powered evaluation of drawn images
for neuropsychological testing.

**Scope:**
- CERAD drawing scoring (circle, diamond, cube, clock)
- Trained circle classifier model for shape quality assessment
- Image feature extraction for cognitive test drawings
- Designed for clinical research, cognitive screening studies
- Builds on cat-stack's image classification infrastructure

**Status:** ✅ Created as thin wrapper on cat-stack, v0.1.0. Fine-tuned vision models
deferred to v0.2.0. PyPI pending.

---

### `cat-search` *(refactor from llm-web-research)*
Web content classification and precision fact-checking. Thin wrapper on cat-stack
for URL-based classification, plus standalone web research functions.

**Scope:**
- `classify()` / `extract()` / `explore()` — thin wrappers that pass URLs to
  cat-stack (which handles fetching via its URL input type). Adds web-specific
  context injection (source domain, fetch timestamp, etc.)
- `web_research()` — fast single-step web search + LLM extraction (existing, kept as-is)
- `precise_web_research()` — Funnel of Verification (FoVe) pipeline for high-accuracy
  fact-checking with confidence thresholds (existing, kept as-is)
- `tavily_search()` — Tavily API integration (existing, kept as-is)
- Designed for dataset building at scale

**Status:** Existing code in `llm-web-research` repo. Needs to be renamed to `cat-search`,
add `cat-stack` as dependency, and add thin wrapper `classify()` / `extract()` / `explore()`
that leverage cat-stack's URL input type.

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
cat-stack/                    ← general base + shared infra (like rlang)
├── src/cat_stack/
│   ├── _providers.py           ← UnifiedLLMClient, PROVIDER_CONFIG
│   ├── _batch.py               ← batch API logic
│   ├── text_functions.py       ← shared text utilities
│   ├── text_functions_ensemble.py ← ensemble loop, consensus voting
│   ├── image_functions.py      ← image classification
│   ├── pdf_functions.py        ← PDF page processing
│   ├── classify.py             ← domain-agnostic classify()
│   ├── extract.py              ← domain-agnostic extract()
│   ├── explore.py              ← domain-agnostic explore()
│   └── summarize.py            ← domain-agnostic summarize()

cat-survey/                     ← survey package (depends on cat-stack)
│   survey-tuned classify(), extract(), explore(), summarize()
│   R + Stata wrappers

cat-vader/                      ← social media (depends on cat-stack)
│   classify(), extract(), explore() with social metadata

cat-ademic/                     ← academic/PDF (depends on cat-stack)
│   classify(), extract() for academic papers, OpenAlex integration

cat-cog/                        ← cognitive assessment (depends on cat-stack)
│   CERAD scoring, circle classifier, drawing test evaluation

cat-search/                     ← web research (depends on cat-stack)
│   classify(), extract(), explore() via URL input type
│   web_research(), precise_web_research(), tavily_search()

cat-pol/                        ← political text (depends on cat-stack)
│   classify(), extract() with political science prompt library

cat-llm/                        ← pure meta-package, keeps the brand
│   depends on: cat-stack, cat-survey, cat-vader, cat-ademic, cat-search, cat-cog, cat-pol
│   src/catllm/__init__.py re-exports all sub-package public APIs
│   No domain logic lives here
```

---

## Migration Path

### Phase 1 — Stabilize ✅
- Shipped feature parity, code hygiene, README documentation.

### Phase 2 — Create `cat-stack` ✅
- Forked cat-llm, renamed namespace to `cat_stack`.
- Stripped domain-specific code: CERAD, circle classifier, survey-specific prompt language.
- Kept full input type support: text, image, PDF.
- Neutralized prompt framing (no "respondent" or "survey" language in LLM prompts).
- Publish `cat-stack` to PyPI (pending).

### Phase 3 — Wire domain packages to `cat-stack` ✅
- Updated `cat-vader` to thin wrapper on `cat-stack` (v1.13.0).
- Updated `cat-ademic` to thin wrapper on `cat-stack`, removed CERAD.
- Created `cat-cog` with CERAD scoring as thin wrapper on `cat-stack` (v0.1.0).
- Created `cat-survey` with survey framing as thin wrapper on `cat-stack` (v0.1.0).

### Phase 4 — Slim `cat-llm` to meta-package ✅
- Replaced `src/catllm/` source with thin `__init__.py` re-exporting
  domain-suffixed aliases (classify_survey, classify_social, classify_academic,
  cerad_drawn_score) from all sub-packages.
- Neutral `classify()` / `extract()` / `explore()` / `summarize()` point to `cat-stack`.
- Updated `pyproject.toml` to list sub-packages as dependencies.
- Removed all duplicated source files (all logic lives in sub-packages).
- Cut CERAD from cat-llm (now in cat-cog).
- Version bumped cat-llm to v3.0.0.
- Publish all packages to PyPI (pending).

### Phase 5 — URL input type + `cat-search` + `cat-pol` *(next)*
- Add URL as a new input type to `cat-stack` (alongside text, image, PDF).
  - `_web_fetch.py` — fetch HTML from URLs, extract text content.
  - URL auto-detection in input type router (detect `http://` / `https://` strings).
  - All domain packages inherit URL support for free via `**kwargs` passthrough.
- Rename `llm-web-research` → `cat-search`. Add `cat-stack` dependency.
  - Keep existing functions (`web_research`, `precise_web_research`, `tavily_search`).
  - Add thin wrapper `classify()` / `extract()` / `explore()` that leverage
    cat-stack's URL input type with web-specific context injection.
  - Wire into `cat-llm` as `classify_web()` / `extract_web()` / `web_research()` /
    `precise_web_research()`.
- Wire `cat-pol` into `cat-llm` as `classify_political()` / `extract_political()`.
- Publish and update all packages.

---

## API Design

### Sub-package API
Each domain package exposes the same four verbs with domain-specific parameters:

| Function | Purpose |
|----------|---------|
| `classify()` | Assign predefined categories to documents |
| `extract()` | Discover categories from a corpus |
| `explore()` | Raw extraction for saturation analysis |
| `summarize()` | Summarize documents |

Domain-specific behavior is injected through keyword arguments
(`sm_source=` in cat-vader; `journal_name=` in cat-ademic),
never by changing function names. Learn once, apply anywhere.

### Meta-package API (`cat-llm`)
`cat-llm` re-exports all sub-package functions with domain-suffixed names,
so users only need `import catllm` and can tab-complete to find everything:

```python
import catllm

# Neutral base (from cat-stack)
catllm.classify(...)
catllm.extract(...)
catllm.explore(...)
catllm.summarize(...)

# Survey (from cat-survey)
catllm.classify_survey(...)
catllm.extract_survey(...)
catllm.explore_survey(...)

# Social media (from cat-vader)
catllm.classify_social(...)
catllm.extract_social(...)
catllm.explore_social(...)

# Academic (from cat-ademic)
catllm.classify_academic(...)
catllm.extract_academic(...)
catllm.explore_academic(...)

# Web research (from cat-search)
catllm.classify_web(...)
catllm.extract_web(...)
catllm.explore_web(...)
catllm.web_research(...)
catllm.precise_web_research(...)

# Political text (from cat-pol)
catllm.classify_political(...)
catllm.extract_political(...)
catllm.explore_political(...)

# Cognitive assessment (from cat-cog)
catllm.cerad_drawn_score(...)
```

The `__init__.py` for `cat-llm` is purely aliases:

```python
from cat_stack import classify, extract, explore, summarize
from cat_survey import classify as classify_survey
from cat_survey import extract as extract_survey
from cat_survey import explore as explore_survey
from catvader import classify as classify_social
from catvader import extract as extract_social
from catvader import explore as explore_social
from catademic import classify as classify_academic
from catademic import extract as extract_academic
from catademic import explore as explore_academic
from cat_search import classify as classify_web
from cat_search import extract as extract_web
from cat_search import explore as explore_web
from cat_search import web_research, precise_web_research
from cat_pol import classify as classify_political
from cat_pol import extract as extract_political
from cat_pol import explore as explore_political
from cat_cog import cerad_drawn_score
```

Users who want a lighter install can use sub-packages directly:
```python
pip install cat-survey   # just survey + cat-stack
import cat_survey
cat_survey.classify(...)
```

### `pyproject.toml` for `cat-llm`

```toml
[project]
name = "cat-llm"
dependencies = [
    "cat-stack~=1.0",
    "cat-survey~=1.0",
    "cat-vader~=1.0",
    "cat-ademic~=1.0",
    "cat-search~=1.0",
    "cat-cog~=1.0",
    "cat-pol~=1.0",
]
```

---

## Open Questions

- **Versioning**: Compatible-release (`~=`) constraints so patch updates propagate
  automatically but breaking changes require an explicit `cat-llm` bump.
- **R/Stata wrappers**: Live in `cat-survey` (v0.2.0+). A future R meta-package could
  mirror the Python ecosystem.
- **HuggingFace Spaces**: One Space per domain package (current approach). A unified
  multi-tab Space under `CatLLM/` is a long-term goal once 2+ Spaces exist.
- **`cat-stack` marketing**: Position it as "classify any text, image, or PDF" —
  useful for researchers outside the survey/social-media domains who want the
  LLM pipeline without domain framing.
- **`cat-cog` scope**: Starts with CERAD drawing scoring. Fine-tuned vision models
  (circle classifier, HuggingFace-hosted) come in v0.2.0+. Could expand to other
  cognitive screening instruments (MoCA, MMSE scoring from scanned forms, clock
  drawing tests) as the field adopts LLM-based scoring.
