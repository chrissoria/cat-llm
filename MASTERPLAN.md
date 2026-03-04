# CatLLM Ecosystem — Master Plan

## Vision

**cat-llm** evolves from a monolithic package into a meta-package — the tidyverse of
LLM-powered social science text analysis. Installing `cat-llm` pulls in a family of
domain-specific sub-packages, each with its own PyPI release, versioning, and focus,
but sharing a common provider infrastructure and a consistent API surface.

```
pip install cat-llm
# brings in:
#   cat-survey    survey response classification & extraction
#   cat-vader     social media (Reddit, Twitter/X, forums)
#   cat-ademic    academic papers, PDFs, citations
#   cat-pol       political text (manifestos, speeches, legislation)
```

---

## The Sub-Packages

### cat-survey *(extract from current cat-llm core)*
Survey response classification and category extraction.
This is the current heart of `cat-llm` — it gets spun out as its own package.

**Scope:**
- `classify()` for open-ended survey responses (text only)
- `extract()` / `explore()` for category discovery
- `summarize()` for response-level summarization
- All prompt strategies tested in the paper (CoT, CoVe, step-back, few-shot)
- R and Stata wrappers live here

**Status:** Exists inside `cat-llm/src/catllm/`. Needs to be extracted into its own repo
(`cat-survey`) and published separately, then re-imported by `cat-llm`.

---

### cat-vader *(already separate)*
Social media text classification with platform-aware context injection.

**Scope:**
- `classify()` / `extract()` / `explore()` with social media metadata fields
  (platform, handle, hashtags, engagement metrics injected into prompt)
- Reddit thread classification (nested comment structure)
- HuggingFace Space: `CatLLM/CatVader`

**Status:** Separate repo and PyPI package. Needs to be wired into `cat-llm` as a
declared dependency and re-exported.

---

### cat-ademic *(planned)*
Classification and extraction for academic and long-form documents.

**Scope:**
- PDF-first classify/extract — builds on current `pdf_functions.py` logic
- Per-page and whole-document classification modes
- Citation and abstract extraction
- CERAD scoring (currently in `cat-llm` — natural home here given image + PDF focus)
- Designed for systematic reviews, coding codebooks, content analysis of papers

**Status:** Not yet started. Core PDF machinery exists in `cat-llm/src/catllm/pdf_functions.py`
and `image_functions.py` — these become the seed.

---

### cat-pol *(future)*
Political text analysis — manifestos, speeches, legislation, news.

**Scope:**
- Domain-tuned prompts for political science categories
  (ideology, policy area, sentiment toward government, actor identification)
- Multi-language support (EU parliament, UN documents)
- Manifesto coding scheme compatibility (MARPOR/CMP)
- Time-series classification across document corpora

**Status:** Concept only. No code yet.

---

## Shared Infrastructure — `catcore`

Modeled on how tidyverse handles shared infrastructure (e.g., `rlang`, `cli`):
a dedicated base package that sub-packages depend on directly, with the meta-package
(`cat-llm`) knowing about sub-packages but sub-packages having no knowledge of the
meta-package.

**`catcore`** contains:
- `_providers.py` — `UnifiedLLMClient`, `PROVIDER_CONFIG`, `detect_provider`, Ollama utils
- `_batch.py` — async batch API logic for all providers
- `text_functions.py` — `build_json_schema`, `extract_json`, `validate_classification_json`

Each sub-package lists `catcore` as a dependency. `cat-llm` lists all sub-packages
as dependencies and is otherwise empty — just an `__init__.py` that imports and
re-exports everything. This mirrors exactly how `tidyverse` works in R.

**Dependency graph:**

```
catcore                        ← no cat-* deps (leaf)
    ↑
cat-survey  cat-vader  cat-ademic  cat-pol    ← each depends only on catcore
    ↑           ↑          ↑          ↑
                    cat-llm                    ← pure meta-package
                                               (depends on all four + catcore)
```

This means every sub-package is independently installable:
```
pip install cat-survey   # works standalone, no cat-llm required
pip install cat-llm      # installs everything
```

---

## Target Package Structure

```
catcore/                      ← shared infrastructure (like rlang in R)
├── src/catcore/
│   ├── _providers.py         ← UnifiedLLMClient, PROVIDER_CONFIG
│   ├── _batch.py             ← batch API logic
│   └── text_functions.py     ← shared text utilities

cat-survey/                   ← survey response package
│   depends on: catcore
│   classify(), extract(), explore(), summarize()
│   R + Stata wrappers

cat-vader/                    ← social media package (already exists)
│   depends on: catcore
│   classify(), extract(), explore() with social metadata

cat-ademic/                   ← academic/PDF package (planned)
│   depends on: catcore
│   classify(), extract() for PDFs, CERAD scoring

cat-pol/                      ← political text package (future)
│   depends on: catcore
│   classify(), extract() with political science prompt library

cat-llm/                      ← pure meta-package (like tidyverse in R)
│   depends on: cat-survey, cat-vader, cat-ademic, cat-pol, catcore
│   src/catllm/__init__.py re-exports all sub-package public APIs
│   No domain logic lives here
```

---

## Migration Path

### Phase 1 — Stabilize the core (now)
- Keep everything in `cat-llm` as-is; ship features normally.
- Identify and document the exact boundary between `catcore` code and
  survey-specific code — this is the prerequisite for Phase 2.
- Continue publishing `cat-llm` as the all-in-one package.

### Phase 2 — Create `catcore`
- Extract `_providers.py`, `_batch.py`, and shared text utilities into a new `catcore` repo.
- Publish `catcore` to PyPI.
- Update `cat-llm` and `cat-vader` to depend on `catcore` instead of carrying their own copies.

### Phase 3 — Extract `cat-survey`
- Move survey-specific code (`classify`, `extract`, `explore`, `summarize`, R/Stata wrappers)
  into a new `cat-survey` repo depending on `catcore`.
- Publish `cat-survey` to PyPI independently.
- Slim `cat-llm` down to a meta-package: `pyproject.toml` lists sub-packages as dependencies,
  `__init__.py` re-exports their public APIs.
- Version bump; update README.

### Phase 4 — Build `cat-ademic`
- Seed from existing `pdf_functions.py`, `image_functions.py`, CERAD functions in cat-llm.
- Publish `cat-ademic` to PyPI; add to `cat-llm` dependencies.

### Phase 5 — `cat-pol` (when ready)
- Develop domain-tuned prompt library for political science.
- Publish and wire into `cat-llm`.

---

## API Consistency Across Sub-Packages

Modeled on tidyverse's verb consistency (every package uses `filter`, `select`, `mutate` etc.):
all cat-* packages expose the same four core functions.

| Function | Purpose |
|----------|---------|
| `classify()` | Assign predefined categories to documents |
| `extract()` | Discover categories from a corpus |
| `explore()` | Raw extraction for saturation analysis |
| `summarize()` | Summarize documents |

Domain-specific behavior is injected through keyword arguments
(e.g., `platform=`, `metadata=` in cat-vader; `page_mode=` in cat-ademic),
not by changing function names. This keeps the learning curve flat: learn once, apply anywhere.

The `pyproject.toml` for `cat-llm` will look like:

```toml
[project]
name = "cat-llm"
dependencies = [
    "catcore>=1.0",
    "cat-survey>=1.0",
    "cat-vader>=1.0",
    "cat-ademic>=1.0",
]
```

---

## Open Questions

- **Namespace**: Sub-packages use flat names (`import catsurvey`, `import catvader`) matching
  the tidyverse pattern (`library(dplyr)`, not `library(tidyverse::dplyr)`). When installed via
  `cat-llm`, they are still importable by their own names.
- **Versioning**: Use compatible-release (`~=`) constraints in `cat-llm`'s dependencies
  (e.g., `cat-survey~=1.0`) so patch updates propagate automatically but breaking changes
  require an explicit `cat-llm` bump. Same approach tidyverse uses.
- **R/Stata wrappers**: Live in `cat-survey` — they wrap survey-specific functions.
  A future `cat-llm` R meta-package could re-export them.
- **HuggingFace Spaces**: Keep one Space per sub-package (current approach for cat-vader).
  A unified multi-tab Space under `CatLLM/` is a nice long-term goal once 2+ sub-packages
  have Spaces.
