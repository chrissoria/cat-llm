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

## Shared Infrastructure

All sub-packages depend on a common base layer. Two options:

### Option A — `catcore` (dedicated base package)
Extract `_providers.py`, `_batch.py`, `text_functions.py`, and `build_json_schema` /
`extract_json` utilities into a standalone `catcore` package. Each sub-package lists
`catcore` as a dependency, not `cat-llm`.

**Pros:** Clean separation, `cat-llm` is a true thin meta-package.
**Cons:** Another repo/package to maintain; versioning coordination overhead.

### Option B — `cat-llm` remains the provider layer
`cat-llm` keeps `_providers.py`, `_batch.py`, and shared utilities. Sub-packages list
`cat-llm` as a dependency. `cat-llm` re-exports sub-package entry points.

**Pros:** Fewer repos; provider upgrades propagate automatically.
**Cons:** Circular dependency risk if sub-packages ever need to import each other.

**Current leaning: Option B** — simpler to execute now. Revisit when there are 3+
active sub-packages and provider churn becomes painful.

---

## Target Package Structure

```
cat-llm/                      ← meta-package + shared provider layer
├── src/catllm/
│   ├── _providers.py         ← UnifiedLLMClient, PROVIDER_CONFIG (shared)
│   ├── _batch.py             ← batch API logic (shared)
│   ├── text_functions.py     ← shared text utilities
│   └── __init__.py           ← re-exports from cat-survey, cat-vader, cat-ademic, cat-pol

cat-survey/                   ← survey-specific package
│   classify(), extract(), explore(), summarize()
│   R + Stata wrappers

cat-vader/                    ← social media package (already exists)
│   classify(), extract(), explore() with social metadata

cat-ademic/                   ← academic/PDF package
│   classify(), extract() for PDFs, CERAD scoring

cat-pol/                      ← political text package (future)
│   classify(), extract() with political science prompt library
```

---

## Migration Path

### Phase 1 — Stabilize the core (now)
- Keep everything in `cat-llm` as-is.
- Ensure `_providers.py` and `_batch.py` are the clean shared foundation.
- Continue publishing `cat-llm` as the all-in-one package.
- Wire `cat-vader` as a declared `cat-llm` optional dependency.

### Phase 2 — Extract cat-survey
- Move survey-specific code into a new `cat-survey` repo.
- Publish `cat-survey` to PyPI.
- `cat-llm` adds `cat-survey` as a dependency and re-exports its API.
- `cat-llm` version bump; update README to reflect the new structure.

### Phase 3 — Build cat-ademic
- Seed from existing `pdf_functions.py`, `image_functions.py`, CERAD functions.
- Publish `cat-ademic` to PyPI.
- Add to `cat-llm` dependencies.

### Phase 4 — cat-pol (when ready)
- Develop domain-tuned prompt library for political science.
- Publish and wire in.

---

## API Consistency Across Sub-Packages

All sub-packages expose the same core verbs:

| Function | Purpose |
|----------|---------|
| `classify()` | Assign predefined categories to documents |
| `extract()` | Discover categories from a corpus |
| `explore()` | Raw extraction for saturation analysis |
| `summarize()` | Summarize documents |

Domain-specific behavior is injected through keyword arguments
(e.g., `platform=`, `metadata=` in cat-vader; `page_mode=` in cat-ademic),
not by changing function names. This keeps the learning curve flat across packages.

---

## Open Questions

- **Namespace**: Should sub-packages use `import catsurvey` or `import catllm.survey`?
  The latter is cleaner for users but requires a non-trivial namespace package setup.
- **Versioning**: Lock sub-package versions in `cat-llm`'s `pyproject.toml`, or use
  compatible-release (`~=`) constraints? Compatible-release is safer for stability.
- **R/Stata wrappers**: Stay in `cat-survey` or live in `cat-llm`? Probably `cat-survey`
  since they wrap survey-specific functions.
- **HuggingFace Spaces**: One Space per sub-package (current approach) or a unified
  multi-tab app under `CatLLM/`?
