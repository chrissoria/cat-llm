# CatLLM Module Dependency Map

High-level view of how `src/catllm/` modules depend on each other.

```
                         __init__.py
                    (public API re-exports)
                 /     |       |       \
                v      v       v        v
          classify  extract  explore  summarize
            |   \      |       |         |
            |    \     |       |         |
            v     v    v       v         v
   text_functions_ensemble  text_functions
        |        |              |    ^
        |        |              |    |
        v        v              v    |
  pdf_functions  image_functions  _providers
        |              |        (single source of truth:
        |              |         UnifiedLLMClient,
        v              v         PROVIDER_CONFIG,
   calls/pdf_*    calls/image_*  detect_provider,
                                 Ollama utilities)
                     calls/
              (stepback, CoVe, top_n)
              leaf modules, no intra-pkg deps
```

## Key relationships

| Module | Role | Imports from |
|--------|------|-------------|
| `_providers.py` | Provider infrastructure (UnifiedLLMClient, PROVIDER_CONFIG, detect_provider, Ollama utils) | *nothing* (leaf) |
| `text_functions.py` | Single-model classification + re-exports provider infra | `_providers`, `calls/*` |
| `text_functions_ensemble.py` | Multi-model ensemble logic | `text_functions`, `pdf_functions`, `image_functions` |
| `classify.py` | Unified `classify()` entry point | `_providers`, `text_functions_ensemble`, `text_functions`, `image_functions`, `pdf_functions` |
| `extract.py` | Category exploration entry point | `_providers`, `text_functions`, `image_functions`, `pdf_functions` |
| `summarize.py` | Summarization entry point | `_providers`, `text_functions_ensemble` |
| `explore.py` | Thin wrapper for `explore_common_categories` | `text_functions` |
| `pdf_functions.py` | PDF classification | `text_functions`, `calls/pdf_*` |
| `image_functions.py` | Image classification | `text_functions`, `calls/image_*` |
| `calls/*` | Provider-specific prompt logic (stepback, CoVe, top_n) | *nothing* (leaf modules) |

## `classify()` call chain

`classify()` is a thin wrapper — it delegates everything to `classify_ensemble()`.
Even single-model calls go through the ensemble path (with a 1-model list).

```
classify()                                          classify.py
  └─ classify_ensemble()                            text_functions_ensemble.py
       │
       ├─ prepare_model_configs()                   text_functions_ensemble.py
       │    └─ detect_provider()                    _providers.py (via text_functions re-export)
       │    └─ check_ollama_running/model/pull()    _providers.py (via text_functions re-export)
       │
       ├─ [if categories="auto"]
       │    └─ extract()                            main.py
       │         └─ explore_common_categories()     text_functions.py
       │              └─ UnifiedLLMClient.complete() _providers.py
       │
       ├─ prepare_json_schemas()                     text_functions_ensemble.py
       │    └─ build_json_schema()                   text_functions.py
       │
       ├─ [if step_back_prompt]
       │    └─ gather_stepback_insights()           text_functions_ensemble.py
       │         └─ _get_stepback_insight()         text_functions.py
       │              └─ get_stepback_insight_*()   calls/stepback.py
       │
       ├─ classify_single() [per model, per item]   text_functions_ensemble.py (inner function)
       │    │
       │    ├─ UnifiedLLMClient(provider, key, model)   _providers.py
       │    │
       │    ├─ [text input]
       │    │    ├─ [if ollama] ollama_two_step_classify()       text_functions.py
       │    │    │    └─ client.complete() x2                    _providers.py
       │    │    │    └─ extract_json() + validate_json()        text_functions.py
       │    │    │
       │    │    └─ [else] build_text_classification_prompt()    text_functions_ensemble.py
       │    │         └─ client.complete()                       _providers.py
       │    │         └─ extract_json()                          text_functions.py
       │    │         └─ [if CoVe] build_cove_prompts()          text_functions_ensemble.py
       │    │              └─ run_chain_of_verification()        text_functions_ensemble.py
       │    │                   └─ client.complete() x3-6        _providers.py
       │    │                   └─ extract_json()                text_functions.py
       │    │
       │    ├─ [pdf input]
       │    │    └─ _prepare_page_data()                         text_functions_ensemble.py
       │    │         └─ _extract_page_as_image_bytes()          pdf_functions.py
       │    │         └─ _extract_page_text()                    pdf_functions.py
       │    │    └─ build_pdf_classification_prompt()            text_functions_ensemble.py
       │    │    └─ client.complete()                            _providers.py
       │    │    └─ extract_json()                               text_functions.py
       │    │
       │    └─ [image input]
       │         └─ _prepare_image_data()                        text_functions_ensemble.py
       │              └─ _encode_image()                         image_functions.py
       │         └─ build_image_classification_prompt()          text_functions_ensemble.py
       │         └─ client.complete()                            _providers.py
       │         └─ extract_json()                               text_functions.py
       │
       └─ build_output_dataframes()                 text_functions_ensemble.py
            └─ consensus/agreement calculations
            └─ CSV output
```

### Where each strategy comes from

| Strategy | Orchestration | Provider-specific prompts | API calls via |
|----------|--------------|--------------------------|---------------|
| **Standard classify** | `build_text_classification_prompt()` in `text_functions_ensemble.py` | N/A (prompt built inline) | `UnifiedLLMClient.complete()` in `_providers.py` |
| **Chain of Thought** | prompt flag in `build_text_classification_prompt()` | N/A (added to prompt text) | same |
| **Step-back prompting** | `gather_stepback_insights()` in `text_functions_ensemble.py` | `get_stepback_insight_*()` in `calls/stepback.py` | same |
| **Chain of Verification** | `build_cove_prompts()` + `run_chain_of_verification()` in `text_functions_ensemble.py` | N/A (4-step prompts built inline) | same |
| **Ollama two-step** | `ollama_two_step_classify()` in `text_functions.py` | N/A | same |
| **Context prompt** | prompt flag in `build_text_classification_prompt()` | N/A (expert prefix added) | same |
| **Thinking/reasoning** | `thinking_budget` param passed through to `client.complete()` | handled per-provider inside `UnifiedLLMClient._build_payload()` in `_providers.py` | same |

## Important notes

- **`_providers.py`** is the single source of truth for all provider infrastructure. `text_functions.py` re-exports its names so existing importers don't need to change.
- **`calls/`** submodules are leaf nodes with only external dependencies (requests, json). They contain provider-specific prompt templates.
- **`text_functions_ensemble.py`** is the heaviest module (~2700 lines) and pulls from `text_functions`, `pdf_functions`, and `image_functions` to support multi-modal ensemble classification.
