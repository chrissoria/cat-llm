# cat.stack 0.2.1

## New parameter in `classify()`

* `multi_label` — logical, default `TRUE` (unchanged behavior). The Python
  engine has always supported single-label classification, but the R
  wrapper did not forward the flag, so `classify()` could only ever request
  multi-label output. Setting `multi_label = FALSE` now reaches the engine
  and switches the prompt to "assign the single most appropriate category"
  (one `1`, the rest `0`), for mutually exclusive coding frames.

# cat.stack 0.2.0

Tracks the Python `cat-stack` 1.6.0 release. The version floor in
`install_cat_stack()` is now `cat-stack >= 1.6.0` — older Python installs
will hit `unexpected keyword argument` errors when R forwards the new
parameters listed below.

## New parameters in `classify()`

* `batch_mode`, `batch_poll_interval`, `batch_timeout` — opt into the
  async batch APIs (OpenAI / Anthropic / Google / Mistral / xAI) for
  ~50% cost savings and higher rate limits. HuggingFace / Perplexity /
  Ollama fall back to synchronous calls.
* `json_retries` — per-row retries when the LLM returns JSON that
  fails schema validation. On the final attempt the formatter
  fallback fires (if `json_formatter` is enabled).
* `json_formatter` — three-state (`TRUE` / `FALSE` / `NULL`) control
  for the local JSON-repair model. Default `NULL` triggers an
  interactive consent prompt on the first malformed row; non-TTY
  contexts decline silently. Requires `cat-stack[formatter]`.
* `two_step_classify` — split classification into reasoning + JSON
  formatting steps for weaker models (lower-tier API + local Ollama).
  Auto-enables `json_formatter` when set.
* `embedding_tiebreaker` + `min_centroid_size` — resolve true
  ensemble 50/50 ties via embedding-centroid similarity instead of
  the default "tie → 0". Adds `category_N_resolved_by` audit column
  (`"vote"` or `"centroid"`). Multi-model ensemble + text input only;
  not yet supported in `batch_mode`. Requires `cat-stack[embeddings]`.

## Behavior changes

* **`consensus_threshold = "majority"` is now strict majority.** A
  50/50 tie on an even-model ensemble (2-2 of 4, 3-3 of 6, 1-1 of 2)
  resolves to `"0"`, not `"1"`. Matches sklearn's `VotingClassifier`
  default and standard ensemble literature. Numeric thresholds
  (e.g. `consensus_threshold = 0.5`) keep `>=` semantics — the user
  picked a number, they get the literal interpretation. For 2-model
  ensembles, `"majority"` now effectively requires both models to
  agree on positive. Use 3+ models for a non-degenerate majority
  vote, or pair with `embedding_tiebreaker = TRUE`.
* `batch_retries` default lowered from `2L` to `1L` to match the
  Python default (changed in `cat-stack` 1.4.1).

## Documentation

* `consensus_threshold` man page now describes the strict-majority
  semantics, the 2-model degeneracy, the numeric-input escape hatch,
  and the `embedding_tiebreaker` companion.
* Added a new example block showing strict-majority + tiebreaker and
  `batch_mode = TRUE`.

## Python-side fixes that flow through automatically

These are in Python `cat-stack` 1.6.0 and pass through the reticulate
bridge without R-side changes — listed here so users know the
behaviors they're getting after `install_cat_stack(upgrade = TRUE)`:

* Google preflight 400 fixed (`additionalProperties` stripped before
  reaching `responseSchema`).
* Per-model batch-job failure isolation — one model's batch failure
  no longer kills the ensemble run.
* Anthropic batch terminal-state inspection — all-errored batches
  raise instead of silently returning empty results.
* `system_prompt` is no longer silently dropped in `batch_mode`.
* PDF summary synthesis grounds on actual page text instead of the
  page label.
* HuggingFace small-model strip-on-5xx (Llama-3.2-1B / `response_format`).
* Image directory loading is case-insensitive; large images warn.
* `prompt_tune` returns `system_prompt = ""` when no improvement was
  found (instead of returning a non-improving prompt as the default).

# cat.stack 0.1.0

* Initial release of the R interface to the Python `catstack` package.
* Exposes `classify()`, `extract()`, `explore()`, and `summarize()` for
  domain-agnostic LLM-powered text, image, and PDF classification.
* `install_cat_stack()` installs the Python dependency via reticulate.
* Internal helpers (`.strip_quotes()`, `.as_py_int()`, `.convert_models()`,
  `.validate_add_other()`) exported for reuse by sibling domain packages
  (cat.survey, cat.ademic, cat.pol, cat.web, etc.).
