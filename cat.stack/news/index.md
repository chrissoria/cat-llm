# Changelog

## cat.stack 0.1.0

- Initial release of the R interface to the Python `catstack` package.
- Exposes
  [`classify()`](https://christophersoria.com/cat-llm/cat.stack/reference/classify.md),
  [`extract()`](https://christophersoria.com/cat-llm/cat.stack/reference/extract.md),
  [`explore()`](https://christophersoria.com/cat-llm/cat.stack/reference/explore.md),
  and
  [`summarize()`](https://christophersoria.com/cat-llm/cat.stack/reference/summarize.md)
  for domain-agnostic LLM-powered text, image, and PDF classification.
- [`install_cat_stack()`](https://christophersoria.com/cat-llm/cat.stack/reference/install_cat_stack.md)
  installs the Python dependency via reticulate.
- Internal helpers
  ([`.strip_quotes()`](https://christophersoria.com/cat-llm/cat.stack/reference/dot-strip_quotes.md),
  [`.as_py_int()`](https://christophersoria.com/cat-llm/cat.stack/reference/dot-as_py_int.md),
  [`.convert_models()`](https://christophersoria.com/cat-llm/cat.stack/reference/dot-convert_models.md),
  [`.validate_add_other()`](https://christophersoria.com/cat-llm/cat.stack/reference/dot-validate_add_other.md))
  exported for reuse by sibling domain packages (cat.survey, cat.ademic,
  cat.pol, cat.web, etc.).
