# cat.stack 0.1.0

* Initial release of the R interface to the Python `catstack` package.
* Exposes `classify()`, `extract()`, `explore()`, and `summarize()` for
  domain-agnostic LLM-powered text, image, and PDF classification.
* `install_cat_stack()` installs the Python dependency via reticulate.
* Internal helpers (`.strip_quotes()`, `.as_py_int()`, `.convert_models()`,
  `.validate_add_other()`) exported for reuse by sibling domain packages
  (cat.survey, cat.ademic, cat.pol, cat.web, etc.).
