# Changelog

## cat.survey 0.1.5

- [`classify()`](https://christophersoria.com/cat-llm/cat.survey/reference/classify.md)
  now takes `...` and forwards it to the Python engine, so arguments the
  R wrapper does not name (including diagnostic ones such as
  `test = TRUE`, which needs cat-stack \>= 2.6.6) reach
  `catstack.classify()`.

## cat.survey 0.1.4

- [`classify()`](https://christophersoria.com/cat-llm/cat.survey/reference/classify.md)
  now forwards `json_formatter` to the Python engine, so R users can
  turn on the local JSON-repair fallback (it runs only on replies the
  engine cannot use, never on usable or sparse ones).

## cat.survey 0.1.3

- [`classify()`](https://christophersoria.com/cat-llm/cat.survey/reference/classify.md)
  now forwards `batch_mode`, `batch_poll_interval`, `batch_timeout`, and
  `system_prompt` to the Python engine, matching
  [`cat.stack::classify()`](https://christophersoria.com/cat-llm/cat.stack/reference/classify.html).
  Previously these could only be reached by calling the Python function
  directly.

## cat.survey 0.1.0

- Initial release of the R interface to the Python `catsurvey` package.
- Thin domain wrapper around `cat.stack` that injects survey-specific
  prompt framing (“A respondent was asked: …”) into classification,
  extraction, and exploration calls.
- Exports
  [`classify()`](https://christophersoria.com/cat-llm/cat.survey/reference/classify.md),
  [`extract()`](https://christophersoria.com/cat-llm/cat.survey/reference/extract.md),
  and
  [`explore()`](https://christophersoria.com/cat-llm/cat.survey/reference/explore.md).
