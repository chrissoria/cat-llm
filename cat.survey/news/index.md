# Changelog

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
