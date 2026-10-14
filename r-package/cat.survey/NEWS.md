# cat.survey 0.1.4

* `classify()` now forwards `json_formatter` to the Python engine, so R users
  can turn on the local JSON-repair fallback (it runs only on replies the
  engine cannot use, never on usable or sparse ones).

# cat.survey 0.1.3

* `classify()` now forwards `batch_mode`, `batch_poll_interval`,
  `batch_timeout`, and `system_prompt` to the Python engine, matching
  `cat.stack::classify()`. Previously these could only be reached by calling
  the Python function directly.

# cat.survey 0.1.0

* Initial release of the R interface to the Python `catsurvey` package.
* Thin domain wrapper around `cat.stack` that injects survey-specific
  prompt framing ("A respondent was asked: ...") into classification,
  extraction, and exploration calls.
* Exports `classify()`, `extract()`, and `explore()`.
