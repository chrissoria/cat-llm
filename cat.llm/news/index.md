# Changelog

## cat.llm 3.1.0

- Added `cat.pol` and `cat.web` to the meta-package; full ecosystem is
  now 7 sub-packages (cat.stack base + 6 domain packages).
- New domain-suffixed aliases:
  [`classify_political()`](https://christophersoria.com/cat-llm/cat.llm/reference/catllm-aliases.md),
  [`extract_political()`](https://christophersoria.com/cat-llm/cat.llm/reference/catllm-aliases.md),
  [`explore_political()`](https://christophersoria.com/cat-llm/cat.llm/reference/catllm-aliases.md),
  [`summarize_political()`](https://christophersoria.com/cat-llm/cat.llm/reference/catllm-aliases.md),
  [`classify_web()`](https://christophersoria.com/cat-llm/cat.llm/reference/catllm-aliases.md),
  [`extract_web()`](https://christophersoria.com/cat-llm/cat.llm/reference/catllm-aliases.md),
  [`explore_web()`](https://christophersoria.com/cat-llm/cat.llm/reference/catllm-aliases.md),
  [`summarize_web()`](https://christophersoria.com/cat-llm/cat.llm/reference/catllm-aliases.md).
- Startup message now lists all 7 sub-packages.

## cat.llm 3.0.0

- Initial meta-package release. Installs and loads the full CatLLM
  ecosystem (cat.stack, cat.survey, cat.vader, cat.ademic, cat.cog).
- Provides domain-suffixed aliases
  ([`classify_survey()`](https://christophersoria.com/cat-llm/cat.llm/reference/catllm-aliases.md),
  [`classify_social()`](https://christophersoria.com/cat-llm/cat.llm/reference/catllm-aliases.md),
  [`classify_academic()`](https://christophersoria.com/cat-llm/cat.llm/reference/catllm-aliases.md),
  [`cerad_drawn_score()`](https://christophersoria.com/cat-llm/cat.llm/reference/catllm-aliases.md))
  re-exported from each sub-package.
