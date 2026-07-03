# Changelog

## cat.pol 0.1.0

- Initial release of the R interface to the Python `catpol` package.
- Domain wrapper around `cat.stack` that adds a registered-source
  fetcher (city ordinances, federal laws, executive orders, presidential
  speeches, social-media archives) and policy-document prompt framing.
- Exports
  [`classify()`](https://christophersoria.com/cat-llm/cat.pol/reference/classify.md),
  [`extract()`](https://christophersoria.com/cat-llm/cat.pol/reference/extract.md),
  [`explore()`](https://christophersoria.com/cat-llm/cat.pol/reference/explore.md),
  [`summarize()`](https://christophersoria.com/cat-llm/cat.pol/reference/summarize.md),
  and
  [`list_sources()`](https://christophersoria.com/cat-llm/cat.pol/reference/list_sources.md).
