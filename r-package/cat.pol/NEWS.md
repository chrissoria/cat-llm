# cat.pol 0.1.0

* Initial release of the R interface to the Python `catpol` package.
* Domain wrapper around `cat.stack` that adds a registered-source fetcher
  (city ordinances, federal laws, executive orders, presidential speeches,
  social-media archives) and policy-document prompt framing.
* Exports `classify()`, `extract()`, `explore()`, `summarize()`, and
  `list_sources()`.
