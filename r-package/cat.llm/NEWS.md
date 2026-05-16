# cat.llm 3.1.0

* Added `cat.pol` and `cat.web` to the meta-package; full ecosystem is
  now 7 sub-packages (cat.stack base + 6 domain packages).
* New domain-suffixed aliases: `classify_political()`, `extract_political()`,
  `explore_political()`, `summarize_political()`, `classify_web()`,
  `extract_web()`, `explore_web()`, `summarize_web()`.
* Startup message now lists all 7 sub-packages.

# cat.llm 3.0.0

* Initial meta-package release. Installs and loads the full CatLLM
  ecosystem (cat.stack, cat.survey, cat.vader, cat.ademic, cat.cog).
* Provides domain-suffixed aliases (`classify_survey()`, `classify_social()`,
  `classify_academic()`, `cerad_drawn_score()`) re-exported from each
  sub-package.
