********************************************************************************
* 08-extracting-and-exploring-with-ollama.do
*
* Run `catllm extract` and `catllm explore` entirely on a local Ollama
* model -- no API keys, no cloud costs, full data privacy.
*
* What this script covers:
*   1. catllm extract  -- normalized, deduplicated category list
*   2. catllm explore  -- raw category frequency for saturation analysis
*   3. Saturation check -- do new iterations still add new categories?
*   4. classify()      -- apply discovered (then curated) categories
*
* Mirrors r-package/examples/08-extracting-and-exploring-with-ollama.Rmd
*
* Prereqs (one-time):
*   ollama pull llama3.2      # ~2.0 GB
*   ollama pull qwen2.5:7b   # ~4.7 GB  (optional -- used in classify step)
*
* Cost:    $0 (entirely local)
* Runtime: ~2-5 minutes on Apple Silicon depending on model and iterations
********************************************************************************

clear all
set more off

adopath + "`c(pwd)'/.."

local MODEL "llama3.2"
local SURVEY_Q "Why did you move to your current residence?"

* --- Example data ------------------------------------------------------------
input str200 response
"I moved to be closer to family."
"Better job opportunities in this city."
"Lower cost of living."
"Wanted a warmer climate."
"My partner got into graduate school here."
"I was priced out of my old neighborhood."
"Looking for a fresh start after my divorce."
"Company relocated our office."
"Needed more space for a home office."
"Better schools for the kids."
"I retired and wanted to be near the beach."
"Cheaper rent and more outdoor activities."
"Transferred to a new branch."
"Family illness -- needed to be nearby."
"Just wanted to live somewhere more walkable."
end

* --- 1. extract() -- normalized, deduplicated category list ------------------
* extract() runs multiple chunked iterations, semantically merges the
* results, and returns a ranked, deduplicated list. Use this as the first
* step before classify().
di _n "{hline 60}"
di "Step 1: catllm extract -- discover and deduplicate categories"
di "{hline 60}"

catllm extract response,                    ///
    apikey("_")                             ///
    model("`MODEL'")                        ///
    provider("ollama")                      ///
    researchquestion("`SURVEY_Q'")          ///
    maxcategories(10)                       ///
    iterations(3)

di _n "Top categories returned by extract():"
local n_cats = `r(n_categories)'
forvalues i = 1/`n_cats' {
    di "  `i'. `r(cat`i')'"
}

* Save the discovered scheme to classify in step 4
local discovered_n `r(n_categories)'
forvalues i = 1/`discovered_n' {
    local discovered_cat`i' `"`r(cat`i')'"'
}

* --- 2. explore() -- raw category list for saturation analysis ---------------
* explore() returns every category string from every chunk across every
* iteration -- duplicates intact. Use this to check whether your category
* space is saturating (i.e. new iterations stop producing new categories).
di _n "{hline 60}"
di "Step 2: catllm explore -- raw category frequency"
di "{hline 60}"

catllm explore response,                    ///
    apikey("_")                             ///
    model("`MODEL'")                        ///
    provider("ollama")                      ///
    researchquestion("`SURVEY_Q'")          ///
    iterations(3)

di _n "Total raw extractions: `r(n_raw)'"
di "Unique categories:     `r(n_unique)'"

di _n "Sample raw categories (up to 20):"
local show = min(`r(n_raw)', 20)
forvalues i = 1/`show' {
    di "  - `r(cat`i')'"
}

* --- 3. Saturation note -------------------------------------------------------
* New unique categories should decrease with each iteration. If iteration 3
* still introduces as many new categories as iteration 1, add more
* iterations (increase the iterations() option).
di _n "Note: compare n_unique across iterations to check saturation."
di "If iteration 3 still adds many new categories, increase iterations()."

* --- 4. classify() with curated verbose labels --------------------------------
* Pass extract()'s top categories directly to classify(), or rewrite them
* as verbose definitions first (verbose labels improve accuracy on small
* local models).
di _n "{hline 60}"
di "Step 3: classify with curated verbose labels"
di "{hline 60}"

catllm classify response,                                                                          ///
    categories(                                                                                    ///
        "Employment/Career: Moving for a new job, promotion, transfer, or better career prospects." ///
        "Family/Relationships: Relocating to be near family, following a partner, or after a major life event." ///
        "Cost/Affordability: Driven by housing costs, rent increases, or seeking a cheaper area."  ///
        "Lifestyle/Environment: Seeking a preferred climate, outdoor access, walkability, or quality of life." ///
        "Education: Moving for better schools for children, or a partner's graduate program."      ///
        "Other: The response does not fit any of the above categories.")                           ///
    apikey("_")                                                                                    ///
    model("qwen2.5:7b")                                                                            ///
    provider("ollama")                                                                             ///
    surveyquestion("`SURVEY_Q'")                                                                   ///
    generate(move_reason)

list response move_reason, separator(0) abbreviate(40)
tab move_reason

di _n "Done."
