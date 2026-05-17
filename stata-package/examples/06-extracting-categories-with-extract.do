********************************************************************************
* 06-extracting-categories-with-extract.do
*
* Discover a category scheme with `catllm extract`. Use this when you
* have free-text data and no pre-defined coding scheme -- let the model
* surface the recurring themes for you.
*
* Behind the scenes, extract() chunks your data, runs the discovery
* prompt multiple times (across `iterations`), merges semantically
* similar labels, and returns a ranked list of top categories.
*
* Compare this to example 04, which is end-to-end (extract -> classify).
* This example focuses on the tuning knobs of extract itself.
*
* Requires: OPENAI_API_KEY
* Cost:     ~$0.10 (two extract runs at different specificity)
* Runtime:  ~1 minute
********************************************************************************

clear all
set more off

adopath + "`c(pwd)'/.."

global OPENAI_API_KEY : env OPENAI_API_KEY
if "$OPENAI_API_KEY" == "" {
    di as error "OPENAI_API_KEY not set."
    exit 198
}

* --- Example data: a mixed bag of neighborhood comments ----------------------
input str200 response
"Just had the best coffee ever! Highly recommend the new place downtown."
"The bus was 40 minutes late again. Public transit here is a joke."
"Loved the museum exhibit on California water rights."
"Why is rent so expensive in this neighborhood?"
"Tried the new ramen spot -- totally worth the wait."
"Construction noise on 4th street has been brutal lately."
"The library's new author series is amazing."
"Need to find an affordable place before lease ends."
"Saw a great documentary at the indie theater last night."
"Trash pickup hasn't come in three weeks. Calling 311."
end

* --- 1. Basic extract --------------------------------------------------------
di _n "{hline 60}"
di "1. Basic extract -- broad themes"
di "{hline 60}"

catllm extract response,                                ///
    apikey($OPENAI_API_KEY)                             ///
    model("gpt-4o-mini")                                ///
    maxcategories(8)                                    ///
    iterations(6)

di _n "Returned in r():"
return list

* Individual categories are stored in r(cat1)..r(catN)
di _n "Programmatic access:"
forvalues i = 1/`r(n_categories)' {
    di "  cat`i' = " `"`r(cat`i')'"'
}

* The full quoted list is in r(categories), ready to pipe into classify():
*   local cats = `"`r(categories)'"'
*   catllm classify response, categories(`cats') apikey(...)
*
* See example 04 for the end-to-end pipeline.

* --- 2. Tune the discovery: specific vs broad --------------------------------
* `pyoptions("specificity=...")` controls how granular the themes are.
* "broad" (default) gives general themes; "specific" gives finer-grained ones.
di _n "{hline 60}"
di "2. Specific extract -- finer-grained themes"
di "{hline 60}"

catllm extract response,                                ///
    apikey($OPENAI_API_KEY)                             ///
    model("gpt-4o-mini")                                ///
    maxcategories(12)                                   ///
    iterations(10)                                      ///
    pyoptions("specificity='specific'")

di _n "Specific scheme:"
forvalues i = 1/`r(n_categories)' {
    di "  cat`i' = " `"`r(cat`i')'"'
}

* --- 3. Tuning reference ----------------------------------------------------
*
*  maxcategories     -- Cap on the number of categories returned. Smaller
*                       forces the model to merge similar labels.
*  catsperchunk      -- Categories the model proposes per chunk before
*                       consolidation. Higher = more variety, more API cost.
*  divisions         -- Number of chunks the data is split into per pass.
*                       Higher = better coverage of long datasets.
*  iterations        -- Number of passes. More = more stable but more
*                       expensive. 6-12 is the typical sweet spot.
*  pyoptions("specificity='broad'|'specific'")
*                    -- Granularity. "broad" for high-level themes,
*                       "specific" for finer distinctions.
*  pyoptions("research_question=...")
*                    -- Free-text framing for the discovery prompt.

di _n "Done."
