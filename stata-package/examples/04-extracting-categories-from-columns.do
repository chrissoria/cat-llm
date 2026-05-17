********************************************************************************
* 04-extracting-categories-from-columns.do
*
* End-to-end workflow: take a Stata variable of open-ended text, discover
* a category scheme with `catllm extract`, classify every row against
* those categories, then export the labeled dataset.
*
* Canonical CatLLM workflow when you don't have a pre-defined coding
* scheme:
*   1. `catllm extract` to discover the categories
*   2. (optional) hand-curate the labels so they're verbose
*   3. `catllm classify` to assign every row
*   4. Export to CSV or save as .dta for analysis
*
* Requires: OPENAI_API_KEY
* Cost:     ~$0.10 (one extract + one classify against 20 rows)
* Runtime:  ~1-2 minutes
********************************************************************************

clear all
set more off

adopath + "`c(pwd)'/.."

global OPENAI_API_KEY : env OPENAI_API_KEY
if "$OPENAI_API_KEY" == "" {
    di as error "OPENAI_API_KEY not set."
    exit 198
}

* --- Load data ---------------------------------------------------------------
* Replace this block with your own data. For the example, we inline 20
* responses to a "why did you move?" question.
input str200 open_response
"The weather was just too hot where I lived"
"I could no longer afford the rent in my old neighborhood"
"My company offered me a promotion but it required relocating"
"I wanted to be closer to my aging parents"
"The schools in my old district were underperforming"
"I needed a fresh start after my divorce"
"The cost of living was eating up most of my paycheck"
"I got accepted to a great university here"
"My spouse got a new job in this city"
"The crime rate where I lived was getting worse"
"I wanted access to better healthcare"
"The job market in my field was much better here"
"I retired and wanted somewhere quieter"
"My family kept asking me to move back home"
"The political climate in my old state didn't fit me"
"I wanted to be closer to nature and the outdoors"
"My apartment building was sold and I had to leave"
"I followed my partner who is doing a residency here"
"The public transportation here is much better"
"I always wanted to live in a city this size"
end

di "Loaded `c(N)' responses."
list open_response in 1/5, separator(0) abbreviate(40)

* --- Step 1: Extract a category scheme ---------------------------------------
* domain(survey) routes through cat-survey which builds survey-question
* framing into the prompt. `pyoptions("survey_question=...")` forwards
* the question text to the Python call.
di _n "{hline 60}"
di "Step 1: Discover categories with catllm extract"
di "{hline 60}"

catllm extract open_response,                                          ///
    apikey($OPENAI_API_KEY)                                            ///
    model("gpt-4o-mini")                                               ///
    maxcategories(12)                                                  ///
    iterations(6)                                                      ///
    description("Reasons people gave for moving to a new city")        ///
    pyoptions("survey_question='Why did you move to a new city?'")

return list

* Grab the discovered scheme into a local for later use
local discovered = `"`r(categories)'"'
di _n `"Discovered scheme: `discovered'"'

* --- Step 2: Curate the categories (optional but recommended) ----------------
* Verbose, definition-style labels classify several percentage points more
* accurately than one-word labels. Edit `verbose_cats` to whatever scheme
* you decided to use after reviewing r(categories) from step 1.
di _n "{hline 60}"
di "Step 2: Curate the scheme (verbose labels classify better)"
di "{hline 60}"

local verbose_cats                                                                     ///
    `""Job/school: A change in employment, education, or career, including transfers and retirement." "' ///
    `""Family: Relationship changes, having children, supporting relatives, or relocating to be near family." "' ///
    `""Cost of living: Housing affordability, cost of goods, or general economic pressure." "' ///
    `""Lifestyle: Climate, recreation, cultural fit, or change of scenery." "' ///
    `""Other: The response does not fit any of the above categories.""'

* --- Step 3: Classify every row against the curated scheme -------------------
di _n "{hline 60}"
di "Step 3: Classify every row"
di "{hline 60}"

catllm classify open_response,                                         ///
    categories(`verbose_cats')                                         ///
    apikey($OPENAI_API_KEY)                                            ///
    model("gpt-4o-mini")                                               ///
    generate(category)                                                 ///
    pyoptions("survey_question='Why did you move to a new city?'")

list open_response category, separator(0) abbreviate(20)
tab category

* --- Step 4: Export or save --------------------------------------------------
di _n "{hline 60}"
di "Step 4: Export"
di "{hline 60}"

* Save as a Stata dataset:
* save survey_classified.dta, replace

* Or export to CSV:
* export delimited survey_classified.csv, replace

* Tips:
*  - Run on a subsample first to verify the category scheme makes sense
*    before classifying thousands of rows.
*  - Use ensemble mode (see example 02) for high-stakes coding.
*  - Validate against a hand-labeled subsample for any published research.

di _n "Done."
