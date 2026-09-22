********************************************************************************
* catllm_stata_conference.do
*
* Companion do-file for the poster:
*
*     "Coding Open-Ended Survey Text Without Leaving Stata:
*      the catllm Package"
*     Chris Soria, UC Berkeley
*
* Everything on the poster is reproduced here, in order:
*
*     PART 0  Preflight -- is the Python backend installed?
*     PART 1  The data (embedded -- no downloads, no file paths)
*     PART 2  The core pipeline: classify -> indicator variables -> tab/regress
*     PART 3  Provider comparison: same categories, three vendors
*     PART 4  Ensemble + consensus voting, with per-model agreement scores
*     PART 5  Category discovery when you have no coding scheme
*     PART 6  The zero-cost local route (Ollama)
*
* HOW TO RUN
*   Interactive:  do catllm_stata_conference.do
*   Batch:        stata-se -b do catllm_stata_conference.do
*
* Parts 3, 4 and 6 are guarded: each is skipped, with a message, if the
* key or local model it needs isn't available. So the file runs start to
* finish with only one API key -- or with none at all if you have Ollama.
*
* REQUIREMENTS
*   Stata 16+ with Python integration   (check: python query)
*   Python 3.8+
*   One API key, or a local Ollama install
*
* COST
*   The whole file classifies 24 responses a handful of times. With
*   gpt-4o-mini and the other small models used here, that is a few US
*   cents end to end. Part 6 is free.
*
* Package:  https://github.com/chrissoria/cat-llm
* Contact:  chrissoria@berkeley.edu
********************************************************************************

clear all
set more off
version 16

********************************************************************************
* PART 0 -- Preflight
********************************************************************************

di _n(2) "{hline 78}"
di "PART 0 -- Preflight"
di "{hline 78}"

* catllm is a thin Stata layer over the cat-stack Python package. Every
* option you pass is shuttled to Python; nothing about your data leaves
* the Stata session except the text you are classifying.
*
* First time only, run these two lines (uncommented):
*
*     net install catllm, from("https://raw.githubusercontent.com/chrissoria/cat-llm/main/stata-package/") replace
*     catllm setup
*
* After the SSC release:  ssc install catllm

capture which catllm
if _rc {
    di as error "catllm is not installed. Run the net install line above, then re-run."
    exit 111
}

* `catllm setup, check` probes what the Python side actually has.
capture noisily catllm setup, check

* Keys are read from the environment so nothing secret is ever typed into
* -- or saved in -- a do-file. Export them in your shell before starting
* Stata, or set them in profile.do.
global OPENAI_API_KEY    : env OPENAI_API_KEY
global ANTHROPIC_API_KEY : env ANTHROPIC_API_KEY
global GOOGLE_API_KEY    : env GOOGLE_API_KEY

if "$OPENAI_API_KEY" == "" {
    di as error "OPENAI_API_KEY is not set. Parts 2-5 need it; Part 6 does not."
}

********************************************************************************
* PART 1 -- The data
********************************************************************************

di _n(2) "{hline 78}"
di "PART 1 -- The data"
di "{hline 78}"

* A classic open-ended survey item: "Why did you move?" -- free text, no
* coding scheme, exactly the kind of variable that usually gets dropped
* from the analysis because coding it by hand is too expensive.
*
* Embedded inline so this file is self-contained. Swap in your own data
* by replacing this block with a -use- and renaming your text variable
* to `response`.

input str120 response int age byte female
"The weather was just too hot where I lived"                          61 1
"I could no longer afford the rent in my old neighborhood"            34 0
"My company offered me a promotion but it required relocating"        42 0
"I wanted to be closer to my aging parents"                           55 1
"Got divorced and needed a fresh start somewhere new"                 47 1
"The schools in our old district were not good for the kids"          38 1
"Job market collapsed after the layoffs at the plant"                 52 0
"We bought our first house, finally"                                  31 0
"My landlord raised the rent by four hundred dollars"                 29 1
"Moved for graduate school"                                           24 0
"Wanted to be closer to the grandkids"                                68 1
"The neighborhood got too dangerous at night"                         44 1
"Transferred to a different office within the same company"           39 0
"Needed somewhere with a lower cost of living after I retired"        71 0
"My partner got a job in another state"                               36 1
"Family pressure to move back home"                                   27 1
"Our apartment building was sold and everyone had to leave"           49 1
"I just wanted a change of scenery honestly"                          33 0
"Health problems, needed to be near a better hospital"                77 1
"Couldn't afford to stay in the Bay Area on one income"               41 0
"Started a new job downtown and the commute was killing me"           35 0
"To take care of my mother after her stroke"                          58 1
"Rent control ended on our unit"                                      45 0
"Finished my degree and moved for work"                               26 0
end

label variable response "Why did you move? (open-ended)"
label variable age      "Age at interview"
label variable female   "Respondent is female"

list response in 1/5, abbreviate(20) noobs
di _n "N = " _N " responses."

********************************************************************************
* PART 2 -- The core pipeline
********************************************************************************

di _n(2) "{hline 78}"
di "PART 2 -- classify -> indicator variables -> analysis"
di "{hline 78}"

* Two things matter more than model choice for accuracy:
*
*   1. Write each category as a ONE-SENTENCE DEFINITION, not a one-word
*      label. Verbose definitions classify several percentage points
*      better on every model tested.
*   2. Always include an explicit "Other" category. Without it the model
*      is forced to guess on responses that genuinely fit nothing.
*
* Store the scheme in a local so the identical definitions are reused in
* every part below -- that reuse is what makes the provider comparison in
* Part 3 a fair one.

local cats                                                                      ///
    "Housing cost: The respondent moved because of rent, mortgage, affordability, or general cost of living." ///
    "Employment: The respondent moved for a job, a transfer, a promotion, education, or because of job loss." ///
    "Family: The respondent moved because of family -- caregiving, partners, children, or being near relatives." ///
    "Neighborhood: The respondent moved because of the local area itself -- safety, schools, climate, or amenities." ///
    "Other: The response does not fit any of the above categories."

if "$OPENAI_API_KEY" != "" {

    * generate() is a PREFIX, not a variable name. One byte variable per
    * category comes back: reason_Housing_cost, reason_Employment, and so
    * on, each 0/1. Multi-label by construction -- a response can be both
    * a housing-cost move and an employment move, which is usually what
    * you want and what a single categorical variable cannot represent.

    catllm classify response,                                                   ///
        categories(`cats')                                                      ///
        apikey($OPENAI_API_KEY)                                                 ///
        provider("openai")                                                      ///
        model("gpt-4o-mini")                                                    ///
        surveyquestion("Why did you move to your current residence?")           ///
        generate(reason)

    di _n "--- Coded responses ---"
    list response reason_*, separator(0) abbreviate(18)

    di _n "--- Marginal frequencies ---"
    tab1 reason_*

    * The payoff: those indicators are ordinary Stata variables. The
    * open-ended item is now available to the same tools as every
    * closed-ended item in the dataset.
    di _n "--- Open-ended item, now in a regression ---"
    regress reason_Housing_cost age female

    * Multi-label means the row total can exceed 1. Worth reporting.
    egen n_reasons = rowtotal(reason_*)
    di _n "--- Categories assigned per response ---"
    tabulate n_reasons
    drop n_reasons
}
else {
    di as txt "  (Part 2 skipped -- OPENAI_API_KEY not set)"
}

********************************************************************************
* PART 3 -- Provider comparison
********************************************************************************

di _n(2) "{hline 78}"
di "PART 3 -- Same categories, three providers"
di "{hline 78}"

* Identical prompt, identical categories, identical data -- only the
* vendor changes. Disagreement you see here is vendor disagreement and
* nothing else, which is what makes this comparison interpretable.

local have_all = ("$OPENAI_API_KEY" != "") & ("$ANTHROPIC_API_KEY" != "") ///
               & ("$GOOGLE_API_KEY" != "")

if `have_all' {

    catllm classify response, categories(`cats') apikey($OPENAI_API_KEY)        ///
        provider("openai")    model("gpt-4o-mini")             generate(oai)

    catllm classify response, categories(`cats') apikey($ANTHROPIC_API_KEY)     ///
        provider("anthropic") model("claude-haiku-4-5-20251001") generate(ant)

    catllm classify response, categories(`cats') apikey($GOOGLE_API_KEY)        ///
        provider("google")    model("gemini-2.5-flash")         generate(goo)

    * Pairwise agreement on one category, plus Cohen's kappa. kap is the
    * honest number to show: raw agreement is inflated whenever a category
    * is rare, and in survey coding most categories are rare.
    di _n "--- OpenAI vs. Anthropic, housing-cost category ---"
    tabulate oai_Housing_cost ant_Housing_cost, row
    capture noisily kap oai_Housing_cost ant_Housing_cost

    di _n "--- Three-way agreement across every category ---"
    foreach c in Housing_cost Employment Family Neighborhood Other {
        capture confirm variable oai_`c' ant_`c' goo_`c'
        if !_rc {
            quietly count if oai_`c' == ant_`c' & ant_`c' == goo_`c'
            local pct = 100 * r(N) / _N
            di as txt %-16s "`c'" as res %6.1f `pct' as txt "%  all three agree"
        }
    }

    drop oai_* ant_* goo_*
}
else {
    di as txt "  (Part 3 skipped -- needs OPENAI, ANTHROPIC and GOOGLE keys)"
}

********************************************************************************
* PART 4 -- Ensemble and consensus voting
********************************************************************************

di _n(2) "{hline 78}"
di "PART 4 -- Ensemble classification with consensus voting"
di "{hline 78}"

* Instead of picking a vendor, run several and vote. models() takes
* semicolon-separated entries, each "<model> <provider> <api_key>".
*
* consensus() sets the bar for assigning a category:
*
*     "unanimous"          every model must agree  (highest precision)
*     "two-thirds"         at least ~67% agree
*     "majority"           more than half agree    (default)
*     any number in [0,1]  your own threshold
*
* The Stata wrapper writes the consensus label per row. To also see how
* much the models disagreed, dump the full Python DataFrame with
* pyoptions("filename=...") and read it back -- that file carries a
* *_agreement column per category. Disagreement is a measurement, not a
* nuisance: high-disagreement rows are exactly the ones worth reading
* yourself.

if `have_all' {

    tempfile dump
    local dump_csv "`dump'.csv"

    local ens "gpt-4o-mini openai $OPENAI_API_KEY; claude-haiku-4-5-20251001 anthropic $ANTHROPIC_API_KEY; gemini-2.5-flash google $GOOGLE_API_KEY"

    catllm classify response,                                                   ///
        categories(`cats')                                                      ///
        apikey($OPENAI_API_KEY)                                                 ///
        models("`ens'")                                                         ///
        consensus("two-thirds")                                                 ///
        generate(vote)                                                          ///
        pyoptions("filename='`dump_csv''")

    di _n "--- Consensus labels (2 of 3 models must agree) ---"
    list response vote_*, separator(0) abbreviate(18)
    tab1 vote_*

    * --- Read the agreement scores back in --------------------------------
    capture frame drop ens
    frame create ens
    frame ens {
        quietly import delimited "`dump_csv'", clear varnames(1) stringcols(_all)

        capture ds *_agreement
        if !_rc & "`r(varlist)'" != "" {
            foreach v of varlist *_agreement {
                quietly destring `v', replace force
            }
            di _n "--- Model agreement per category (1.00 = all three agreed) ---"
            summarize *_agreement

            * Rows where the models split are the rows to hand-check.
            egen min_agree = rowmin(*_agreement)
            quietly count if min_agree < 1
            di _n as txt "Responses with any model disagreement: " as res r(N)
            di as txt "These are the rows to review by hand -- the ensemble has"
            di as txt "told you exactly where it is least sure."
        }
        else {
            di as txt "  (no *_agreement columns in the dump -- check cat-stack version)"
        }
    }
    capture frame drop ens
}
else {
    di as txt "  (Part 4 skipped -- needs OPENAI, ANTHROPIC and GOOGLE keys)"
}

********************************************************************************
* PART 5 -- Category discovery
********************************************************************************

di _n(2) "{hline 78}"
di "PART 5 -- Discovering a coding scheme you don't have yet"
di "{hline 78}"

* Parts 2-4 assumed a scheme. Often you don't have one, and inventing it
* by reading 3,000 responses is the actual bottleneck.
*
* catllm extract makes repeated independent passes over random slices of
* the corpus and keeps the categories that keep recurring. One LLM call
* is not reproducible; frequency across many calls is much closer to it.
* Use this to draft a scheme, then read it, edit it, and feed your
* edited version to classify. It is a starting point, not an answer.

if "$OPENAI_API_KEY" != "" {

    catllm extract response,                                                    ///
        apikey($OPENAI_API_KEY)                                                 ///
        provider("openai")                                                      ///
        model("gpt-4o-mini")                                                    ///
        researchquestion("Why do older adults move residence?")                 ///
        maxcategories(6)                                                        ///
        divisions(4)                                                            ///
        iterations(4)                                                           ///
        randomseed(20260922)

    di _n "--- Discovered categories ---"
    di as txt "n_categories = " as res r(n_categories)
    forvalues i = 1/`=r(n_categories)' {
        di as txt "  `i'. " as res "`r(cat`i')'"
    }

    * randomseed() is what makes this reproducible run to run. Report it
    * in your methods section along with divisions() and iterations().
}
else {
    di as txt "  (Part 5 skipped -- OPENAI_API_KEY not set)"
}

********************************************************************************
* PART 6 -- The zero-cost local route
********************************************************************************

di _n(2) "{hline 78}"
di "PART 6 -- Running entirely locally with Ollama"
di "{hline 78}"

* No API key, no per-token cost, and no text leaving your machine --
* which is often the deciding factor with restricted-use survey data
* under an IRB or a data-use agreement.
*
* One-time setup:
*     ollama pull qwen2.5:14b        (~9 GB; use qwen2.5:7b if tight on disk)
*
* provider("ollama") needs a non-empty placeholder for apikey(). cat-stack
* starts the local server itself if it isn't already running, and turns on
* a JSON-recovery formatter automatically, because small local models
* malform their JSON more often than cloud models do.

capture noisily catllm classify response,                                       ///
    categories(`cats')                                                          ///
    apikey("_")                                                                 ///
    provider("ollama")                                                          ///
    model("qwen2.5:14b")                                                        ///
    generate(oll)

if _rc {
    di as txt "  (Part 6 skipped -- Ollama not installed, or qwen2.5:14b not pulled)"
}
else {
    di _n "--- Local model results ---"
    tab1 oll_*

    * Local vs. cloud on identical text. On straightforward items the gap
    * is usually small; it widens on interpretive ones.
    capture confirm variable reason_Housing_cost
    if !_rc {
        di _n "--- Local vs. cloud, housing-cost category ---"
        tabulate oll_Housing_cost reason_Housing_cost
        capture noisily kap oll_Housing_cost reason_Housing_cost
    }
}

********************************************************************************
* Where to go next
********************************************************************************

di _n(2) "{hline 78}"
di "Done."
di "{hline 78}"
di "  help catllm                 command reference"
di "  catllm setup, check         what the Python backend has installed"
di ""
di "  Eight further worked examples ship with the package, in"
di "  stata-package/examples/ -- including prompt tuning (09) and"
di "  saturation analysis with explore (04)."
di ""
di "  Package:  https://github.com/chrissoria/cat-llm"
di "  Contact:  chrissoria@berkeley.edu"
