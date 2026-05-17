********************************************************************************
* 04-exploring-categories-with-explore.do
*
* Use `catllm explore` for SATURATION ANALYSIS. Unlike `catllm extract`
* (which de-duplicates and merges similar labels), `catllm explore`
* returns every raw category string from every chunk across every
* iteration -- with duplicates intact. That makes it the right tool for
* understanding category stability: do the same themes keep appearing,
* or are new ones still being discovered with each iteration?
*
* Use case: you have ~500 open-ended survey responses and want to know,
* "If I sample 50 of them, am I capturing most of the themes? Or do I
* need to look at all 500?"
*
* Requires: OPENAI_API_KEY (or set provider() to your preferred backend)
* Cost:     ~$0.05 across the two runs below
* Runtime:  ~30 seconds for the basic run + ~30 seconds for the iteration
*           sweep
********************************************************************************

clear all
set more off

adopath + "`c(pwd)'/.."

global OPENAI_API_KEY : env OPENAI_API_KEY
if "$OPENAI_API_KEY" == "" {
    di as error "OPENAI_API_KEY not set."
    exit 198
}

* --- Example data: reasons people gave for moving ----------------------------
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

* --- 1. Basic explore run with survey-question framing -----------------------
* domain(survey) routes through cat-survey which injects the survey question
* into the prompt for more accurate theme discovery.
di _n "{hline 60}"
di "1. Raw category extraction (12 iterations, 8 divisions)"
di "{hline 60}"

catllm explore open_response,                       ///
    apikey($OPENAI_API_KEY)                         ///
    model("gpt-4o-mini")                            ///
    iterations(12)                                  ///
    divisions(8)                                    ///
    catsperchunk(10)                                ///
    maxcategories(15)                               ///
    description("Reasons people gave for moving to a new city")  ///
    pyoptions("research_question='Why did you move to this city?'")

return list

* Stata-friendly summary of the top themes
di _n "Top categories returned by explore() (top 20 macro):"
di `"  `r(top_categories)'"'

* --- 2. Saturation: re-run at fewer iterations to see how unique-count grows ---
* If the rate of new unique categories per iteration is dropping toward zero,
* you've reached saturation. We approximate that here by running at three
* iteration counts and comparing r(n_unique).
di _n "{hline 60}"
di "2. Saturation sweep (3 vs 6 vs 12 iterations)"
di "{hline 60}"

foreach iters in 3 6 12 {
    capture drop sat_run
    quietly catllm explore open_response,           ///
        apikey($OPENAI_API_KEY)                     ///
        model("gpt-4o-mini")                        ///
        iterations(`iters')                         ///
        divisions(4)                                ///
        catsperchunk(8)                             ///
        maxcategories(15)
    di "  iterations = " %2.0f `iters' "    n_raw = " r(n_raw) ///
       "    n_unique = " r(n_unique)
}

* A flattening unique-count trend means you've hit saturation.
* A still-rising trend means more iterations are still revealing new themes.

* --- 3. When to use explore() vs extract() -----------------------------------
*
*  explore   -- raw category strings (with duplicates)
*               Best for: saturation analysis, finding stable themes,
*               methods-section justification of category schemes.
*
*  extract   -- de-duplicated, normalized top categories
*               Best for: building a clean codebook to pass into
*               `catllm classify`. See example 05.

di _n "Done."
