********************************************************************************
* test_stata_package.do
* Tests catllm Stata package functions
********************************************************************************

clear all
set more off

* --- Add stata-package dir to search path ---
local pkg_dir = subinstr("`c(pwd)'", "/examples", "", .)
adopath + "`pkg_dir'"

* --- Step 1: Setup check ---
di _n "{hline 60}"
di "Step 1: Verify catllm Python backend"
di "{hline 60}"

catllm setup, check

* --- Set API key ---
* Set OPENAI_API_KEY as a global before running, e.g.:
*   global OPENAI_API_KEY "sk-..."
* or pass via command line:
*   stata-se -b do test_stata_package.do OPENAI_API_KEY=sk-...
if "$OPENAI_API_KEY" == "" {
    di as error "OPENAI_API_KEY global not set. Set it and re-run."
    exit 198
}

* --- Step 2: Create test data ---
di _n "{hline 60}"
di "Step 2: Create test data"
di "{hline 60}"

input str200 response
"The healthcare system needs major reform, especially for rural areas"
"I think the economy is doing well but wages need to increase"
"Education funding is my top priority, especially for public schools"
"Climate change is the most important issue facing our generation"
"We need better infrastructure and roads in our community"
"Healthcare costs are too high and prescription drugs are unaffordable"
"The economy is terrible and inflation is out of control"
"Schools need more teachers and better technology"
"We should invest more in renewable energy sources"
"Public transportation needs improvement in suburban areas"
end

list response

* --- Step 3: classify ---
di _n "{hline 60}"
di "Step 3: catllm classify"
di "{hline 60}"

catllm classify response,                           ///
    categories("Healthcare" "Economy"               ///
               "Education" "Environment"            ///
               "Infrastructure")                    ///
    apikey($OPENAI_API_KEY)                         ///
    generate(topic)

list response topic, separator(0) abbreviate(20)
tab topic

* --- Step 4: extract ---
di _n "{hline 60}"
di "Step 4: catllm extract"
di "{hline 60}"

catllm extract response,                            ///
    apikey($OPENAI_API_KEY)                         ///
    maxcategories(5)                                ///
    description("Open-ended responses about policy priorities")

return list

* --- Step 5: summarize ---
di _n "{hline 60}"
di "Step 5: catllm summarize"
di "{hline 60}"

catllm summarize response,                          ///
    apikey($OPENAI_API_KEY)                         ///
    generate(summary)                               ///
    instructions("Summarize in 10 words or fewer")

list response summary, separator(0) abbreviate(30)

di _n "All tests complete."
