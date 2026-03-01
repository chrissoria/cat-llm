********************************************************************************
* catllm_example.do
* Demonstration of the catllm Stata package
* Requires: catllm Python backend, an OpenAI API key
********************************************************************************

clear all
set more off

* --- Setup (one-time) --------------------------------------------------------
* Install the Python backend (skip if already installed)
catllm setup, check
* catllm setup             // uncomment to install
* catllm setup, pdf        // uncomment to install with PDF support

* --- Store your API key -------------------------------------------------------
* Option 1: Set in your do-file (not recommended for shared scripts)
* global OPENAI_API_KEY "sk-..."

* Option 2: Set in your profile.do so it persists across sessions
* Run: doedit "`c(sysdir_personal)'profile.do"
* Add: global OPENAI_API_KEY "sk-..."

* --- Create example data ------------------------------------------------------
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

* --- Example 1: Classify with predefined categories ---------------------------
di _n "{hline 60}"
di "Example 1: Classify survey responses"
di "{hline 60}"

catllm classify response,                                       ///
    categories("Healthcare" "Economy" "Education"               ///
               "Environment" "Infrastructure")                  ///
    apikey($OPENAI_API_KEY)                                     ///
    generate(topic)

list response topic, separator(0) abbreviate(20)
tab topic

* --- Example 2: Discover categories automatically ----------------------------
di _n "{hline 60}"
di "Example 2: Extract categories from data"
di "{hline 60}"

catllm extract response,                                        ///
    apikey($OPENAI_API_KEY)                                     ///
    maxcategories(5)                                            ///
    description("Open-ended responses about policy priorities")

return list

* --- Example 3: Use discovered categories for classification ------------------
di _n "{hline 60}"
di "Example 3: Extract then classify pipeline"
di "{hline 60}"

* Get the discovered categories
local discovered = r(categories)
di "Discovered: `discovered'"

* Classify using those categories
catllm classify response,                                       ///
    categories(`discovered')                                    ///
    apikey($OPENAI_API_KEY)                                     ///
    generate(auto_topic) replace

list response auto_topic, separator(0) abbreviate(20)

* --- Example 4: Summarize responses -------------------------------------------
di _n "{hline 60}"
di "Example 4: Summarize text"
di "{hline 60}"

catllm summarize response,                                      ///
    apikey($OPENAI_API_KEY)                                     ///
    generate(summary)                                           ///
    instructions("Summarize in 10 words or fewer")

list response summary, separator(0) abbreviate(30)

* --- Example 5: Chain-of-thought classification -------------------------------
di _n "{hline 60}"
di "Example 5: Chain-of-thought for harder classifications"
di "{hline 60}"

catllm classify response,                                       ///
    categories("Individual Concern" "Systemic Issue")           ///
    apikey($OPENAI_API_KEY)                                     ///
    generate(scope)                                             ///
    chainofthought                                              ///
    description("Whether the response focuses on a personal/individual concern or a systemic/societal issue")

list response scope, separator(0) abbreviate(20)
tab scope

di _n "Done! See {bf:help catllm} for full documentation."
