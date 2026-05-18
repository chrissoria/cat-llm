********************************************************************************
* 09-prompt-tuning.do
*
* INTERACTIVE: Automatic Prompt Optimization (APO) with prompt_tune.
*
* prompt_tune iteratively refines the classification system prompt by
* analyzing category-level errors in a small labeled sample:
*   1. Classifies a random sample with the current prompt
*   2. Opens a browser window -- you correct misclassifications
*   3. A meta-LLM rewrites targeted per-category instructions from your fixes
*   4. Re-classifies the sample with the updated prompt
*   5. Keeps the new prompt only if accuracy improves
*
* Categories are NEVER modified -- only the system prompt changes.
*
* What this script covers:
*   1. Baseline classify() without tuning
*   2. classify() with prompt_tune enabled (browser UI opens automatically)
*   3. Applying the tuned prompt to a full dataset
*
* Mirrors r-package/examples/09-prompt-tuning.Rmd
*
* Cost:    ~$0.10-0.30 (a handful of small model calls)
* Runtime: ~2-5 minutes including your time labeling corrections
*
* Requires: OPENAI_API_KEY set in environment (or substitute another provider)
********************************************************************************

clear all
set more off

adopath + "`c(pwd)'/.."

global OPENAI_API_KEY : env OPENAI_API_KEY
if "$OPENAI_API_KEY" == "" {
    di as error "OPENAI_API_KEY is not set. Export it first:"
    di as error `"  export OPENAI_API_KEY="sk-...""'
    exit 198
}

local MODEL        "gpt-4o-mini"
local SURVEY_Q     "How do you feel about the new employee wellness program?"

* --- Sample data -------------------------------------------------------------
input str200 response
"The new wellness program is great, I've been using it daily."
"It's confusing and the app crashes constantly."
"I haven't tried it yet but it sounds promising."
"The interface is intuitive and I like the dashboard."
"Customer support never responded to my emails."
"Pretty average -- nothing special either way."
"Best benefit my employer has ever offered."
"I tried it once and forgot about it."
"The login keeps failing on my phone."
"Really useful for tracking my steps and sleep."
"Would love more customization options."
"Absolutely love it -- recommend to everyone."
"Too many steps to get anything done."
"Fine for what it is, nothing more."
"It helped me sleep better within a week."
end

* --- 1. Baseline -- classify without tuning ----------------------------------
di _n "{hline 60}"
di "Step 1: Baseline classify() -- no prompt tuning"
di "{hline 60}"

catllm classify response,                                                              ///
    categories(                                                                        ///
        "Positive: The respondent expresses satisfaction, approval, or favorable sentiment." ///
        "Negative: The respondent expresses dissatisfaction, frustration, or criticism."    ///
        "Neutral: The respondent is factual, ambivalent, or does not express clear sentiment." ///
        "Other: The response does not fit any of the above categories.")               ///
    apikey($OPENAI_API_KEY)                                                            ///
    model("`MODEL'")                                                                   ///
    provider("openai")                                                                 ///
    surveyq("`SURVEY_Q'")                                                              ///
    generate(sentiment_baseline)

list response sentiment_baseline, separator(0) abbreviate(40)
tab sentiment_baseline

* --- 2. classify() with prompt_tune ------------------------------------------
* pyoptions() forwards arbitrary kwargs to cat_stack.classify(). We pass:
*   prompt_tune=5         -- show 5 rows per correction round (increase for more signal)
*   tune_iterations=1     -- one optimization pass (increase to 2-3 for real data)
*   tune_ui='browser'     -- browser-based correction UI (opens automatically)
*   tune_optimize='balanced' -- maximize avg(accuracy, sensitivity, precision)
*
* A browser window will open BEFORE this command returns. Correct any
* misclassified rows, then click Submit. classify() resumes automatically.
di _n "{hline 60}"
di "Step 2: classify() with prompt_tune -- browser window will open shortly"
di "Correct any misclassified rows, then click Submit."
di "{hline 60}"

catllm classify response,                                                              ///
    categories(                                                                        ///
        "Positive: The respondent expresses satisfaction, approval, or favorable sentiment." ///
        "Negative: The respondent expresses dissatisfaction, frustration, or criticism."    ///
        "Neutral: The respondent is factual, ambivalent, or does not express clear sentiment." ///
        "Other: The response does not fit any of the above categories.")               ///
    apikey($OPENAI_API_KEY)                                                            ///
    model("`MODEL'")                                                                   ///
    provider("openai")                                                                 ///
    surveyq("`SURVEY_Q'")                                                              ///
    generate(sentiment_tuned)                                                          ///
    pyoptions("prompt_tune=5, tune_iterations=1, tune_ui='browser', tune_optimize='balanced'")

di _n "{hline 60}"
di "Results after prompt_tune"
di "{hline 60}"

list response sentiment_baseline sentiment_tuned, separator(0) abbreviate(40)
tab sentiment_tuned

* --- What to look for --------------------------------------------------------
*
*   - A browser window opens automatically with 5 sample rows.
*     Checkboxes show the model's current classification for each row.
*     Correct any mistakes and click Submit.
*   - classify() re-runs with the updated system prompt.
*   - The tuned prompt is kept only if accuracy improved -- no risk of
*     making things worse.
*   - Compare sentiment_baseline vs sentiment_tuned to see the effect.
*
* Tips:
*   - Increase prompt_tune= (e.g. 10) for more signal per iteration.
*   - Increase tune_iterations= (e.g. 2-3) for more polish.
*   - Use tune_optimize='sensitivity' when false negatives are costly.
*   - Use tune_optimize='precision' when false positives are costly.
*   - Works with Ollama too: swap model() and provider("ollama").

di _n "Done."
