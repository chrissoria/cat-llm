********************************************************************************
* test_prompt_tune.do
*
* INTERACTIVE TEST: verify that `prompt_tune` (cat-stack's automatic prompt
* optimization) is reachable through the Stata wrapper via pyoptions().
*
* prompt_tune iteratively refines the classification system prompt by
* asking you to correct a small sample. It is fundamentally interactive
* — a browser window opens with checkboxes for each (row, category) cell;
* you fix mistakes; the meta-LLM rewrites the prompt; the sample is
* re-classified; repeat until accuracy converges or max_iterations is hit.
*
* This script is NOT a regression test. It is a manual verification you
* run once before tagging a release / submitting to SSC, to confirm that
*   1. The native prompttune() / tuneiterations() / tuneui() options
*      flow into cat_stack.classify() correctly
*   2. The browser UI opens and accepts corrections
*   3. classify() returns a labeled dataset after tuning completes
*
* What you should see:
*   - "Running 1 iteration(s) of automatic prompt optimization..."
*   - A browser window auto-opens showing 5 sample rows with category
*     checkboxes. Mark up to ~3 of them as you would label them. Submit.
*   - cat-stack rewrites the system prompt and re-classifies.
*   - Final labels print as a Stata `list` and `tab`.
*
* Requires: OPENAI_API_KEY (or substitute another provider/model).
*
* Cost:    ~$0.10 (a handful of small gpt-4o-mini calls)
* Runtime: ~1-3 minutes including your time labeling
********************************************************************************

clear all
set more off

adopath + "`c(pwd)'/.."

global OPENAI_API_KEY : env OPENAI_API_KEY
if "$OPENAI_API_KEY" == "" {
    di as error "OPENAI_API_KEY not set."
    exit 198
}

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
end

* --- Classify with prompt_tune enabled ---------------------------------------
* Native Stata options drive cat-stack's APO loop:
*   prompttune(5)           -- run APO with a 5-row sample (small for testing)
*   tuneiterations(1)       -- one instruction attempt per category (fast)
*   tuneui("browser")       -- browser-based corrections (readable)
*   tuneoptimize("balanced") -- avg(accuracy, sensitivity, precision)
*
* NOTE: The browser window will pop open BEFORE this command returns. Submit
* your corrections, then control returns to Stata.
di _n "{hline 60}"
di "Launching prompt_tune. A browser window will open shortly."
di "Mark a few rows the way you'd label them, then click Submit."
di "{hline 60}"

catllm classify response,                                                              ///
    categories(                                                                        ///
        "Positive: The respondent expresses satisfaction, approval, or favorable sentiment." ///
        "Negative: The respondent expresses dissatisfaction, frustration, or criticism."    ///
        "Neutral: The respondent is factual, ambivalent, or does not express clear sentiment." ///
        "Other: The response does not fit any of the above categories.")               ///
    apikey($OPENAI_API_KEY)                                                            ///
    model("gpt-4o-mini")                                                               ///
    provider("openai")                                                                 ///
    generate(sentiment)                                                                ///
    prompttune(5)                                                                      ///
    tuneiterations(1)                                                                  ///
    tuneui("browser")                                                                  ///
    tuneoptimize("balanced")

* --- Verify the run produced labels ------------------------------------------
di _n "{hline 60}"
di "Results after prompt_tune"
di "{hline 60}"

list response sentiment_*, separator(0) abbreviate(40)
tab1 sentiment_*

* --- What this verified -------------------------------------------------------
*
*   pyoptions() correctly forwards prompt_tune / tune_iterations /
*       tune_ui / tune_optimize to cat_stack.classify()
*   The browser-based correction UI launches and accepts input
*   classify() returns a labeled column after tuning completes
*   Stata's output handling captures the labels correctly
*
* If the browser UI did not open, check that:
*   - cat-stack is up-to-date: catllm setup, upgrade
*   - You have a default browser configured
*   - No firewall is blocking localhost ports (the UI runs on a random
*     port on 127.0.0.1)
*
* For the conceptual / cost / runtime overview of prompt_tune, see the
* Python README's "Automatic Prompt Optimization" section.

di _n "Done."
