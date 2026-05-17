********************************************************************************
* 02-ensemble-classification-cloud-and-local.do
*
* Multi-model ensemble classification -- run the same input through
* several LLMs and combine their predictions via majority / unanimous
* voting. Often improves accuracy by reducing individual model bias.
* Per the CatLLM evaluation, inexpensive open-weight ensembles can
* match or exceed individual frontier-class models at a fraction of
* the cost.
*
* Mirrors r-package/examples/02-ensemble-classification-cloud-and-local.Rmd.
*
* Requires:
*   OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY (cloud section)
*   Ollama installed + `ollama pull qwen2.5:14b` (hybrid section)
*
* Cost:    ~$0.05 in cloud tokens for the 5-row cloud-only ensemble.
*          The hybrid section also runs a local model -- free but slower.
* Runtime: ~30 seconds cloud-only, ~1-2 minutes for the hybrid section.
********************************************************************************

clear all
set more off

adopath + "`c(pwd)'/.."

global OPENAI_API_KEY    : env OPENAI_API_KEY
global ANTHROPIC_API_KEY : env ANTHROPIC_API_KEY
global GOOGLE_API_KEY    : env GOOGLE_API_KEY

* --- Example data ------------------------------------------------------------
input str200 response
"Took a new job in Chicago"
"Wanted to be closer to grandkids"
"Couldn't afford rent in the Bay Area"
"Job market collapsed after the layoffs"
"Family pressure to move home"
end

* --- 1. Cloud-only ensemble (3 providers, unanimous voting) ------------------
* The `models()` option specifies one model per semicolon-separated entry,
* each as "<model> <provider> <api_key>". Build the string in a single-line
* local so the macro contents pass through cleanly. Verbose, definition-style
* category labels classify more accurately than one-word labels.
di _n "{hline 60}"
di "1. Cloud ensemble (OpenAI + Anthropic + Google, unanimous voting)"
di "{hline 60}"

if "$OPENAI_API_KEY" == "" | "$ANTHROPIC_API_KEY" == "" | "$GOOGLE_API_KEY" == "" {
    di as txt "  (cloud ensemble skipped -- one of OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY not set)"
}
else {
    local cloud_models "gpt-4o-mini openai $OPENAI_API_KEY; claude-haiku-4-5-20251001 anthropic $ANTHROPIC_API_KEY; gemini-2.5-flash google $GOOGLE_API_KEY"

    catllm classify response,                                                                                 ///
        categories(                                                                                           ///
            "Job/school: A change in employment, education, or career, including transfers and retirement."  ///
            "Family: Relationship changes, having children, supporting relatives, or relocating to be near family." ///
            "Cost of living: Housing affordability, cost of goods, or general economic pressure."             ///
            "Other: The response does not fit any of the above categories.")                                  ///
        apikey($OPENAI_API_KEY)                                                                               ///
        generate(reason_cloud)                                                                                ///
        consensus("unanimous")                                                                                ///
        models("`cloud_models'")

    list response reason_cloud, separator(0) abbreviate(30)
    tab reason_cloud
}

* --- 2. Hybrid ensemble (cloud + local Ollama, majority voting) --------------
* Mixing cloud and local models gives you the speed of cloud + the cost /
* privacy benefits of local. Each model's vote counts equally. With 3 models,
* majority = at least 2 of 3 agree.
*
* The local model needs an api_key placeholder ("_" or any non-empty string).
* Use qwen2.5:14b for higher accuracy (~9 GB); switch to qwen2.5:7b (~4.7 GB)
* if disk-constrained.
di _n "{hline 60}"
di "2. Hybrid ensemble (OpenAI + Anthropic + local Ollama, majority voting)"
di "{hline 60}"

if "$OPENAI_API_KEY" == "" | "$ANTHROPIC_API_KEY" == "" {
    di as txt "  (hybrid ensemble skipped -- OPENAI_API_KEY or ANTHROPIC_API_KEY not set)"
}
else {
    local hybrid_models "gpt-4o-mini openai $OPENAI_API_KEY; claude-haiku-4-5-20251001 anthropic $ANTHROPIC_API_KEY; qwen2.5:14b ollama _"

    capture noisily catllm classify response,                                                                 ///
        categories(                                                                                           ///
            "Job/school: A change in employment, education, or career, including transfers and retirement."  ///
            "Family: Relationship changes, having children, supporting relatives, or relocating to be near family." ///
            "Cost of living: Housing affordability, cost of goods, or general economic pressure."             ///
            "Other: The response does not fit any of the above categories.")                                  ///
        apikey($OPENAI_API_KEY)                                                                               ///
        generate(reason_hybrid)                                                                               ///
        consensus("majority")                                                                                 ///
        models("`hybrid_models'")

    if _rc {
        di as txt "  (hybrid ensemble skipped -- likely Ollama not installed or qwen2.5:14b not pulled)"
    }
    else {
        list response reason_hybrid, separator(0) abbreviate(30)
        tab reason_hybrid
    }
}

* --- Consensus thresholds you can pass to consensus() ------------------------
*
*  "unanimous"           Only assign a category if every model agrees.
*                        Highest precision (default).
*  "majority"            Assign if more than half agree (50%). Good balance.
*  "two-thirds"          At least two-thirds must agree (~67%).
*  numeric (0.0--1.0)    Any threshold between 0 and 1.

* --- Output columns -----------------------------------------------------------
*
* For an ensemble with N models classifying into K categories, cat-stack
* produces a DataFrame with:
*
*   category_<i>_<model_slug>   one (model x category) indicator each
*   category_<i>_consensus      the voted result (0/1) per category
*   <category>_agreement        how many models agreed
*
* The Stata wrapper reads the *_consensus columns to write the single
* assigned label into generate(). For full per-model output, call the
* Python cat-stack ensemble path directly via reticulate (R) or python:
* (Stata), or save the result with the underlying filename / save_directory
* kwargs via pyoptions().

di _n "Done."
