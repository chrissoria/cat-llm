********************************************************************************
* 02-ensemble-classification-cloud-and-local.do
*
* Multi-model ensemble classification. Run the same input through several
* models and combine their predictions via majority / unanimous voting.
* Often improves accuracy by reducing individual model bias. Per the
* CatLLM evaluation, inexpensive open-weight ensembles can match or
* exceed individual frontier-class models at a fraction of the cost.
*
* Requires: OPENAI_API_KEY, ANTHROPIC_API_KEY, optionally GOOGLE_API_KEY.
* Hybrid section additionally requires Ollama installed locally.
*
* Cost:    ~$0.05 in cloud tokens for the 5-row cloud-only ensemble.
*          The hybrid section also runs a local model -- free but slower.
* Runtime: ~30 seconds cloud-only, ~1-2 minutes for the hybrid section
********************************************************************************

clear all
set more off

adopath + "`c(pwd)'/.."

global OPENAI_API_KEY    : env OPENAI_API_KEY
global ANTHROPIC_API_KEY : env ANTHROPIC_API_KEY
global GOOGLE_API_KEY    : env GOOGLE_API_KEY

if "$OPENAI_API_KEY" == "" | "$ANTHROPIC_API_KEY" == "" {
    di as error "OPENAI_API_KEY and ANTHROPIC_API_KEY must both be set."
    exit 198
}

* --- Example data ------------------------------------------------------------
input str200 response
"Took a new job in Chicago"
"Wanted to be closer to grandkids"
"Couldn't afford rent in the Bay Area"
"Job market collapsed after the layoffs"
"Family pressure to move home"
end

* --- 1. Cloud-only ensemble (unanimous voting) -------------------------------
* The `models()` option specifies one model per semicolon-separated entry,
* each as "<model> <provider> <api_key>". Build the string in a single-line
* local so the macro contents pass through cleanly. Verbose category labels
* with definitions classify more accurately than one-word labels.
di _n "{hline 60}"
di "1. Cloud ensemble (OpenAI + Anthropic, unanimous voting)"
di "{hline 60}"

local cloud_models "gpt-4o-mini openai $OPENAI_API_KEY; claude-haiku-4-5-20251001 anthropic $ANTHROPIC_API_KEY"

catllm classify response,                                                                  ///
    categories(                                                                            ///
        "Job/school: A change in employment, education, or career."                        ///
        "Family: Relationship changes, having children, or relocating to be near family."  ///
        "Cost of living: Housing affordability or general economic pressure."              ///
        "Other: The response does not fit any of the above.")                              ///
    apikey($OPENAI_API_KEY)                                                                ///
    generate(reason_cloud)                                                                 ///
    consensus("unanimous")                                                                 ///
    models("`cloud_models'")

list response reason_cloud, separator(0) abbreviate(30)
tab reason_cloud

* --- 2. Hybrid ensemble (cloud + local) -- comment out if Ollama not installed ---
di _n "{hline 60}"
di "2. Hybrid ensemble (cloud + local Ollama, majority voting)"
di "{hline 60}"

* The local model needs an api_key placeholder ("_" or any non-empty string)
* in the models() string. Each model's vote counts equally. Majority = at
* least 2 of 3 agree.
local hybrid_models "gpt-4o-mini openai $OPENAI_API_KEY; claude-haiku-4-5-20251001 anthropic $ANTHROPIC_API_KEY; qwen2.5:7b ollama _"

capture noisily catllm classify response,                                                  ///
    categories(                                                                            ///
        "Job/school: A change in employment, education, or career."                        ///
        "Family: Relationship changes, having children, or relocating to be near family."  ///
        "Cost of living: Housing affordability or general economic pressure."              ///
        "Other: The response does not fit any of the above.")                              ///
    apikey($OPENAI_API_KEY)                                                                ///
    generate(reason_hybrid)                                                                ///
    consensus("majority")                                                                  ///
    models("`hybrid_models'")

if _rc {
    di as txt "  (hybrid ensemble skipped -- likely Ollama not installed or model not pulled)"
}
else {
    list response reason_hybrid, separator(0) abbreviate(30)
    tab reason_hybrid
}

* --- 3. Consensus thresholds you can pass to `consensus()` -------------------
* "unanimous"   -- only assign category if every model agrees (default; highest precision)
* "majority"    -- assign if more than half agree (50%)
* "two-thirds"  -- at least two-thirds must agree (~67%)
* "0.0"-"1.0"   -- any numeric threshold

di _n "Done. Use the per-model columns (if exposed) to diagnose disagreements."
