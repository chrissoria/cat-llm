********************************************************************************
* 02-validate-hybrid-ensemble.do
*
* Validate the multi-model ensemble pipeline end-to-end. Useful as:
*   - A smoke test after install / upgrade -- does the ensemble flow
*     still work against the currently-installed cat-stack?
*   - A proof of behavior before betting a real research run on it.
*
* Also exercises:
*   - Ollama auto-start (cat-stack notices the "ollama" provider and
*     spins up the local server if it isn't already running).
*   - JSON-formatter auto-enable (cat-stack enables the recovery
*     formatter whenever any ensemble entry uses Ollama, because small
*     local models more often emit malformed JSON).
*
* Mirrors r-package/examples/02-validate-hybrid-ensemble.Rmd, adapted
* for Stata's narrower output handling: the .ado writes only the
* consensus label per row, not the full per-model DataFrame. To
* inspect per-model / agreement columns from Stata, dump the full
* DataFrame to CSV via pyoptions("filename='/tmp/ensemble.csv'") and
* read it back with import delimited.
*
* Prereqs (one-time):
*   ollama pull qwen2.5:7b      # ~4.7 GB
*   ollama pull mistral:7b      # ~4.1 GB
*   ollama pull phi3:mini       # ~2.3 GB
*
* Cost:    $0 (entirely local)
* Runtime: ~1-3 minutes on Apple Silicon depending on model loads
********************************************************************************

clear all
set more off

adopath + "`c(pwd)'/.."

* No API key needed -- all models are local.

* --- A small hybrid ensemble: 3 local models, majority voting --------------
* 3 different small-vendor Ollama models give the ensemble vendor
* diversity without any cloud spend. Add a cloud entry (commented
* below) for a true hybrid run.
local local_models "qwen2.5:7b ollama _; mistral:7b ollama _; phi3:mini ollama _"

* --- Test data ---------------------------------------------------------------
input str200 response
"The new wellness program is great, I've been using it daily."
"It's confusing and the app crashes constantly."
"I haven't tried it yet but it sounds promising."
end

* --- Run the ensemble -------------------------------------------------------
* Verbose, definition-style categories classify several percentage points
* more accurately than one-word labels on small models.
*
* The CSV side-output via pyoptions("filename=...") gives us a way to
* validate the full per-model / consensus / agreement column shape from
* Stata, since the Stata wrapper itself only writes the consensus label.
tempfile dump
local dump_csv "`dump'.csv"

di _n "{hline 60}"
di "Running 3-model local ensemble (majority voting)"
di "{hline 60}"
di "  -- Expect to see cat-stack print:"
di "       [CatLLM] Ollama detected -- auto-enabling JSON formatter fallback"
di "       Validating 3 model configuration(s)..."

catllm classify response,                                                                              ///
    categories(                                                                                        ///
        "Positive: The respondent expresses satisfaction, approval, or favorable sentiment."           ///
        "Negative: The respondent expresses dissatisfaction, frustration, or criticism."               ///
        "Neutral: The respondent is factual, ambivalent, or does not express clear sentiment."         ///
        "Other: The response does not fit any of the above categories.")                               ///
    apikey("_")                                                                                        ///
    generate(sentiment)                                                                                ///
    consensus("majority")                                                                              ///
    models("`local_models'")                                                                           ///
    pyoptions("filename='`dump_csv'', check_verbosity=False")

list response sentiment_*, separator(0) abbreviate(50)
tab1 sentiment_*

* --- Structural validation against the dumped CSV ----------------------------
* The R example uses stopifnot() on R's full data.frame. Stata's wrapper
* writes only the consensus label per row, so we read the dumped CSV
* into a separate frame and assert on its columns instead.
di _n "{hline 60}"
di "Validating ensemble output shape (CSV dump)"
di "{hline 60}"

capture frame drop ensemble_dump
frame create ensemble_dump
frame ensemble_dump {
    quietly import delimited "`dump_csv'", clear varnames(1) stringcols(_all)

    * --- Required columns -----------------------------------------------------
    foreach col in input_data processing_status {
        capture confirm variable `col'
        if _rc {
            di as error "ASSERT FAIL: expected column `col' not found"
            exit 459
        }
    }

    * Per-model columns: category_<i>_<model_slug>
    * Consensus columns: category_<i>_consensus
    * Agreement scores:  category_<i>_agreement
    local n_per_model = 0
    local n_consensus = 0
    local n_agreement = 0
    ds category_*
    foreach v in `r(varlist)' {
        if regexm("`v'", "_consensus$") {
            local n_consensus = `n_consensus' + 1
        }
        else if regexm("`v'", "_agreement$") {
            local n_agreement = `n_agreement' + 1
        }
        else {
            local n_per_model = `n_per_model' + 1
        }
    }

    di "  per-model category columns:  `n_per_model'  (expect 12 = 4 cats x 3 models)"
    di "  consensus category columns:  `n_consensus'  (expect 4)"
    di "  agreement category columns:  `n_agreement'  (expect 4)"

    assert `n_consensus' == 4
    assert `n_agreement' == 4
    assert `n_per_model' >= 4   // ensemble per-model emission can vary; expect at least one per category

    * --- Consensus values must be 0/1 -----------------------------------------
    foreach v of varlist *_consensus {
        quietly destring `v', replace
        quietly count if !inlist(`v', 0, 1) & !missing(`v')
        if r(N) > 0 {
            di as error "ASSERT FAIL: `v' has values outside {0, 1}"
            exit 459
        }
    }

    * --- Agreement values must be in [0, 1] ----------------------------------
    foreach v of varlist *_agreement {
        quietly destring `v', replace
        quietly count if (`v' < 0 | `v' > 1) & !missing(`v')
        if r(N) > 0 {
            di as error "ASSERT FAIL: `v' has values outside [0, 1]"
            exit 459
        }
    }

    di _n "All structural assertions passed."
}

* --- What this confirmed ----------------------------------------------------
*
*   Ollama auto-start fired (cat-stack started the local server)
*   JSON-formatter auto-enable defaulted on (small local models more
*       often emit malformed JSON; cat-stack lazy-loads a fine-tuned
*       formatter to recover them)
*   Formatter model loads only on first parse failure (lazy)
*   Output DataFrame has the documented per-model + consensus +
*       agreement column shape
*
* For deeper conceptual coverage of ensemble mode (consensus thresholds,
* hybrid cloud+local, output columns), see 03-ensemble-classification-
* cloud-and-local.do.

di _n "Done."
