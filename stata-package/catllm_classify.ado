*! catllm_classify -- Classify text into categories using LLMs
*! Version 1.0.0

program define catllm_classify, rclass
    version 16

    syntax varname(string) [if] [in],                           ///
        Categories(string asis) APIkey(string)                  ///
        [                                                       ///
            GENerate(name)                                      ///
            Model(string)                                       ///
            Provider(string)                                    ///
            Description(string asis)                            ///
            SURVEYquestion(string asis)                         ///
            CREativity(real -1)                                 ///
            CHAINofthought                                      ///
            THINKing(integer 0)                                 ///
            STEPback                                            ///
            CONText                                             ///
            CONSensus(string)                                   ///
            MODels(string asis)                                 ///
            MAXWorkers(integer 0)                               ///
            MAXRetries(integer 5)                               ///
            RETRYdelay(real 1.0)                                ///
            ROWdelay(real 0.0)                                  ///
            FAILstrategy(string)                                ///
            NOJSONschema                                        ///
            REPLACE                                             ///
        ]

    * ----- defaults -----
    if "`generate'" == "" local generate "_catllm_class"
    if "`model'"    == "" local model "gpt-4o"
    if "`provider'" == "" local provider "auto"
    if "`consensus'" == "" local consensus "majority"
    if "`failstrategy'" == "" local failstrategy "partial"

    * ----- validate -----
    confirm string variable `varlist'

    if "`replace'" == "" {
        confirm new variable `generate'
    }
    else {
        capture confirm variable `generate'
        if !_rc {
            drop `generate'
        }
    }

    * ----- mark sample -----
    marksample touse, strok
    quietly count if `touse'
    local nobs = r(N)
    if `nobs' == 0 {
        di as error "no observations"
        exit 2000
    }
    di as txt "Classifying `nobs' observations..."

    * ----- create result variable -----
    quietly gen str244 `generate' = ""

    * ----- store parameters for Python -----
    local _catllm_var      "`varlist'"
    local _catllm_gen      "`generate'"
    local _catllm_cats     `"`categories'"'
    local _catllm_key      "`apikey'"
    local _catllm_model    "`model'"
    local _catllm_provider "`provider'"
    local _catllm_desc     `"`description'"'
    local _catllm_survey   `"`surveyquestion'"'
    local _catllm_touse    "`touse'"
    local _catllm_nobs     "`nobs'"
    local _catllm_cot      "`chainofthought'"
    local _catllm_think    "`thinking'"
    local _catllm_stepback "`stepback'"
    local _catllm_context  "`context'"
    local _catllm_consensus "`consensus'"
    local _catllm_models   `"`models'"'
    local _catllm_workers  "`maxworkers'"
    local _catllm_retries  "`maxretries'"
    local _catllm_rdelay   "`retrydelay'"
    local _catllm_rowdelay "`rowdelay'"
    local _catllm_failstr  "`failstrategy'"
    local _catllm_nojson   "`nojsonschema'"

    if `creativity' == -1 {
        local _catllm_creat ""
    }
    else {
        local _catllm_creat "`creativity'"
    }

    * ----- call Python -----
    python: _catllm_do_classify()

    * ----- return results -----
    quietly count if `generate' != "" & `touse'
    local classified = r(N)
    return scalar N = `nobs'
    return scalar N_classified = `classified'
    return local variable "`generate'"
    return local model "`model'"
    return local provider "`provider'"

    di as txt ""
    di as txt "Classification complete."
    di as txt "  Observations: `nobs'"
    di as txt "  Classified:   `classified'"
    di as txt "  Variable:     {res}`generate'"
end

python:
def _catllm_do_classify():
    from sfi import Data, Macro, SFIToolkit
    import catllm

    # --- read Stata parameters ---
    varname    = Macro.getLocal("_catllm_var")
    genname    = Macro.getLocal("_catllm_gen")
    cats_str   = Macro.getLocal("_catllm_cats")
    api_key    = Macro.getLocal("_catllm_key")
    model      = Macro.getLocal("_catllm_model")
    provider   = Macro.getLocal("_catllm_provider")
    desc       = Macro.getLocal("_catllm_desc")
    survey_q   = Macro.getLocal("_catllm_survey")
    touse      = Macro.getLocal("_catllm_touse")
    cot        = Macro.getLocal("_catllm_cot") != ""
    think      = int(Macro.getLocal("_catllm_think") or "0")
    stepback   = Macro.getLocal("_catllm_stepback") != ""
    context    = Macro.getLocal("_catllm_context") != ""
    consensus  = Macro.getLocal("_catllm_consensus")
    models_str = Macro.getLocal("_catllm_models")
    workers    = int(Macro.getLocal("_catllm_workers") or "0")
    retries    = int(Macro.getLocal("_catllm_retries") or "5")
    rdelay     = float(Macro.getLocal("_catllm_rdelay") or "1.0")
    rowdelay   = float(Macro.getLocal("_catllm_rowdelay") or "0.0")
    failstr    = Macro.getLocal("_catllm_failstr")
    nojson     = Macro.getLocal("_catllm_nojson") != ""
    creat_str  = Macro.getLocal("_catllm_creat")

    creativity = float(creat_str) if creat_str else None

    # --- parse categories ---
    # Accepts: "Cat A" "Cat B" "Cat C"  or  Cat_A Cat_B Cat_C
    import shlex
    try:
        categories = shlex.split(cats_str)
    except ValueError:
        categories = cats_str.split()

    # --- parse models for ensemble ---
    models = None
    if models_str:
        # Format: "model1 provider1 key1; model2 provider2 key2"
        models = []
        for entry in models_str.split(";"):
            parts = entry.strip().split()
            if len(parts) >= 3:
                models.append(tuple(parts[:3]))
            elif len(parts) == 2:
                models.append((parts[0], parts[1], api_key))

    # --- read text data from Stata ---
    var_idx   = Data.getVarIndex(varname)
    gen_idx   = Data.getVarIndex(genname)
    touse_idx = Data.getVarIndex(touse)
    n         = Data.getObsTotal()

    texts = []
    obs_map = []  # maps position in texts[] back to Stata obs number
    for i in range(n):
        if Data.getNum(touse_idx, i) == 1:
            val = Data.getStr(var_idx, i)
            texts.append(val if val else "")
            obs_map.append(i)

    if not texts:
        SFIToolkit.errprintln("{err}No valid observations found.")
        return

    # --- call catllm ---
    kwargs = dict(
        input_data=texts,
        categories=categories,
        api_key=api_key,
        user_model=model,
        model_source=provider,
        description=desc,
        survey_question=survey_q,
        chain_of_thought=cot,
        thinking_budget=think,
        step_back_prompt=stepback,
        context_prompt=context,
        consensus_threshold=consensus,
        use_json_schema=not nojson,
        max_retries=retries,
        retry_delay=rdelay,
        row_delay=rowdelay,
        fail_strategy=failstr,
        add_other=False,
        check_verbosity=False,
    )

    if models:
        kwargs["models"] = models
    if workers > 0:
        kwargs["max_workers"] = workers
    if creativity is not None:
        kwargs["creativity"] = creativity

    try:
        result_df = catllm.classify(**kwargs)
    except Exception as e:
        SFIToolkit.errprintln("{err}catllm.classify() failed: " + str(e))
        return

    # --- determine classification per row ---
    # Single model: columns are category names with 0/1
    # Ensemble: columns are {cat}_consensus with 0/1
    cols = list(result_df.columns)

    # Try consensus columns first (ensemble mode)
    consensus_cols = [c for c in cols if c.endswith("_consensus")]
    if consensus_cols:
        cat_cols = consensus_cols
        # Map consensus column back to category name
        col_to_cat = {}
        for cc in consensus_cols:
            base = cc.rsplit("_consensus", 1)[0]
            col_to_cat[cc] = base
    else:
        # Single model: category columns match the input categories
        # Try exact match first
        cat_cols = [c for c in cols if c in categories]
        if not cat_cols:
            # Columns might be in a different format; try all non-metadata cols
            meta = {"survey_input", "processing_status", "failed_models",
                    "pdf_path", "page_index", "image_path"}
            cat_cols = [c for c in cols if c not in meta
                        and not c.endswith("_agreement")]
        col_to_cat = {c: c for c in cat_cols}

    # --- write results back to Stata ---
    for row_i in range(len(result_df)):
        stata_obs = obs_map[row_i]
        row = result_df.iloc[row_i]

        # Find which category is assigned (value == 1)
        assigned = ""
        for cc in cat_cols:
            try:
                if int(row[cc]) == 1:
                    assigned = col_to_cat.get(cc, cc)
                    break
            except (ValueError, TypeError):
                continue

        if assigned:
            Data.setStr(gen_idx, assigned, stata_obs)

    SFIToolkit.displayln("{txt}Python classification complete.")
end
