*! catllm_classify_social -- Classify social media text using catvader
*! Version 1.0.0

program define catllm_classify_social, rclass
    version 16

    syntax varname(string) [if] [in],                           ///
        Categories(string asis) APIkey(string)                  ///
        [                                                       ///
            GENerate(name)                                      ///
            Model(string)                                       ///
            Provider(string)                                    ///
            Description(string asis)                            ///
            SMSource(string)                                    ///
            SMHandle(string)                                    ///
            SMLimit(integer 50)                                 ///
            SMMonths(integer 0)                                 ///
            SMDays(integer 0)                                   ///
            PLATFORM(string)                                    ///
            HANDLE(string)                                      ///
            HASHTAGS(string asis)                                ///
            FEEDquestion(string asis)                            ///
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
    di as txt "Classifying `nobs' observations (social media domain)..."

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
    local _catllm_smsrc    "`smsource'"
    local _catllm_smhand   "`smhandle'"
    local _catllm_smlim    "`smlimit'"
    local _catllm_smmon    "`smmonths'"
    local _catllm_smdays   "`smdays'"
    local _catllm_platform "`platform'"
    local _catllm_handle   "`handle'"
    local _catllm_hashtags `"`hashtags'"'
    local _catllm_feedq    `"`feedquestion'"'
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
    python: _catllm_do_classify_social()

    * ----- return results -----
    quietly count if `generate' != "" & `touse'
    local classified = r(N)
    return scalar N = `nobs'
    return scalar N_classified = `classified'
    return local variable "`generate'"
    return local model "`model'"
    return local provider "`provider'"

    di as txt ""
    di as txt "Classification complete (social media domain)."
    di as txt "  Observations: `nobs'"
    di as txt "  Classified:   `classified'"
    di as txt "  Variable:     {res}`generate'"
end

python:
def _catllm_do_classify_social():
    from sfi import Data, Macro, SFIToolkit
    import catvader

    # --- read Stata parameters ---
    varname    = Macro.getLocal("_catllm_var")
    genname    = Macro.getLocal("_catllm_gen")
    cats_str   = Macro.getLocal("_catllm_cats")
    api_key    = Macro.getLocal("_catllm_key")
    model      = Macro.getLocal("_catllm_model")
    provider   = Macro.getLocal("_catllm_provider")
    desc       = Macro.getLocal("_catllm_desc")
    sm_source  = Macro.getLocal("_catllm_smsrc")
    sm_handle  = Macro.getLocal("_catllm_smhand")
    sm_limit   = int(Macro.getLocal("_catllm_smlim") or "50")
    sm_months  = int(Macro.getLocal("_catllm_smmon") or "0")
    sm_days    = int(Macro.getLocal("_catllm_smdays") or "0")
    platform   = Macro.getLocal("_catllm_platform")
    handle     = Macro.getLocal("_catllm_handle")
    hashtags_s = Macro.getLocal("_catllm_hashtags")
    feed_q     = Macro.getLocal("_catllm_feedq")
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
    import shlex
    try:
        categories = shlex.split(cats_str)
    except ValueError:
        categories = cats_str.split()

    # --- parse hashtags ---
    hashtags = None
    if hashtags_s:
        try:
            hashtags = shlex.split(hashtags_s)
        except ValueError:
            hashtags = hashtags_s.split()

    # --- parse models for ensemble ---
    models = None
    if models_str:
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
    obs_map = []
    for i in range(n):
        if Data.getAt(touse_idx, i) == 1:
            val = Data.getAt(var_idx, i)
            texts.append(val if val else "")
            obs_map.append(i)

    if not texts:
        SFIToolkit.errprintln("{err}No valid observations found.")
        return

    # --- call catvader.classify ---
    kwargs = dict(
        input_data=texts,
        categories=categories,
        api_key=api_key,
        user_model=model,
        model_source=provider,
        description=desc,
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
    )

    if sm_source:
        kwargs["sm_source"] = sm_source
    if sm_handle:
        kwargs["sm_handle"] = sm_handle
    if sm_limit > 0:
        kwargs["sm_limit"] = sm_limit
    if sm_months > 0:
        kwargs["sm_months"] = sm_months
    if sm_days > 0:
        kwargs["sm_days"] = sm_days
    if platform:
        kwargs["platform"] = platform
    if handle:
        kwargs["handle"] = handle
    if hashtags:
        kwargs["hashtags"] = hashtags
    if feed_q:
        kwargs["feed_question"] = feed_q
    if models:
        kwargs["models"] = models
    if workers > 0:
        kwargs["max_workers"] = workers
    if creativity is not None:
        kwargs["creativity"] = creativity

    try:
        result_df = catvader.classify(**kwargs)
    except Exception as e:
        SFIToolkit.errprintln("{err}catvader.classify() failed: " + str(e))
        return

    # --- determine classification per row ---
    cols = list(result_df.columns)
    consensus_cols = [c for c in cols if c.endswith("_consensus")]
    if consensus_cols:
        cat_cols = consensus_cols
        col_to_cat = {}
        for cc in consensus_cols:
            base = cc.rsplit("_consensus", 1)[0]
            col_to_cat[cc] = base
    else:
        meta = {"input_data", "processing_status", "failed_models",
                "pdf_path", "page_index", "image_path"}
        cat_cols = [c for c in cols if c not in meta
                    and not c.endswith("_agreement")]
        col_to_cat = {}
        for i, cc in enumerate(cat_cols):
            col_to_cat[cc] = categories[i] if i < len(categories) else cc

    # --- write results back to Stata ---
    for row_i in range(len(result_df)):
        stata_obs = obs_map[row_i]
        row = result_df.iloc[row_i]
        assigned = ""
        for cc in cat_cols:
            try:
                if int(row[cc]) == 1:
                    assigned = col_to_cat.get(cc, cc)
                    break
            except (ValueError, TypeError):
                continue
        if assigned:
            Data.storeAt(gen_idx, stata_obs, assigned)

    SFIToolkit.displayln("{txt}Python classification complete (social media domain).")
end
