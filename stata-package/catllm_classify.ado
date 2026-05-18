*! version 2.0.0  18may2026
*! catllm_classify -- Classify text into categories using LLMs

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
            TWOSTEPclassify(string)                             ///
            PROMPTtune(integer 0)                               ///
            TUNEiterations(integer 3)                           ///
            TUNEui(string)                                      ///
            TUNEoptimize(string)                                ///
            DOMain(string)                                      ///
            PYOptions(string asis)                              ///
            REPLACE                                             ///
        ]

    * ----- defaults -----
    * generate() is now a PREFIX for one byte variable per category
    * (e.g. categories("Positive" "Negative") + generate(sent) -->
    *  sent_Positive, sent_Negative as 0/1 byte variables).
    if "`generate'" == "" local generate "cat"
    if "`model'"    == "" local model "gpt-4o"
    if "`provider'" == "" local provider "auto"
    if "`consensus'" == "" local consensus "majority"
    if "`failstrategy'" == "" local failstrategy "partial"

    * ----- validate -----
    confirm string variable `varlist'

    * Variable creation is deferred to the python: block because we don't
    * know the per-category sanitized names until we've parsed the
    * categories() string.  The `replace` flag is passed through.

    * ----- mark sample -----
    marksample touse, strok
    quietly count if `touse'
    local nobs = r(N)
    if `nobs' == 0 {
        di as error "no observations"
        exit 2000
    }
    di as txt "Classifying `nobs' observations..."

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
    local _catllm_twostep  "`twostepclassify'"
    local _catllm_pt       "`prompttune'"
    local _catllm_tuneiter "`tuneiterations'"
    local _catllm_tuneui   "`tuneui'"
    local _catllm_tuneopt  "`tuneoptimize'"
    local _catllm_domain   "`domain'"
    local _catllm_pyopts   `"`pyoptions'"'
    local _catllm_replace  "`replace'"

    if `creativity' == -1 {
        local _catllm_creat ""
    }
    else {
        local _catllm_creat "`creativity'"
    }

    * ----- call Python -----
    local _catllm_failed ""
    local _catllm_ret_vars ""
    local _catllm_ret_classified ""
    python: _catllm_do_classify()
    if "`_catllm_failed'" != "" {
        exit 198
    }

    * ----- return results -----
    * _catllm_ret_vars is set by python to a space-separated list of
    * indicator variables created.  _catllm_ret_classified is the count
    * of rows with at least one category marked 1.
    local genvars     "`_catllm_ret_vars'"
    local classified  "`_catllm_ret_classified'"
    return scalar N = `nobs'
    return scalar N_classified = `classified'
    return local prefix "`generate'"
    return local variables "`genvars'"
    return local model "`model'"
    return local provider "`provider'"

    di as txt ""
    di as txt "Classification complete."
    di as txt "  Observations: `nobs'"
    di as txt "  Classified:   `classified'"
    di as txt "  Variables:    {res}`genvars'"
end

python:
def _catllm_do_classify():
    """Thin wrapper over catstack.classify_labels.

    All helper logic (domain resolution, kwargs parsing, models parsing,
    short-label collapse, schema canary, column->label mapping) lives
    server-side in cat-stack >= 1.2.0.  This function does only:
      1. read inputs from Stata via sfi
      2. call one cat-stack function
      3. write one label per row back to Stata
    """
    from sfi import Data, Macro, SFIToolkit

    # --- read Stata locals ---
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
    twostep_s  = Macro.getLocal("_catllm_twostep")
    pt_n       = int(Macro.getLocal("_catllm_pt") or "0")
    tune_iter  = int(Macro.getLocal("_catllm_tuneiter") or "3")
    tune_ui    = Macro.getLocal("_catllm_tuneui")
    tune_opt   = Macro.getLocal("_catllm_tuneopt")
    domain     = Macro.getLocal("_catllm_domain")
    pyopts_str = Macro.getLocal("_catllm_pyopts")
    creat_str  = Macro.getLocal("_catllm_creat")

    replace_flag = Macro.getLocal("_catllm_replace") != ""

    # --- version guard: this .ado requires cat-stack >= 1.4.0 ---
    try:
        import cat_stack
        _v = tuple(int(x) for x in cat_stack.__version__.split(".")[:2])
        if _v < (1, 4):
            raise ImportError(
                "catllm_classify 2.0 requires cat-stack >= 1.4.0 "
                "(installed: " + cat_stack.__version__ + "). "
                "Run: catllm setup, upgrade"
            )
    except Exception as e:
        SFIToolkit.errprintln("{err}" + str(e))
        Macro.setLocal("_catllm_failed", "1")
        return

    # --- twostep: "true"/"false"/"" (unset) -> bool/None ---
    two_step_classify = None
    if twostep_s:
        s = twostep_s.strip().lower()
        if s in ("true", "yes", "1", "on"):
            two_step_classify = True
        elif s in ("false", "no", "0", "off"):
            two_step_classify = False
        else:
            SFIToolkit.errprintln(
                "{err}twostepclassify() must be true/false (got: '"
                + twostep_s + "')"
            )
            Macro.setLocal("_catllm_failed", "1")
            return

    creativity = float(creat_str) if creat_str else None

    # --- delegate string parsing + domain resolution to cat-stack ---
    try:
        module       = cat_stack.get_backend(domain)
        extra_kwargs = cat_stack.parse_kwargs_string(pyopts_str)
        models       = cat_stack.parse_models_string(models_str, api_key)
        import shlex
        try:
            categories = shlex.split(cats_str)
        except ValueError:
            categories = cats_str.split()
    except Exception as e:
        SFIToolkit.errprintln("{err}" + str(e))
        Macro.setLocal("_catllm_failed", "1")
        return

    # --- build per-category Stata variable names ---
    # generate() is now a PREFIX. For each user category we create one
    # byte variable: <prefix>_<safe_short_label>.  Sanitize: replace
    # non-[A-Za-z0-9_] chars with _, collapse repeats, strip edges,
    # prefix leading digits, truncate to 32 chars, disambiguate collisions.
    import re as _re
    def _safe_var_name(prefix, raw):
        s = _re.sub(r"[^A-Za-z0-9_]+", "_", str(raw)).strip("_")
        s = _re.sub(r"_+", "_", s) or "cat"
        if s[0].isdigit():
            s = "_" + s
        # leave room for "_N" suffix if we later collide
        base = (prefix + "_" + s)[:30]
        return base

    short_cats = [cat_stack.short_label(c) for c in categories]
    var_names = []
    used = set()
    for sc in short_cats:
        base = _safe_var_name(genname, sc)
        name = base
        i = 1
        while name in used:
            name = base + "_" + str(i)
            i += 1
        used.add(name)
        var_names.append(name)

    # --- replace existing vars or fail loudly if any already exist ---
    # sfi.Data.getVarIndex raises ValueError when the var doesn't exist,
    # so probe with a try/except instead of testing the return value.
    for vn in var_names:
        try:
            Data.getVarIndex(vn)
            exists = True
        except ValueError:
            exists = False
        if exists:
            if replace_flag:
                SFIToolkit.stata("quietly drop " + vn)
            else:
                SFIToolkit.errprintln(
                    "{err}variable " + vn + " already exists; pass `replace` to overwrite"
                )
                Macro.setLocal("_catllm_failed", "1")
                return

    # --- create the byte indicator variables ---
    for vn in var_names:
        Data.addVarByte(vn)
        # Default new values are missing (.); we want 0/1, so the writeback
        # loop below initializes 0 for in-sample rows.

    # --- read text column from Stata (sfi-only — stays) ---
    var_idx   = Data.getVarIndex(varname)
    touse_idx = Data.getVarIndex(touse)
    n         = Data.getObsTotal()

    texts, obs_map = [], []
    for i in range(n):
        if Data.getAt(touse_idx, i) == 1:
            val = Data.getAt(var_idx, i)
            texts.append(val if val else "")
            obs_map.append(i)

    if not texts:
        SFIToolkit.errprintln("{err}No valid observations found.")
        Macro.setLocal("_catllm_failed", "1")
        return

    # --- build kwargs ---
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
    if two_step_classify is not None:
        kwargs["two_step_classify"] = two_step_classify
    if pt_n > 0:
        kwargs["prompt_tune"] = pt_n
        kwargs["tune_iterations"] = tune_iter
        if tune_ui:
            kwargs["tune_ui"] = tune_ui
        if tune_opt:
            kwargs["tune_optimize"] = tune_opt
    if creativity is not None:
        kwargs["creativity"] = creativity
    kwargs.update(extra_kwargs)

    # --- ONE cat-stack call: classify and return one 0/1 list per category ---
    try:
        indicators = module.classify_indicators(short_labels=True, **kwargs)
    except Exception as e:
        SFIToolkit.errprintln(
            "{err}" + module.__name__ + ".classify_indicators() failed: " + str(e)
        )
        Macro.setLocal("_catllm_failed", "1")
        return

    # --- write per-category indicators back to Stata ---
    # Indicator dict is keyed by short_label; align with our var_names by
    # the same short_cats order.
    var_indices = [Data.getVarIndex(vn) for vn in var_names]
    classified_count = 0
    for row_i in range(len(obs_map)):
        stata_obs = obs_map[row_i]
        any_set = False
        for k, vn in enumerate(var_names):
            key = short_cats[k]
            vals = indicators.get(key, None)
            v = 1 if (vals is not None and row_i < len(vals) and vals[row_i] == 1) else 0
            Data.storeAt(var_indices[k], stata_obs, v)
            if v == 1:
                any_set = True
        if any_set:
            classified_count += 1

    # --- hand back to Stata ---
    Macro.setLocal("_catllm_ret_vars", " ".join(var_names))
    Macro.setLocal("_catllm_ret_classified", str(classified_count))

    SFIToolkit.displayln("{txt}Python classification complete.")
end
