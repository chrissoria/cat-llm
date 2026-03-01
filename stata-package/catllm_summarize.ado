*! catllm_summarize -- Summarize text or PDFs using LLMs
*! Version 1.0.0

program define catllm_summarize, rclass
    version 16

    syntax varname(string) [if] [in],                           ///
        APIkey(string)                                          ///
        [                                                       ///
            GENerate(name)                                      ///
            Model(string)                                       ///
            Provider(string)                                    ///
            Description(string asis)                            ///
            Instructions(string asis)                           ///
            MAXLength(integer 0)                                ///
            FOCus(string asis)                                  ///
            CREativity(real -1)                                 ///
            CHAINofthought                                      ///
            STEPback                                            ///
            CONText                                             ///
            MODels(string asis)                                 ///
            REPLACE                                             ///
        ]

    * ----- defaults -----
    if "`generate'" == "" local generate "_catllm_summ"
    if "`model'"    == "" local model "gpt-4o"
    if "`provider'" == "" local provider "auto"

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
    di as txt "Summarizing `nobs' observations..."

    * ----- create result variable (strL for long summaries) -----
    quietly gen strL `generate' = ""

    * ----- store parameters for Python -----
    local _catllm_var     "`varlist'"
    local _catllm_gen     "`generate'"
    local _catllm_key     "`apikey'"
    local _catllm_model   "`model'"
    local _catllm_prov    "`provider'"
    local _catllm_desc    `"`description'"'
    local _catllm_instr   `"`instructions'"'
    local _catllm_maxlen  "`maxlength'"
    local _catllm_focus   `"`focus'"'
    local _catllm_cot     "`chainofthought'"
    local _catllm_step    "`stepback'"
    local _catllm_ctx     "`context'"
    local _catllm_models  `"`models'"'
    local _catllm_touse   "`touse'"

    if `creativity' == -1 {
        local _catllm_creat ""
    }
    else {
        local _catllm_creat "`creativity'"
    }

    * ----- call Python -----
    python: _catllm_do_summarize()

    * ----- return results -----
    quietly count if `generate' != "" & `touse'
    local summarized = r(N)
    return scalar N = `nobs'
    return scalar N_summarized = `summarized'
    return local variable "`generate'"
    return local model "`model'"

    di as txt ""
    di as txt "Summarization complete."
    di as txt "  Observations: `nobs'"
    di as txt "  Summarized:   `summarized'"
    di as txt "  Variable:     {res}`generate'"
end

python:
def _catllm_do_summarize():
    from sfi import Data, Macro, SFIToolkit
    import catllm

    # --- read Stata parameters ---
    varname    = Macro.getLocal("_catllm_var")
    genname    = Macro.getLocal("_catllm_gen")
    api_key    = Macro.getLocal("_catllm_key")
    model      = Macro.getLocal("_catllm_model")
    provider   = Macro.getLocal("_catllm_prov")
    desc       = Macro.getLocal("_catllm_desc")
    instr      = Macro.getLocal("_catllm_instr")
    maxlen_str = Macro.getLocal("_catllm_maxlen")
    focus      = Macro.getLocal("_catllm_focus")
    cot        = Macro.getLocal("_catllm_cot") != ""
    stepback   = Macro.getLocal("_catllm_step") != ""
    context    = Macro.getLocal("_catllm_ctx") != ""
    models_str = Macro.getLocal("_catllm_models")
    touse      = Macro.getLocal("_catllm_touse")
    creat_str  = Macro.getLocal("_catllm_creat")

    creativity = float(creat_str) if creat_str else None
    maxlen = int(maxlen_str) if maxlen_str and int(maxlen_str) > 0 else None

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

    # --- read text data ---
    var_idx   = Data.getVarIndex(varname)
    gen_idx   = Data.getVarIndex(genname)
    touse_idx = Data.getVarIndex(touse)
    n         = Data.getObsTotal()

    texts = []
    obs_map = []
    for i in range(n):
        if Data.getNum(touse_idx, i) == 1:
            val = Data.getStr(var_idx, i)
            texts.append(val if val else "")
            obs_map.append(i)

    if not texts:
        SFIToolkit.errprintln("{err}No valid observations found.")
        return

    # --- call catllm.summarize ---
    kwargs = dict(
        input_data=texts,
        api_key=api_key,
        user_model=model,
        model_source=provider,
        description=desc,
        chain_of_thought=cot,
        step_back_prompt=stepback,
        context_prompt=context,
    )

    if instr:
        kwargs["instructions"] = instr
    if maxlen:
        kwargs["max_length"] = maxlen
    if focus:
        kwargs["focus"] = focus
    if creativity is not None:
        kwargs["creativity"] = creativity
    if models:
        kwargs["models"] = models

    try:
        result_df = catllm.summarize(**kwargs)
    except Exception as e:
        SFIToolkit.errprintln("{err}catllm.summarize() failed: " + str(e))
        return

    # --- write summaries back to Stata ---
    # Find the summary column
    summ_col = "summary"
    if summ_col not in result_df.columns:
        # Fall back to first non-input column
        for col in result_df.columns:
            if col not in ("survey_input", "processing_status", "failed_models",
                           "pdf_path", "page_index"):
                summ_col = col
                break

    for row_i in range(len(result_df)):
        stata_obs = obs_map[row_i]
        val = result_df.iloc[row_i].get(summ_col, "")
        if val and str(val) != "nan":
            Data.setStr(gen_idx, str(val), stata_obs)

    SFIToolkit.displayln("{txt}Python summarization complete.")
end
