*! catllm_cerad -- Score CERAD drawn shapes using cat_cog
*! Version 1.0.0

program define catllm_cerad, rclass
    version 16

    syntax varname(string) [if] [in],                           ///
        SHAPE(string) APIkey(string)                            ///
        [                                                       ///
            GENerate(name)                                      ///
            Model(string)                                       ///
            Provider(string)                                    ///
            CREativity(real -1)                                 ///
            REPLACE                                             ///
        ]

    * ----- defaults -----
    if "`generate'" == "" local generate "_catllm_cerad"
    if "`model'"    == "" local model "gpt-4o"
    if "`provider'" == "" local provider "auto"

    * ----- validate shape -----
    local shape = lower("`shape'")
    if !inlist("`shape'", "circle", "diamond", "rectangles", "cube") {
        di as error "shape() must be one of: circle, diamond, rectangles, cube"
        exit 198
    }

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
    di as txt "Scoring `nobs' CERAD drawn shapes (shape: `shape')..."

    * ----- create result variable -----
    quietly gen str244 `generate' = ""

    * ----- store parameters for Python -----
    local _catllm_var      "`varlist'"
    local _catllm_gen      "`generate'"
    local _catllm_shape    "`shape'"
    local _catllm_key      "`apikey'"
    local _catllm_model    "`model'"
    local _catllm_provider "`provider'"
    local _catllm_touse    "`touse'"
    local _catllm_nobs     "`nobs'"

    if `creativity' == -1 {
        local _catllm_creat ""
    }
    else {
        local _catllm_creat "`creativity'"
    }

    * ----- call Python -----
    python: _catllm_do_cerad()

    * ----- return results -----
    quietly count if `generate' != "" & `touse'
    local scored = r(N)
    return scalar N = `nobs'
    return scalar N_scored = `scored'
    return local variable "`generate'"
    return local model "`model'"
    return local provider "`provider'"
    return local shape "`shape'"

    di as txt ""
    di as txt "CERAD scoring complete."
    di as txt "  Observations: `nobs'"
    di as txt "  Scored:       `scored'"
    di as txt "  Shape:        `shape'"
    di as txt "  Variable:     {res}`generate'"
end

python:
def _catllm_do_cerad():
    from sfi import Data, Macro, SFIToolkit
    import cat_cog

    # --- read Stata parameters ---
    varname    = Macro.getLocal("_catllm_var")
    genname    = Macro.getLocal("_catllm_gen")
    shape      = Macro.getLocal("_catllm_shape")
    api_key    = Macro.getLocal("_catllm_key")
    model      = Macro.getLocal("_catllm_model")
    provider   = Macro.getLocal("_catllm_provider")
    touse      = Macro.getLocal("_catllm_touse")
    creat_str  = Macro.getLocal("_catllm_creat")

    creativity = float(creat_str) if creat_str else None

    # --- read image paths from Stata ---
    var_idx   = Data.getVarIndex(varname)
    gen_idx   = Data.getVarIndex(genname)
    touse_idx = Data.getVarIndex(touse)
    n         = Data.getObsTotal()

    obs_list = []
    for i in range(n):
        if Data.getAt(touse_idx, i) == 1:
            val = Data.getAt(var_idx, i)
            obs_list.append((i, val if val else ""))

    if not obs_list:
        SFIToolkit.errprintln("{err}No valid observations found.")
        return

    # --- score each image ---
    scored = 0
    for stata_obs, image_path in obs_list:
        if not image_path:
            continue

        kwargs = dict(
            shape=shape,
            image_input=image_path,
            api_key=api_key,
            user_model=model,
            model_source=provider,
        )

        if creativity is not None:
            kwargs["creativity"] = creativity

        try:
            result = cat_cog.cerad_drawn_score(**kwargs)
            # result may be a DataFrame or a scalar; convert to string
            if hasattr(result, 'to_string'):
                score_str = str(result.iloc[0, 0]) if len(result) > 0 else ""
            else:
                score_str = str(result)
            if score_str and score_str != "nan":
                Data.storeAt(gen_idx, stata_obs, score_str)
                scored += 1
        except Exception as e:
            SFIToolkit.errprintln("{err}cat_cog.cerad_drawn_score() failed for obs "
                                  + str(stata_obs + 1) + ": " + str(e))
            continue

    SFIToolkit.displayln("{txt}Python CERAD scoring complete. Scored: " + str(scored))
end
