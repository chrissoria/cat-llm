*! catllm_extract_survey -- Discover categories from survey text using cat_survey
*! Version 1.0.0

program define catllm_extract_survey, rclass
    version 16

    syntax varname(string) [if] [in],                           ///
        APIkey(string)                                          ///
        [                                                       ///
            Model(string)                                       ///
            Provider(string)                                    ///
            Description(string asis)                            ///
            SURVEYquestion(string asis)                         ///
            MAXCategories(integer 12)                           ///
            CATSperchunk(integer 10)                            ///
            DIVisions(integer 12)                               ///
            Iterations(integer 8)                               ///
            CREativity(real -1)                                 ///
            SPECificity(string)                                 ///
            RESEARCHquestion(string asis)                       ///
            FOCus(string asis)                                  ///
            RANDOMseed(integer -1)                              ///
        ]

    * ----- defaults -----
    if "`model'"       == "" local model "gpt-4o"
    if "`provider'"    == "" local provider "auto"
    if "`specificity'" == "" local specificity "broad"

    * ----- validate -----
    confirm string variable `varlist'

    * ----- mark sample -----
    marksample touse, strok
    quietly count if `touse'
    local nobs = r(N)
    if `nobs' == 0 {
        di as error "no observations"
        exit 2000
    }
    di as txt "Extracting categories from `nobs' observations (survey domain)..."

    * ----- store parameters for Python -----
    local _catllm_var     "`varlist'"
    local _catllm_key     "`apikey'"
    local _catllm_model   "`model'"
    local _catllm_prov    "`provider'"
    local _catllm_desc    `"`description'"'
    local _catllm_survey  `"`surveyquestion'"'
    local _catllm_maxcat  "`maxcategories'"
    local _catllm_cpc     "`catsperchunk'"
    local _catllm_div     "`divisions'"
    local _catllm_iter    "`iterations'"
    local _catllm_spec    "`specificity'"
    local _catllm_rq      `"`researchquestion'"'
    local _catllm_focus   `"`focus'"'
    local _catllm_touse   "`touse'"

    if `creativity' == -1 {
        local _catllm_creat ""
    }
    else {
        local _catllm_creat "`creativity'"
    }

    if `randomseed' == -1 {
        local _catllm_seed ""
    }
    else {
        local _catllm_seed "`randomseed'"
    }

    * ----- call Python -----
    python: _catllm_do_extract_survey()

    * ----- display and return -----
    di as txt ""
    di as txt "Extraction complete (survey domain). Categories stored in r()."
    di as txt "Use {bf:return list} to see results."
end

python:
def _catllm_do_extract_survey():
    from sfi import Data, Macro, Scalar, Matrix, SFIToolkit
    import cat_survey

    # --- read Stata parameters ---
    varname   = Macro.getLocal("_catllm_var")
    api_key   = Macro.getLocal("_catllm_key")
    model     = Macro.getLocal("_catllm_model")
    provider  = Macro.getLocal("_catllm_prov")
    desc      = Macro.getLocal("_catllm_desc")
    survey_q  = Macro.getLocal("_catllm_survey")
    maxcat    = int(Macro.getLocal("_catllm_maxcat"))
    cpc       = int(Macro.getLocal("_catllm_cpc"))
    div       = int(Macro.getLocal("_catllm_div"))
    iters     = int(Macro.getLocal("_catllm_iter"))
    spec      = Macro.getLocal("_catllm_spec")
    rq        = Macro.getLocal("_catllm_rq")
    focus     = Macro.getLocal("_catllm_focus")
    touse     = Macro.getLocal("_catllm_touse")
    creat_str = Macro.getLocal("_catllm_creat")
    seed_str  = Macro.getLocal("_catllm_seed")

    creativity = float(creat_str) if creat_str else None
    random_state = int(seed_str) if seed_str else None

    # --- read text data from Stata ---
    var_idx   = Data.getVarIndex(varname)
    touse_idx = Data.getVarIndex(touse)
    n         = Data.getObsTotal()

    texts = []
    for i in range(n):
        if Data.getAt(touse_idx, i) == 1:
            val = Data.getAt(var_idx, i)
            if val:
                texts.append(val)

    if not texts:
        SFIToolkit.errprintln("{err}No valid text observations found.")
        return

    # --- call cat_survey.extract ---
    kwargs = dict(
        input_data=texts,
        api_key=api_key,
        survey_question=survey_q,
        description=desc,
        user_model=model,
        model_source=provider,
        max_categories=maxcat,
        categories_per_chunk=cpc,
        divisions=div,
        iterations=iters,
        specificity=spec,
    )

    if rq:
        kwargs["research_question"] = rq
    if focus:
        kwargs["focus"] = focus
    if creativity is not None:
        kwargs["creativity"] = creativity
    if random_state is not None:
        kwargs["random_state"] = random_state

    try:
        result = cat_survey.extract(**kwargs)
    except Exception as e:
        SFIToolkit.errprintln("{err}cat_survey.extract() failed: " + str(e))
        return

    # --- store results in r() ---
    top_cats = result.get("top_categories", [])
    counts_df = result.get("counts_df", None)

    Scalar.setValue("r(n_categories)", len(top_cats), vtype="hidden")

    for i, cat in enumerate(top_cats, 1):
        Macro.setGlobal("r(cat{})".format(i), cat)

    Macro.setGlobal("r(categories)", " ".join('"{}"'.format(c) for c in top_cats))

    if counts_df is not None and len(counts_df) > 0:
        nrows = len(counts_df)
        mat_name = "r(counts)"
        try:
            Matrix.create(mat_name, nrows, 1, 0)
            count_col = None
            for col in counts_df.columns:
                if col.lower() in ("count", "counts", "n", "frequency"):
                    count_col = col
                    break
            if count_col is None:
                for col in counts_df.columns:
                    if counts_df[col].dtype in ("int64", "float64", "int32", "float32"):
                        count_col = col
            if count_col:
                for i in range(nrows):
                    try:
                        Matrix.setVal(mat_name, i, 0, float(counts_df.iloc[i][count_col]))
                    except (ValueError, TypeError):
                        pass
        except Exception:
            pass

    # Display results
    SFIToolkit.displayln("")
    SFIToolkit.displayln("{txt}Discovered categories (survey domain):")
    for i, cat in enumerate(top_cats, 1):
        count_str = ""
        if counts_df is not None and len(counts_df) > i - 1:
            try:
                count_col = None
                for col in counts_df.columns:
                    if col.lower() in ("count", "counts", "n", "frequency"):
                        count_col = col
                        break
                if count_col:
                    count_str = "  (n={})".format(int(counts_df.iloc[i-1][count_col]))
            except Exception:
                pass
        SFIToolkit.displayln("  {res}" + str(i) + ". " + cat + count_str)
end
