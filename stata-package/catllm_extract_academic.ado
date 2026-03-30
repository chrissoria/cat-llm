*! catllm_extract_academic -- Discover categories from academic text using catademic
*! Version 1.0.0

program define catllm_extract_academic, rclass
    version 16

    syntax varname(string) [if] [in],                           ///
        APIkey(string)                                          ///
        [                                                       ///
            Model(string)                                       ///
            Provider(string)                                    ///
            Description(string asis)                            ///
            JOURNALname(string)                                 ///
            JOURNALissn(string)                                 ///
            JOURNALfield(string)                                ///
            TOPICname(string asis)                               ///
            PAPERlimit(integer 50)                              ///
            POLITEemail(string)                                 ///
            DATEfrom(string)                                    ///
            DATEto(string)                                      ///
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
    di as txt "Extracting categories from `nobs' observations (academic domain)..."

    * ----- store parameters for Python -----
    local _catllm_var     "`varlist'"
    local _catllm_key     "`apikey'"
    local _catllm_model   "`model'"
    local _catllm_prov    "`provider'"
    local _catllm_desc    `"`description'"'
    local _catllm_jname   "`journalname'"
    local _catllm_jissn   "`journalissn'"
    local _catllm_jfield  "`journalfield'"
    local _catllm_topic   `"`topicname'"'
    local _catllm_plimit  "`paperlimit'"
    local _catllm_polite  "`politeemail'"
    local _catllm_datefrom "`datefrom'"
    local _catllm_dateto  "`dateto'"
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
    python: _catllm_do_extract_academic()

    * ----- display and return -----
    di as txt ""
    di as txt "Extraction complete (academic domain). Categories stored in r()."
    di as txt "Use {bf:return list} to see results."
end

python:
def _catllm_do_extract_academic():
    from sfi import Data, Macro, Scalar, Matrix, SFIToolkit
    import catademic

    # --- read Stata parameters ---
    varname   = Macro.getLocal("_catllm_var")
    api_key   = Macro.getLocal("_catllm_key")
    model     = Macro.getLocal("_catllm_model")
    provider  = Macro.getLocal("_catllm_prov")
    desc      = Macro.getLocal("_catllm_desc")
    j_name    = Macro.getLocal("_catllm_jname")
    j_issn    = Macro.getLocal("_catllm_jissn")
    j_field   = Macro.getLocal("_catllm_jfield")
    topic     = Macro.getLocal("_catllm_topic")
    p_limit   = int(Macro.getLocal("_catllm_plimit") or "50")
    polite    = Macro.getLocal("_catllm_polite")
    date_from = Macro.getLocal("_catllm_datefrom")
    date_to   = Macro.getLocal("_catllm_dateto")
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

    # --- call catademic.extract ---
    kwargs = dict(
        input_data=texts,
        api_key=api_key,
        description=desc,
        user_model=model,
        model_source=provider,
        max_categories=maxcat,
        categories_per_chunk=cpc,
        divisions=div,
        iterations=iters,
        specificity=spec,
    )

    if j_name:
        kwargs["journal_name"] = j_name
    if j_issn:
        kwargs["journal_issn"] = j_issn
    if j_field:
        kwargs["journal_field"] = j_field
    if topic:
        kwargs["topic_name"] = topic
    if p_limit > 0:
        kwargs["paper_limit"] = p_limit
    if polite:
        kwargs["polite_email"] = polite
    if date_from:
        kwargs["date_from"] = date_from
    if date_to:
        kwargs["date_to"] = date_to
    if rq:
        kwargs["research_question"] = rq
    if focus:
        kwargs["focus"] = focus
    if creativity is not None:
        kwargs["creativity"] = creativity
    if random_state is not None:
        kwargs["random_state"] = random_state

    try:
        result = catademic.extract(**kwargs)
    except Exception as e:
        SFIToolkit.errprintln("{err}catademic.extract() failed: " + str(e))
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
    SFIToolkit.displayln("{txt}Discovered categories (academic domain):")
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
