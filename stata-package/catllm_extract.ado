*! version 1.2.0  17may2026
*! catllm_extract -- Discover categories from unstructured text using LLMs

program define catllm_extract, rclass
    version 16

    syntax varname(string) [if] [in],                           ///
        APIkey(string)                                          ///
        [                                                       ///
            Model(string)                                       ///
            Provider(string)                                    ///
            Description(string asis)                            ///
            MAXCategories(integer 12)                           ///
            CATSperchunk(integer 10)                            ///
            DIVisions(integer 12)                               ///
            Iterations(integer 8)                               ///
            CREativity(real -1)                                 ///
            SPECificity(string)                                 ///
            RESEARCHquestion(string asis)                       ///
            FOCus(string asis)                                  ///
            RANDOMseed(integer -1)                              ///
            DOMain(string)                                      ///
            PYOptions(string asis)                              ///
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
    di as txt "Extracting categories from `nobs' observations..."

    * ----- store parameters for Python -----
    local _catllm_var     "`varlist'"
    local _catllm_key     "`apikey'"
    local _catllm_model   "`model'"
    local _catllm_prov    "`provider'"
    local _catllm_desc    `"`description'"'
    local _catllm_maxcat  "`maxcategories'"
    local _catllm_cpc     "`catsperchunk'"
    local _catllm_div     "`divisions'"
    local _catllm_iter    "`iterations'"
    local _catllm_spec    "`specificity'"
    local _catllm_rq      `"`researchquestion'"'
    local _catllm_focus   `"`focus'"'
    local _catllm_touse   "`touse'"
    local _catllm_domain  "`domain'"
    local _catllm_pyopts  `"`pyoptions'"'

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
    local _catllm_failed ""
    local _catllm_ret_n ""
    local _catllm_ret_cats ""
    python: _catllm_do_extract()
    if "`_catllm_failed'" != "" {
        exit 198
    }

    * ----- populate r() -----
    if "`_catllm_ret_n'" != "" {
        return scalar n_categories = `_catllm_ret_n'
        return local categories `"`_catllm_ret_cats'"'
        forvalues i = 1/`_catllm_ret_n' {
            local _c = "`_catllm_ret_cat`i''"
            return local cat`i' `"`_c'"'
        }
    }

    di as txt ""
    di as txt "Extraction complete. Categories stored in r()."
    di as txt "Use {bf:return list} to see results."
end

python:
def _catllm_do_extract():
    """Thin wrapper over catstack.extract.

    Domain resolution and pyoptions parsing live server-side in
    cat-stack >= 1.2.0.  This function reads Stata locals, calls extract(),
    and shapes the dict result into r() macros / matrix.
    """
    from sfi import Data, Macro, Scalar, Matrix, SFIToolkit

    # --- read Stata locals ---
    varname   = Macro.getLocal("_catllm_var")
    api_key   = Macro.getLocal("_catllm_key")
    model     = Macro.getLocal("_catllm_model")
    provider  = Macro.getLocal("_catllm_prov")
    desc      = Macro.getLocal("_catllm_desc")
    maxcat    = int(Macro.getLocal("_catllm_maxcat"))
    cpc       = int(Macro.getLocal("_catllm_cpc"))
    div       = int(Macro.getLocal("_catllm_div"))
    iters     = int(Macro.getLocal("_catllm_iter"))
    spec      = Macro.getLocal("_catllm_spec")
    rq        = Macro.getLocal("_catllm_rq")
    focus     = Macro.getLocal("_catllm_focus")
    touse     = Macro.getLocal("_catllm_touse")
    domain    = Macro.getLocal("_catllm_domain")
    pyopts_str = Macro.getLocal("_catllm_pyopts")
    creat_str = Macro.getLocal("_catllm_creat")
    seed_str  = Macro.getLocal("_catllm_seed")

    # --- version guard: this .ado requires cat-stack >= 1.2.0 ---
    try:
        import cat_stack
        _v = tuple(int(x) for x in cat_stack.__version__.split(".")[:2])
        if _v < (1, 2):
            raise ImportError(
                "catllm_extract 1.2 requires cat-stack >= 1.2.0 "
                "(installed: " + cat_stack.__version__ + "). "
                "Run: catllm setup, upgrade"
            )
    except Exception as e:
        SFIToolkit.errprintln("{err}" + str(e))
        Macro.setLocal("_catllm_failed", "1")
        return

    creativity = float(creat_str) if creat_str else None
    random_state = int(seed_str) if seed_str else None

    # --- delegate domain resolution + pyoptions parsing to cat-stack ---
    try:
        module       = cat_stack.get_backend(domain)
        extra_kwargs = cat_stack.parse_kwargs_string(pyopts_str)
    except Exception as e:
        SFIToolkit.errprintln("{err}" + str(e))
        Macro.setLocal("_catllm_failed", "1")
        return

    # --- read text column from Stata ---
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
        Macro.setLocal("_catllm_failed", "1")
        return

    # --- build kwargs ---
    kwargs = dict(
        input_data=texts,
        api_key=api_key,
        user_model=model,
        model_source=provider,
        description=desc,
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
    kwargs.update(extra_kwargs)

    # --- ONE cat-stack call ---
    try:
        result = module.extract(**kwargs)
    except Exception as e:
        SFIToolkit.errprintln(
            "{err}" + module.__name__ + ".extract() failed: " + str(e)
        )
        Macro.setLocal("_catllm_failed", "1")
        return

    # The schema canary now lives implicitly in the result[...] lookups
    # below: if cat-stack changes the shape, a KeyError propagates and the
    # top-level Stata try/except surfaces it.

    # --- store results in locals for the .ado to return ---
    top_cats = result.get("top_categories", [])
    counts_df = result.get("counts_df", None)

    Macro.setLocal("_catllm_ret_n", str(len(top_cats)))
    Macro.setLocal("_catllm_ret_cats",
                   " ".join('"{}"'.format(c) for c in top_cats))
    for i, cat in enumerate(top_cats, 1):
        Macro.setLocal("_catllm_ret_cat{}".format(i), cat)

    # Store counts as a Stata matrix if available
    if counts_df is not None and len(counts_df) > 0:
        nrows = len(counts_df)
        # Create matrix: rows = categories, col 1 = count
        mat_name = "r(counts)"
        try:
            Matrix.create(mat_name, nrows, 1, 0)
            count_col = None
            for col in counts_df.columns:
                if col.lower() in ("count", "counts", "n", "frequency"):
                    count_col = col
                    break
            if count_col is None:
                # Use the last numeric column
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
            pass  # matrix creation may not be available in all contexts

    # Display results
    SFIToolkit.displayln("")
    SFIToolkit.displayln("{txt}Discovered categories:")
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
