*! catllm_explore -- Raw category extraction for saturation analysis
*! Version 1.1.0

program define catllm_explore, rclass
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
            SAVing(string)                                      ///
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
    di as txt "Exploring categories from `nobs' observations..."
    di as txt "(raw extraction, no deduplication -- for saturation analysis)"

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
    local _catllm_saving  "`saving'"
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
    local _catllm_ret_n_raw ""
    local _catllm_ret_n_uniq ""
    local _catllm_ret_top ""
    local _catllm_ret_n_unique_cap ""
    python: _catllm_do_explore()
    if "`_catllm_failed'" != "" {
        exit 198
    }

    * ----- populate r() -----
    if "`_catllm_ret_n_raw'" != "" {
        return scalar n_raw = `_catllm_ret_n_raw'
        return scalar n_unique = `_catllm_ret_n_uniq'
        return local top_categories `"`_catllm_ret_top'"'
        if "`_catllm_ret_n_unique_cap'" != "" {
            forvalues i = 1/`_catllm_ret_n_unique_cap' {
                local _c = "`_catllm_ret_cat`i''"
                return local cat`i' `"`_c'"'
            }
        }
    }

    di as txt ""
    di as txt "Exploration complete. Raw categories stored in r()."
    di as txt "Use {bf:return list} to see results."
end

python:
def _catllm_resolve_backend(domain):
    """Return the python module to call. Empty domain -> cat_stack."""
    from sfi import SFIToolkit
    if not domain:
        try:
            import cat_stack
        except ImportError:
            SFIToolkit.errprintln(
                "{err}cat-stack is not installed. Run: catllm setup"
            )
            raise
        return cat_stack
    d = domain.lower().strip()
    pkg_map = {
        "pol":    ("cat_pol",    "cat-pol"),
        "vader":  ("catvader",   "cat-vader"),
        "ademic": ("catademic",  "cat-ademic"),
        "survey": ("cat_survey", "cat-survey"),
        "cog":    ("cat_cog",    "cat-cog"),
        "web":    ("catweb",     "cat-web"),
    }
    if d not in pkg_map:
        SFIToolkit.errprintln(
            "{err}Unknown domain: '" + domain + "'. "
            "Valid: pol, vader, ademic, survey, cog, web."
        )
        raise ValueError("unknown domain: " + domain)
    mod_name, pkg_name = pkg_map[d]
    try:
        return __import__(mod_name)
    except ImportError:
        SFIToolkit.errprintln(
            "{err}Domain package '" + pkg_name + "' is not installed. "
            "Run: catllm setup, domain(" + d + ")"
        )
        raise

def _catllm_parse_pyoptions(s):
    """Parse 'key=val, key=val' into a dict. Values run through ast.literal_eval."""
    import ast
    out = {}
    if not s or not s.strip():
        return out
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        s = s[1:-1]
    parts, buf, depth, quote = [], "", 0, None
    for ch in s:
        if quote:
            buf += ch
            if ch == quote:
                quote = None
            continue
        if ch in ('"', "'"):
            quote = ch
            buf += ch
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(buf)
            buf = ""
        else:
            buf += ch
    if buf:
        parts.append(buf)
    for piece in parts:
        if "=" not in piece:
            continue
        k, v = piece.split("=", 1)
        k, v = k.strip(), v.strip()
        if not k:
            continue
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out

def _catllm_do_explore():
    from sfi import Data, Macro, Scalar, SFIToolkit

    # --- read Stata parameters ---
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
    saving    = Macro.getLocal("_catllm_saving")
    domain    = Macro.getLocal("_catllm_domain")
    pyopts_str = Macro.getLocal("_catllm_pyopts")
    creat_str = Macro.getLocal("_catllm_creat")
    seed_str  = Macro.getLocal("_catllm_seed")

    creativity = float(creat_str) if creat_str else None
    random_state = int(seed_str) if seed_str else None

    try:
        module = _catllm_resolve_backend(domain)
    except Exception:
        Macro.setLocal("_catllm_failed", "1")
        return
    extra_kwargs = _catllm_parse_pyoptions(pyopts_str)

    # --- read text data ---
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

    # --- call catllm.explore ---
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

    try:
        raw_cats = module.explore(**kwargs)
    except Exception as e:
        SFIToolkit.errprintln("{err}" + module.__name__ + ".explore() failed: " + str(e))
        Macro.setLocal("_catllm_failed", "1")
        return

    # --- schema canary: confirm cat-stack returns an iterable of strings ---
    if raw_cats is None or isinstance(raw_cats, str) or not hasattr(raw_cats, "__iter__"):
        SFIToolkit.errprintln(
            "{err}Unexpected return shape from " + module.__name__
            + ".explore(): expected an iterable of category strings. "
            "Got: " + type(raw_cats).__name__ + ". "
            "This usually means cat-stack changed its output schema -- "
            "pin to a known-good version or report at "
            "https://github.com/chrissoria/cat-llm/issues."
        )
        Macro.setLocal("_catllm_failed", "1")
        return

    # --- store results in locals for the .ado to return ---
    unique_cats = list(set(raw_cats))
    capped = unique_cats[:100]

    from collections import Counter
    freq = Counter(raw_cats)
    top_20 = freq.most_common(20)

    Macro.setLocal("_catllm_ret_n_raw", str(len(raw_cats)))
    Macro.setLocal("_catllm_ret_n_uniq", str(len(unique_cats)))
    Macro.setLocal("_catllm_ret_top",
                   " ".join('"{}"'.format(c) for c, _ in top_20))
    Macro.setLocal("_catllm_ret_n_unique_cap", str(len(capped)))
    for i, cat in enumerate(capped, 1):
        Macro.setLocal("_catllm_ret_cat{}".format(i), cat)

    # Optionally save raw results to a new dataset
    if saving:
        import pandas as pd
        raw_df = pd.DataFrame({"raw_category": raw_cats})
        raw_df.to_stata(saving if saving.endswith(".dta") else saving + ".dta",
                        write_index=False)
        SFIToolkit.displayln("{txt}Raw categories saved to: {res}" + saving)

    # Display summary
    SFIToolkit.displayln("")
    SFIToolkit.displayln("{txt}Raw categories extracted: {res}" + str(len(raw_cats)))
    SFIToolkit.displayln("{txt}Unique categories:       {res}" + str(len(unique_cats)))
    SFIToolkit.displayln("")
    SFIToolkit.displayln("{txt}Top 10 most frequent:")
    for cat, count in top_20[:10]:
        SFIToolkit.displayln("  {res}" + cat + "  {txt}(n=" + str(count) + ")")
end
