*! version 1.1.0  17may2026
*! catllm_summarize -- Summarize text or PDFs using LLMs

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
            DOMain(string)                                      ///
            PYOptions(string asis)                              ///
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
    local _catllm_domain  "`domain'"
    local _catllm_pyopts  `"`pyoptions'"'

    if `creativity' == -1 {
        local _catllm_creat ""
    }
    else {
        local _catllm_creat "`creativity'"
    }

    * ----- call Python -----
    local _catllm_failed ""
    python: _catllm_do_summarize()
    if "`_catllm_failed'" != "" {
        exit 198
    }

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

def _catllm_do_summarize():
    from sfi import Data, Macro, SFIToolkit

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
    domain     = Macro.getLocal("_catllm_domain")
    pyopts_str = Macro.getLocal("_catllm_pyopts")
    creat_str  = Macro.getLocal("_catllm_creat")

    creativity = float(creat_str) if creat_str else None
    maxlen = int(maxlen_str) if maxlen_str and int(maxlen_str) > 0 else None

    try:
        module = _catllm_resolve_backend(domain)
    except Exception:
        Macro.setLocal("_catllm_failed", "1")
        return
    extra_kwargs = _catllm_parse_pyoptions(pyopts_str)

    # --- parse models for ensemble ---
    models = None
    if models_str:
        # Stata's string-asis keeps the surrounding quotes literal -- strip
        # one balanced pair so the per-entry split doesn't capture them.
        models_str = models_str.strip()
        if len(models_str) >= 2 and models_str[0] == models_str[-1] \
                and models_str[0] in ('"', "'"):
            models_str = models_str[1:-1]
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
        if Data.getAt(touse_idx, i) == 1:
            val = Data.getAt(var_idx, i)
            texts.append(val if val else "")
            obs_map.append(i)

    if not texts:
        SFIToolkit.errprintln("{err}No valid observations found.")
        Macro.setLocal("_catllm_failed", "1")
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

    kwargs.update(extra_kwargs)

    try:
        result_df = module.summarize(**kwargs)
    except Exception as e:
        SFIToolkit.errprintln("{err}" + module.__name__ + ".summarize() failed: " + str(e))
        Macro.setLocal("_catllm_failed", "1")
        return

    # --- write summaries back to Stata ---
    # Find the summary column. Try the canonical name first, then any
    # non-metadata column. Fail loudly if neither path finds anything.
    summ_col = None
    if "summary" in result_df.columns:
        summ_col = "summary"
    else:
        for col in result_df.columns:
            if col not in ("input_index", "input_data", "processing_status",
                           "failed_models", "pdf_path", "page_index"):
                summ_col = col
                break

    if summ_col is None:
        SFIToolkit.errprintln(
            "{err}Unexpected return shape from " + module.__name__
            + ".summarize(): no 'summary' column and no fallback "
            "non-metadata column. Columns: "
            + ", ".join(map(str, result_df.columns)) + ". "
            "This usually means cat-stack changed its output schema -- "
            "pin to a known-good version or report at "
            "https://github.com/chrissoria/cat-llm/issues."
        )
        Macro.setLocal("_catllm_failed", "1")
        return

    for row_i in range(len(result_df)):
        stata_obs = obs_map[row_i]
        val = result_df.iloc[row_i].get(summ_col, "")
        if val and str(val) != "nan":
            Data.storeAt(gen_idx, stata_obs, str(val))

    SFIToolkit.displayln("{txt}Python summarization complete.")
end
