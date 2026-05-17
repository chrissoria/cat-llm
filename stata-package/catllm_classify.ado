*! catllm_classify -- Classify text into categories using LLMs
*! Version 1.1.0

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
            DOMain(string)                                      ///
            PYOptions(string asis)                              ///
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
    local _catllm_domain   "`domain'"
    local _catllm_pyopts   `"`pyoptions'"'

    if `creativity' == -1 {
        local _catllm_creat ""
    }
    else {
        local _catllm_creat "`creativity'"
    }

    * ----- call Python -----
    local _catllm_failed ""
    python: _catllm_do_classify()
    if "`_catllm_failed'" != "" {
        exit 198
    }

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

def _catllm_do_classify():
    from sfi import Data, Macro, SFIToolkit

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
    domain     = Macro.getLocal("_catllm_domain")
    pyopts_str = Macro.getLocal("_catllm_pyopts")
    creat_str  = Macro.getLocal("_catllm_creat")

    creativity = float(creat_str) if creat_str else None

    try:
        module = _catllm_resolve_backend(domain)
    except Exception:
        Macro.setLocal("_catllm_failed", "1")
        return
    extra_kwargs = _catllm_parse_pyoptions(pyopts_str)

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

    # --- read text data from Stata ---
    var_idx   = Data.getVarIndex(varname)
    gen_idx   = Data.getVarIndex(genname)
    touse_idx = Data.getVarIndex(touse)
    n         = Data.getObsTotal()

    texts = []
    obs_map = []  # maps position in texts[] back to Stata obs number
    for i in range(n):
        if Data.getAt(touse_idx, i) == 1:
            val = Data.getAt(var_idx, i)
            texts.append(val if val else "")
            obs_map.append(i)

    if not texts:
        SFIToolkit.errprintln("{err}No valid observations found.")
        Macro.setLocal("_catllm_failed", "1")
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

    kwargs.update(extra_kwargs)

    try:
        result_df = module.classify(**kwargs)
    except Exception as e:
        SFIToolkit.errprintln("{err}" + module.__name__ + ".classify() failed: " + str(e))
        Macro.setLocal("_catllm_failed", "1")
        return

    # --- schema canary: confirm cat-stack's return shape still matches ---
    # Single-model results have category_N columns; ensemble has *_consensus.
    # If neither family exists, the underlying schema has changed and this
    # wrapper can't safely map columns back to user category names.
    import re as _re
    _canary = _re.compile(r"^category_\d+(_consensus)?$")
    if not any(_canary.match(c) for c in result_df.columns):
        SFIToolkit.errprintln(
            "{err}Unexpected return shape from " + module.__name__
            + ".classify(): no category_N or category_N_consensus columns. "
            "Columns: " + ", ".join(map(str, result_df.columns)) + ". "
            "This usually means cat-stack changed its output schema -- "
            "pin to a known-good version or report at "
            "https://github.com/chrissoria/cat-llm/issues."
        )
        Macro.setLocal("_catllm_failed", "1")
        return

    # --- determine classification per row ---
    # Single model: columns are category names with 0/1
    # Ensemble: columns are {cat}_consensus with 0/1
    cols = list(result_df.columns)

    # Try consensus columns first (ensemble mode)
    import re
    consensus_cols = [c for c in cols if c.endswith("_consensus")]
    if consensus_cols:
        # cat-stack emits category_N_consensus -- map by trailing index so
        # the assigned label is the user's category text, not "category_N".
        # Fall back to the base name if the column doesn't match the pattern
        # (e.g. older cat-stack emitted Healthcare_consensus directly).
        pat = re.compile(r"^category_(\d+)_consensus$")
        indexed = []
        other = []
        col_to_cat = {}
        for cc in consensus_cols:
            m = pat.match(cc)
            if m:
                indexed.append((int(m.group(1)), cc))
            else:
                other.append(cc)
                col_to_cat[cc] = cc.rsplit("_consensus", 1)[0]
        indexed.sort()
        for n, cc in indexed:
            col_to_cat[cc] = categories[n-1] if n-1 < len(categories) else cc
        cat_cols = [cc for _, cc in indexed] + other
    else:
        # Single model: cat-stack emits columns named category_1, category_2, ...
        # Match positively on that pattern; sort by the trailing index so
        # the order is stable regardless of DataFrame column order.
        pat = re.compile(r"^category_(\d+)$")
        indexed = []
        for c in cols:
            m = pat.match(c)
            if m:
                indexed.append((int(m.group(1)), c))
        indexed.sort()
        cat_cols = [c for _, c in indexed]
        col_to_cat = {}
        for i, cc in enumerate(cat_cols):
            col_to_cat[cc] = categories[i] if i < len(categories) else cc

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
            Data.storeAt(gen_idx, stata_obs, assigned)

    SFIToolkit.displayln("{txt}Python classification complete.")
end
