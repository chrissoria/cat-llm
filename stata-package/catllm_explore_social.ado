*! catllm_explore_social -- Raw category extraction from social media using catvader
*! Version 1.0.0

program define catllm_explore_social, rclass
    version 16

    syntax varname(string) [if] [in],                           ///
        APIkey(string)                                          ///
        [                                                       ///
            Model(string)                                       ///
            Provider(string)                                    ///
            Description(string asis)                            ///
            SMSource(string)                                    ///
            SMLimit(integer 50)                                 ///
            SMMonths(integer 0)                                 ///
            PLATFORM(string)                                    ///
            HANDLE(string)                                      ///
            HASHTAGS(string asis)                                ///
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
    di as txt "Exploring categories from `nobs' observations (social media domain)..."
    di as txt "(raw extraction, no deduplication -- for saturation analysis)"

    * ----- store parameters for Python -----
    local _catllm_var     "`varlist'"
    local _catllm_key     "`apikey'"
    local _catllm_model   "`model'"
    local _catllm_prov    "`provider'"
    local _catllm_desc    `"`description'"'
    local _catllm_smsrc   "`smsource'"
    local _catllm_smlim   "`smlimit'"
    local _catllm_smmon   "`smmonths'"
    local _catllm_platform "`platform'"
    local _catllm_handle  "`handle'"
    local _catllm_hashtags `"`hashtags'"'
    local _catllm_maxcat  "`maxcategories'"
    local _catllm_cpc     "`catsperchunk'"
    local _catllm_div     "`divisions'"
    local _catllm_iter    "`iterations'"
    local _catllm_spec    "`specificity'"
    local _catllm_rq      `"`researchquestion'"'
    local _catllm_focus   `"`focus'"'
    local _catllm_touse   "`touse'"
    local _catllm_saving  "`saving'"

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
    python: _catllm_do_explore_social()

    * ----- display -----
    di as txt ""
    di as txt "Exploration complete (social media domain). Raw categories stored in r()."
    di as txt "Use {bf:return list} to see results."
end

python:
def _catllm_do_explore_social():
    from sfi import Data, Macro, Scalar, SFIToolkit
    import catvader

    # --- read Stata parameters ---
    varname   = Macro.getLocal("_catllm_var")
    api_key   = Macro.getLocal("_catllm_key")
    model     = Macro.getLocal("_catllm_model")
    provider  = Macro.getLocal("_catllm_prov")
    desc      = Macro.getLocal("_catllm_desc")
    sm_source = Macro.getLocal("_catllm_smsrc")
    sm_limit  = int(Macro.getLocal("_catllm_smlim") or "50")
    sm_months = int(Macro.getLocal("_catllm_smmon") or "0")
    platform  = Macro.getLocal("_catllm_platform")
    handle    = Macro.getLocal("_catllm_handle")
    hashtags_s = Macro.getLocal("_catllm_hashtags")
    maxcat    = int(Macro.getLocal("_catllm_maxcat"))
    cpc       = int(Macro.getLocal("_catllm_cpc"))
    div       = int(Macro.getLocal("_catllm_div"))
    iters     = int(Macro.getLocal("_catllm_iter"))
    spec      = Macro.getLocal("_catllm_spec")
    rq        = Macro.getLocal("_catllm_rq")
    focus     = Macro.getLocal("_catllm_focus")
    touse     = Macro.getLocal("_catllm_touse")
    saving    = Macro.getLocal("_catllm_saving")
    creat_str = Macro.getLocal("_catllm_creat")
    seed_str  = Macro.getLocal("_catllm_seed")

    creativity = float(creat_str) if creat_str else None
    random_state = int(seed_str) if seed_str else None

    # --- parse hashtags ---
    import shlex
    hashtags = None
    if hashtags_s:
        try:
            hashtags = shlex.split(hashtags_s)
        except ValueError:
            hashtags = hashtags_s.split()

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
        return

    # --- call catvader.explore ---
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

    if sm_source:
        kwargs["sm_source"] = sm_source
    if sm_limit > 0:
        kwargs["sm_limit"] = sm_limit
    if sm_months > 0:
        kwargs["sm_months"] = sm_months
    if platform:
        kwargs["platform"] = platform
    if handle:
        kwargs["handle"] = handle
    if hashtags:
        kwargs["hashtags"] = hashtags
    if rq:
        kwargs["research_question"] = rq
    if focus:
        kwargs["focus"] = focus
    if creativity is not None:
        kwargs["creativity"] = creativity
    if random_state is not None:
        kwargs["random_state"] = random_state

    try:
        raw_cats = catvader.explore(**kwargs)
    except Exception as e:
        SFIToolkit.errprintln("{err}catvader.explore() failed: " + str(e))
        return

    # --- store results in r() ---
    Scalar.setValue("r(n_raw)", len(raw_cats), vtype="hidden")

    unique_cats = list(set(raw_cats))
    Scalar.setValue("r(n_unique)", len(unique_cats), vtype="hidden")

    for i, cat in enumerate(unique_cats[:100], 1):
        Macro.setGlobal("r(cat{})".format(i), cat)

    from collections import Counter
    freq = Counter(raw_cats)
    top_20 = freq.most_common(20)
    Macro.setGlobal("r(top_categories)",
                   " ".join('"{}"'.format(c) for c, _ in top_20))

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
