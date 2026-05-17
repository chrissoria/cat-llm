*! catllm_setup -- Install the cat-stack Python backend (and optional domain packages)
*! Version 2.0.0

program define catllm_setup
    version 16

    syntax [, PDF UPgrade CHeck DOMain(string)]

    capture python query
    if _rc {
        di as error "Python is not available in this Stata installation."
        di as error "Stata 16+ with Python integration is required."
        di as error "See {bf:help python} for setup instructions."
        exit 198
    }

    if "`check'" != "" {
        di as txt "Checking cat-stack installation..."
        python: _catllm_check()
        exit
    }

    local upflag = cond("`upgrade'" != "", "--upgrade", "")

    if "`domain'" == "" {
        local pkgs = cond("`pdf'" != "", "cat-stack[pdf]", "cat-stack")
        di as txt "Installing `pkgs'..."
    }
    else if "`domain'" == "all" {
        local pkgs = cond("`pdf'" != "", "cat-stack[pdf]", "cat-stack")
        local pkgs "`pkgs' cat-pol cat-vader cat-ademic cat-survey cat-cog cat-web"
        di as txt "Installing cat-stack and all domain packages..."
    }
    else {
        local dom = lower("`domain'")
        if !inlist("`dom'", "pol", "vader", "ademic", "survey", "cog", "web") {
            di as error "unknown domain: `domain'"
            di as error "valid: pol, vader, ademic, survey, cog, web, all"
            exit 198
        }
        local pkgs "cat-`dom'"
        di as txt "Installing `pkgs'..."
    }

    local _catllm_pkgs   "`pkgs'"
    local _catllm_upflag "`upflag'"
    python: _catllm_pip_install()

    di as txt ""
    di as txt "Verifying installation..."
    python: _catllm_check()
end

python:
def _catllm_pip_install():
    import subprocess, sys
    from sfi import Macro, SFIToolkit

    pkgs   = (Macro.getLocal("_catllm_pkgs") or "").split()
    upflag = Macro.getLocal("_catllm_upflag")

    cmd = [sys.executable, "-m", "pip", "install"]
    if upflag:
        cmd.append("--upgrade")
    cmd.extend(pkgs)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            SFIToolkit.displayln("{txt}pip install completed successfully.")
        else:
            SFIToolkit.errprintln("{err}pip install failed:")
            SFIToolkit.errprintln(result.stderr)
    except Exception as e:
        SFIToolkit.errprintln("{err}Installation error: " + str(e))

def _catllm_check():
    from sfi import SFIToolkit
    try:
        import cat_stack
        version = getattr(cat_stack, "__version__", "unknown")
        SFIToolkit.displayln("{txt}cat_stack version: {res}" + version)
        SFIToolkit.displayln("{txt}Status: {res}OK")
    except ImportError:
        SFIToolkit.errprintln("{err}cat_stack is not installed.")
        SFIToolkit.errprintln("{err}Run {bf:catllm setup} to install it.")
        return

    domains = [
        ("pol",    "cat_pol",    "cat-pol"),
        ("vader",  "catvader",   "cat-vader"),
        ("ademic", "catademic",  "cat-ademic"),
        ("survey", "cat_survey", "cat-survey"),
        ("cog",    "cat_cog",    "cat-cog"),
        ("web",    "catweb",     "cat-web"),
    ]
    SFIToolkit.displayln("")
    SFIToolkit.displayln("{txt}Domain packages:")
    for short, mod_name, pkg_name in domains:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "unknown")
            SFIToolkit.displayln("  {res}" + short.ljust(8) + " " + pkg_name.ljust(12) + " {txt}installed (" + ver + ")")
        except ImportError:
            SFIToolkit.displayln("  {txt}" + short.ljust(8) + " " + pkg_name.ljust(12) + " not installed -- catllm setup, domain(" + short + ")")
end
