*! catllm_setup -- Install the catllm Python backend
*! Version 1.0.0

program define catllm_setup
    version 16

    syntax [, PDF UPgrade CHeck]

    * Check Python availability
    capture python query
    if _rc {
        di as error "Python is not available in this Stata installation."
        di as error "Stata 16+ with Python integration is required."
        di as error "See {bf:help python} for setup instructions."
        exit 198
    }

    if "`check'" != "" {
        di as txt "Checking catllm installation..."
        python: _catllm_check()
        exit
    }

    * Install catllm via pip
    if "`pdf'" != "" {
        di as txt "Installing catllm with PDF support..."
        local pkg "catllm[pdf]"
    }
    else {
        di as txt "Installing catllm..."
        local pkg "catllm"
    }

    if "`upgrade'" != "" {
        local upflag "--upgrade"
    }
    else {
        local upflag ""
    }

    python: _catllm_pip_install()

    di as txt ""
    di as txt "Verifying installation..."
    python: _catllm_check()
end

python:
def _catllm_pip_install():
    import subprocess, sys
    from sfi import Macro

    pkg = Macro.getLocal("pkg")
    upflag = Macro.getLocal("upflag")

    cmd = [sys.executable, "-m", "pip", "install"]
    if upflag:
        cmd.append("--upgrade")
    cmd.append(pkg)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            from sfi import SFIToolkit
            SFIToolkit.displayln("{txt}pip install completed successfully.")
        else:
            from sfi import SFIToolkit
            SFIToolkit.errprintln("{err}pip install failed:")
            SFIToolkit.errprintln(result.stderr)
    except Exception as e:
        from sfi import SFIToolkit
        SFIToolkit.errprintln("{err}Installation error: " + str(e))

def _catllm_check():
    from sfi import SFIToolkit
    try:
        import catllm
        version = getattr(catllm, '__version__', 'unknown')
        SFIToolkit.displayln("{txt}catllm version: {res}" + version)
        SFIToolkit.displayln("{txt}Status: {res}OK")
    except ImportError:
        SFIToolkit.errprintln("{err}catllm is not installed.")
        SFIToolkit.errprintln("{err}Run {bf:catllm setup} to install it.")
end
