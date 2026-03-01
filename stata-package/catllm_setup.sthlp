{smcl}
{* *! version 1.0.0}{...}
{viewerjumpto "Syntax" "catllm_setup##syntax"}{...}
{viewerjumpto "Description" "catllm_setup##description"}{...}
{viewerjumpto "Options" "catllm_setup##options"}{...}
{viewerjumpto "Examples" "catllm_setup##examples"}{...}

{title:Title}

{phang}
{bf:catllm setup} {hline 2} Install or check the catllm Python backend


{marker syntax}{...}
{title:Syntax}

{p 8 16 2}
{cmd:catllm setup} [{cmd:,} {it:options}]

{synoptset 20 tabbed}{...}
{synopthdr}
{synoptline}
{synopt:{cmd:pdf}}install with PDF processing support{p_end}
{synopt:{cmdab:up:grade}}upgrade an existing installation{p_end}
{synopt:{cmdab:ch:eck}}check installation status without installing{p_end}
{synoptline}


{marker description}{...}
{title:Description}

{pstd}
{cmd:catllm setup} installs the {bf:catllm} Python package using pip into the
Python environment that Stata uses. This is a one-time setup step.

{pstd}
Requires Stata 16+ with Python integration configured. Check your Python
configuration with:

{phang2}{cmd:. python query}{p_end}


{marker options}{...}
{title:Options}

{phang}
{cmd:pdf} installs catllm with PDF processing extras ({bf:pymupdf}, {bf:pillow}).
Use this if you plan to process PDF files.

{phang}
{opt upgrade} upgrades an existing catllm installation to the latest version.

{phang}
{opt check} reports the installed version without installing or modifying anything.


{marker examples}{...}
{title:Examples}

{pstd}Install catllm:{p_end}
{phang2}{cmd:. catllm setup}{p_end}

{pstd}Install with PDF support:{p_end}
{phang2}{cmd:. catllm setup, pdf}{p_end}

{pstd}Upgrade to latest version:{p_end}
{phang2}{cmd:. catllm setup, upgrade}{p_end}

{pstd}Check current installation:{p_end}
{phang2}{cmd:. catllm setup, check}{p_end}
