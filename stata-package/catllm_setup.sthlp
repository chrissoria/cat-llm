{smcl}
{* *! version 2.0.0}{...}
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
{synopt:{cmd:pdf}}install the base package with PDF processing support{p_end}
{synopt:{cmdab:dom:ain(}{it:string}{cmd:)}}install a domain sub-package; {bf:all} for everything{p_end}
{synopt:{cmdab:up:grade}}upgrade an existing installation{p_end}
{synopt:{cmdab:ch:eck}}check installation status without installing{p_end}
{synoptline}


{marker description}{...}
{title:Description}

{pstd}
{cmd:catllm setup} installs the {bf:cat-stack} Python package (and optionally
its domain sub-packages) using pip into the Python environment that Stata
uses. This is a one-time setup step.

{pstd}
Requires Stata 16+ with Python integration configured. Check your Python
configuration with:

{phang2}{cmd:. python query}{p_end}


{marker options}{...}
{title:Options}

{phang}
{cmd:pdf} installs the base package with PDF processing extras ({bf:pymupdf},
{bf:pillow}). Only applies when {opt domain()} is empty.

{phang}
{opt domain(string)} installs a specific domain sub-package instead of the
base. Valid values: {bf:pol} (cat-pol), {bf:vader} (cat-vader),
{bf:ademic} (cat-ademic), {bf:survey} (cat-survey), {bf:cog} (cat-cog),
{bf:web} (cat-web), or {bf:all} (cat-stack plus all six domain packages).

{phang}
{opt upgrade} upgrades whichever package(s) are being installed.

{phang}
{opt check} reports the installed cat-stack version and probes each domain
package to report which are installed and which can be installed with
{cmd:catllm setup, domain({it:name})}.


{marker examples}{...}
{title:Examples}

{pstd}Install the base cat-stack backend:{p_end}
{phang2}{cmd:. catllm setup}{p_end}

{pstd}Install with PDF support:{p_end}
{phang2}{cmd:. catllm setup, pdf}{p_end}

{pstd}Install a single domain package:{p_end}
{phang2}{cmd:. catllm setup, domain(pol)}{p_end}

{pstd}Install everything in one shot:{p_end}
{phang2}{cmd:. catllm setup, domain(all)}{p_end}

{pstd}Upgrade an existing install:{p_end}
{phang2}{cmd:. catllm setup, upgrade}{p_end}

{pstd}Report installed versions and which domain packages are present:{p_end}
{phang2}{cmd:. catllm setup, check}{p_end}
