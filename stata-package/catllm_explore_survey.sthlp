{smcl}
{* *! version 1.0.0}{...}
{viewerjumpto "Syntax" "catllm_explore_survey##syntax"}{...}
{viewerjumpto "Description" "catllm_explore_survey##description"}{...}
{viewerjumpto "Examples" "catllm_explore_survey##examples"}{...}

{title:Title}

{phang}
{bf:catllm explore_survey} {hline 2} Raw category extraction from survey text for saturation analysis


{marker syntax}{...}
{title:Syntax}

{p 8 16 2}
{cmd:catllm explore_survey} {varname} [{cmd:if}] [{cmd:in}]{cmd:,}
{cmdab:api:key(}{it:string}{cmd:)}
[{it:options}]

{synoptset 28 tabbed}{...}
{synopthdr}
{synoptline}
{syntab:Required}
{synopt:{cmdab:api:key(}{it:string}{cmd:)}}API key for the LLM provider{p_end}

{syntab:Survey-specific}
{synopt:{cmdab:survey:question(}{it:string}{cmd:)}}the survey question that produced the responses{p_end}

{syntab:Output}
{synopt:{cmdab:sav:ing(}{it:filename}{cmd:)}}save raw categories to a .dta file{p_end}
{synoptline}


{marker description}{...}
{title:Description}

{pstd}
{cmd:catllm explore_survey} performs raw category extraction from survey text
without deduplication, using the {bf:cat_survey} Python package. See
{helpb catllm_explore:catllm explore} for full option details.


{marker examples}{...}
{title:Examples}

{pstd}Explore survey responses:{p_end}
{phang2}{cmd:. catllm explore_survey response, apikey($OPENAI_API_KEY) surveyquestion("What concerns you most?") saving(raw_cats)}{p_end}
