{smcl}
{* *! version 1.0.0}{...}
{viewerjumpto "Syntax" "catllm_extract_survey##syntax"}{...}
{viewerjumpto "Description" "catllm_extract_survey##description"}{...}
{viewerjumpto "Options" "catllm_extract_survey##options"}{...}
{viewerjumpto "Examples" "catllm_extract_survey##examples"}{...}

{title:Title}

{phang}
{bf:catllm extract_survey} {hline 2} Discover categories from survey text using cat_survey


{marker syntax}{...}
{title:Syntax}

{p 8 16 2}
{cmd:catllm extract_survey} {varname} [{cmd:if}] [{cmd:in}]{cmd:,}
{cmdab:api:key(}{it:string}{cmd:)}
[{it:options}]

{synoptset 28 tabbed}{...}
{synopthdr}
{synoptline}
{syntab:Required}
{synopt:{cmdab:api:key(}{it:string}{cmd:)}}API key for the LLM provider{p_end}

{syntab:Survey-specific}
{synopt:{cmdab:survey:question(}{it:string}{cmd:)}}the survey question that produced the responses{p_end}

{syntab:Model}
{synopt:{cmdab:mod:el(}{it:string}{cmd:)}}model name; default {bf:gpt-4o}{p_end}
{synopt:{cmdab:prov:ider(}{it:string}{cmd:)}}provider; default {bf:auto}{p_end}

{syntab:Extraction}
{synopt:{cmdab:maxc:ategories(}{it:integer}{cmd:)}}max categories to return; default {bf:12}{p_end}
{synopt:{cmdab:div:isions(}{it:integer}{cmd:)}}number of data chunks; default {bf:12}{p_end}
{synopt:{cmdab:it:erations(}{it:integer}{cmd:)}}number of passes; default {bf:8}{p_end}
{synopt:{cmdab:spec:ificity(}{it:string}{cmd:)}}broad or specific; default {bf:broad}{p_end}
{synoptline}


{marker description}{...}
{title:Description}

{pstd}
{cmd:catllm extract_survey} discovers categories from survey response text using
the {bf:cat_survey} Python package. Like {helpb catllm_extract:catllm extract}
but optimized for survey data with a {opt surveyquestion()} option.


{marker examples}{...}
{title:Examples}

{pstd}Discover categories from survey responses:{p_end}
{phang2}{cmd:. catllm extract_survey response, apikey($OPENAI_API_KEY) surveyquestion("Why did you move to this neighborhood?")}{p_end}
{phang2}{cmd:. return list}{p_end}
