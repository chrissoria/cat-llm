{smcl}
{* *! version 1.0.0}{...}
{viewerjumpto "Syntax" "catllm_explore_academic##syntax"}{...}
{viewerjumpto "Description" "catllm_explore_academic##description"}{...}
{viewerjumpto "Examples" "catllm_explore_academic##examples"}{...}

{title:Title}

{phang}
{bf:catllm explore_academic} {hline 2} Raw category extraction from academic text for saturation analysis


{marker syntax}{...}
{title:Syntax}

{p 8 16 2}
{cmd:catllm explore_academic} {varname} [{cmd:if}] [{cmd:in}]{cmd:,}
{cmdab:api:key(}{it:string}{cmd:)}
[{it:options}]

{synoptset 28 tabbed}{...}
{synopthdr}
{synoptline}
{syntab:Required}
{synopt:{cmdab:api:key(}{it:string}{cmd:)}}API key for the LLM provider{p_end}

{syntab:Academic source}
{synopt:{cmdab:journal:name(}{it:string}{cmd:)}}journal name{p_end}
{synopt:{cmdab:journal:issn(}{it:string}{cmd:)}}journal ISSN{p_end}
{synopt:{cmdab:journal:field(}{it:string}{cmd:)}}journal field/discipline{p_end}
{synopt:{cmdab:topic:name(}{it:string}{cmd:)}}topic for paper search{p_end}
{synopt:{cmdab:paper:limit(}{it:integer}{cmd:)}}max papers; default {bf:50}{p_end}
{synopt:{cmdab:polite:email(}{it:string}{cmd:)}}polite email for API access{p_end}

{syntab:Output}
{synopt:{cmdab:sav:ing(}{it:filename}{cmd:)}}save raw categories to a .dta file{p_end}
{synoptline}


{marker description}{...}
{title:Description}

{pstd}
{cmd:catllm explore_academic} performs raw category extraction from academic
text without deduplication, using the {bf:catademic} Python package. See
{helpb catllm_explore:catllm explore} for full option details.


{marker examples}{...}
{title:Examples}

{pstd}Explore academic abstracts:{p_end}
{phang2}{cmd:. catllm explore_academic abstract, apikey($OPENAI_API_KEY) journalfield("economics") saving(raw_cats)}{p_end}
