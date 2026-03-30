{smcl}
{* *! version 1.0.0}{...}
{viewerjumpto "Syntax" "catllm_explore_social##syntax"}{...}
{viewerjumpto "Description" "catllm_explore_social##description"}{...}
{viewerjumpto "Examples" "catllm_explore_social##examples"}{...}

{title:Title}

{phang}
{bf:catllm explore_social} {hline 2} Raw category extraction from social media text for saturation analysis


{marker syntax}{...}
{title:Syntax}

{p 8 16 2}
{cmd:catllm explore_social} {varname} [{cmd:if}] [{cmd:in}]{cmd:,}
{cmdab:api:key(}{it:string}{cmd:)}
[{it:options}]

{synoptset 28 tabbed}{...}
{synopthdr}
{synoptline}
{syntab:Required}
{synopt:{cmdab:api:key(}{it:string}{cmd:)}}API key for the LLM provider{p_end}

{syntab:Social media source}
{synopt:{cmd:smsource(}{it:string}{cmd:)}}social media source{p_end}
{synopt:{cmd:smlimit(}{it:integer}{cmd:)}}max posts to fetch; default {bf:50}{p_end}
{synopt:{cmd:platform(}{it:string}{cmd:)}}platform name{p_end}
{synopt:{cmd:handle(}{it:string}{cmd:)}}user handle{p_end}
{synopt:{cmd:hashtags(}{it:string}{cmd:)}}hashtags to filter by{p_end}

{syntab:Output}
{synopt:{cmdab:sav:ing(}{it:filename}{cmd:)}}save raw categories to a .dta file{p_end}
{synoptline}


{marker description}{...}
{title:Description}

{pstd}
{cmd:catllm explore_social} performs raw category extraction from social media
text without deduplication, using the {bf:catvader} Python package. See
{helpb catllm_explore:catllm explore} for full option details.


{marker examples}{...}
{title:Examples}

{pstd}Explore social media posts:{p_end}
{phang2}{cmd:. catllm explore_social post_text, apikey($OPENAI_API_KEY) platform("reddit") saving(raw_cats)}{p_end}
