{smcl}
{* *! version 1.0.0}{...}
{viewerjumpto "Syntax" "catllm_extract_social##syntax"}{...}
{viewerjumpto "Description" "catllm_extract_social##description"}{...}
{viewerjumpto "Examples" "catllm_extract_social##examples"}{...}

{title:Title}

{phang}
{bf:catllm extract_social} {hline 2} Discover categories from social media text using catvader


{marker syntax}{...}
{title:Syntax}

{p 8 16 2}
{cmd:catllm extract_social} {varname} [{cmd:if}] [{cmd:in}]{cmd:,}
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
{synopt:{cmd:smmonths(}{it:integer}{cmd:)}}fetch posts from last N months{p_end}
{synopt:{cmd:platform(}{it:string}{cmd:)}}platform name{p_end}
{synopt:{cmd:handle(}{it:string}{cmd:)}}user handle{p_end}
{synopt:{cmd:hashtags(}{it:string}{cmd:)}}hashtags to filter by{p_end}

{syntab:Extraction}
{synopt:{cmdab:maxc:ategories(}{it:integer}{cmd:)}}max categories to return; default {bf:12}{p_end}
{synopt:{cmdab:div:isions(}{it:integer}{cmd:)}}number of data chunks; default {bf:12}{p_end}
{synopt:{cmdab:it:erations(}{it:integer}{cmd:)}}number of passes; default {bf:8}{p_end}
{synoptline}


{marker description}{...}
{title:Description}

{pstd}
{cmd:catllm extract_social} discovers categories from social media text using
the {bf:catvader} Python package. See {helpb catllm_extract:catllm extract}
for full extraction option details.


{marker examples}{...}
{title:Examples}

{pstd}Discover categories from social media posts:{p_end}
{phang2}{cmd:. catllm extract_social post_text, apikey($OPENAI_API_KEY) platform("twitter")}{p_end}
{phang2}{cmd:. return list}{p_end}
