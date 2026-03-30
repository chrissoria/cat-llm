{smcl}
{* *! version 1.0.0}{...}
{viewerjumpto "Syntax" "catllm_extract_academic##syntax"}{...}
{viewerjumpto "Description" "catllm_extract_academic##description"}{...}
{viewerjumpto "Examples" "catllm_extract_academic##examples"}{...}

{title:Title}

{phang}
{bf:catllm extract_academic} {hline 2} Discover categories from academic text using catademic


{marker syntax}{...}
{title:Syntax}

{p 8 16 2}
{cmd:catllm extract_academic} {varname} [{cmd:if}] [{cmd:in}]{cmd:,}
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
{synopt:{cmdab:date:from(}{it:string}{cmd:)}}start date (YYYY-MM-DD){p_end}
{synopt:{cmdab:date:to(}{it:string}{cmd:)}}end date (YYYY-MM-DD){p_end}

{syntab:Extraction}
{synopt:{cmdab:maxc:ategories(}{it:integer}{cmd:)}}max categories; default {bf:12}{p_end}
{synopt:{cmdab:div:isions(}{it:integer}{cmd:)}}number of chunks; default {bf:12}{p_end}
{synopt:{cmdab:it:erations(}{it:integer}{cmd:)}}number of passes; default {bf:8}{p_end}
{synoptline}


{marker description}{...}
{title:Description}

{pstd}
{cmd:catllm extract_academic} discovers categories from academic text using
the {bf:catademic} Python package. See {helpb catllm_extract:catllm extract}
for full extraction option details.


{marker examples}{...}
{title:Examples}

{pstd}Discover categories from paper abstracts:{p_end}
{phang2}{cmd:. catllm extract_academic abstract, apikey($OPENAI_API_KEY) journalfield("sociology") maxcategories(10)}{p_end}
{phang2}{cmd:. return list}{p_end}
