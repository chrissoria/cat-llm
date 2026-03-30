{smcl}
{* *! version 1.0.0}{...}
{viewerjumpto "Syntax" "catllm_classify_social##syntax"}{...}
{viewerjumpto "Description" "catllm_classify_social##description"}{...}
{viewerjumpto "Options" "catllm_classify_social##options"}{...}
{viewerjumpto "Examples" "catllm_classify_social##examples"}{...}

{title:Title}

{phang}
{bf:catllm classify_social} {hline 2} Classify social media text into categories using catvader


{marker syntax}{...}
{title:Syntax}

{p 8 16 2}
{cmd:catllm classify_social} {varname} [{cmd:if}] [{cmd:in}]{cmd:,}
{cmdab:cat:egories(}{it:string}{cmd:)} {cmdab:api:key(}{it:string}{cmd:)}
[{it:options}]

{synoptset 28 tabbed}{...}
{synopthdr}
{synoptline}
{syntab:Required}
{synopt:{cmdab:cat:egories(}{it:string}{cmd:)}}category names in double quotes{p_end}
{synopt:{cmdab:api:key(}{it:string}{cmd:)}}API key for the LLM provider{p_end}

{syntab:Social media source}
{synopt:{cmd:smsource(}{it:string}{cmd:)}}social media source (e.g., "twitter", "reddit"){p_end}
{synopt:{cmd:smhandle(}{it:string}{cmd:)}}social media handle to fetch from{p_end}
{synopt:{cmd:smlimit(}{it:integer}{cmd:)}}max posts to fetch; default {bf:50}{p_end}
{synopt:{cmd:smmonths(}{it:integer}{cmd:)}}fetch posts from last N months{p_end}
{synopt:{cmd:smdays(}{it:integer}{cmd:)}}fetch posts from last N days{p_end}
{synopt:{cmd:platform(}{it:string}{cmd:)}}platform name{p_end}
{synopt:{cmd:handle(}{it:string}{cmd:)}}user handle{p_end}
{synopt:{cmd:hashtags(}{it:string}{cmd:)}}hashtags to filter by{p_end}
{synopt:{cmdab:feed:question(}{it:string}{cmd:)}}question context for the feed{p_end}

{syntab:Output}
{synopt:{cmdab:gen:erate(}{it:newvar}{cmd:)}}name for the classification variable; default {bf:_catllm_class}{p_end}
{synopt:{cmd:replace}}overwrite {it:generate} if it already exists{p_end}

{syntab:Model}
{synopt:{cmdab:mod:el(}{it:string}{cmd:)}}model name; default {bf:gpt-4o}{p_end}
{synopt:{cmdab:prov:ider(}{it:string}{cmd:)}}provider; default {bf:auto}{p_end}
{synopt:{cmdab:cre:ativity(}{it:real}{cmd:)}}temperature (0-2); omit for model default{p_end}
{synoptline}


{marker description}{...}
{title:Description}

{pstd}
{cmd:catllm classify_social} classifies social media text into predefined
categories using the {bf:catvader} Python package. It supports fetching
posts directly from social media platforms and classifying them.

{pstd}
This command has the same base options as {helpb catllm_classify:catllm classify}
plus social-media-specific options for sourcing data.


{marker examples}{...}
{title:Examples}

{pstd}Classify social media posts already in Stata:{p_end}
{phang2}{cmd:. catllm classify_social post_text, categories("Positive" "Negative" "Neutral") apikey($OPENAI_API_KEY)}{p_end}

{pstd}With platform and handle context:{p_end}
{phang2}{cmd:. catllm classify_social tweet, categories("Political" "Entertainment" "News") apikey($OPENAI_API_KEY) platform("twitter") handle("@example")}{p_end}
