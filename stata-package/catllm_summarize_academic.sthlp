{smcl}
{* *! version 1.0.0}{...}
{viewerjumpto "Syntax" "catllm_summarize_academic##syntax"}{...}
{viewerjumpto "Description" "catllm_summarize_academic##description"}{...}
{viewerjumpto "Examples" "catllm_summarize_academic##examples"}{...}

{title:Title}

{phang}
{bf:catllm summarize_academic} {hline 2} Summarize academic text using catademic


{marker syntax}{...}
{title:Syntax}

{p 8 16 2}
{cmd:catllm summarize_academic} {varname} [{cmd:if}] [{cmd:in}]{cmd:,}
{cmdab:api:key(}{it:string}{cmd:)}
[{it:options}]

{synoptset 28 tabbed}{...}
{synopthdr}
{synoptline}
{syntab:Required}
{synopt:{cmdab:api:key(}{it:string}{cmd:)}}API key for the LLM provider{p_end}

{syntab:Output}
{synopt:{cmdab:gen:erate(}{it:newvar}{cmd:)}}name for the summary variable; default {bf:_catllm_summ}{p_end}
{synopt:{cmd:replace}}overwrite {it:generate} if it already exists{p_end}

{syntab:Academic source}
{synopt:{cmdab:journal:name(}{it:string}{cmd:)}}journal name{p_end}
{synopt:{cmdab:journal:issn(}{it:string}{cmd:)}}journal ISSN{p_end}
{synopt:{cmdab:journal:field(}{it:string}{cmd:)}}journal field/discipline{p_end}
{synopt:{cmdab:topic:name(}{it:string}{cmd:)}}topic for paper search{p_end}
{synopt:{cmdab:paper:limit(}{it:integer}{cmd:)}}max papers; default {bf:50}{p_end}
{synopt:{cmdab:polite:email(}{it:string}{cmd:)}}polite email for API access{p_end}
{synopt:{cmdab:date:from(}{it:string}{cmd:)}}start date (YYYY-MM-DD){p_end}
{synopt:{cmdab:date:to(}{it:string}{cmd:)}}end date (YYYY-MM-DD){p_end}

{syntab:Summarization}
{synopt:{cmdab:ins:tructions(}{it:string}{cmd:)}}custom instructions{p_end}
{synopt:{cmdab:maxl:ength(}{it:integer}{cmd:)}}max summary length in words{p_end}
{synopt:{cmdab:fo:cus(}{it:string}{cmd:)}}what to emphasize{p_end}

{syntab:Model}
{synopt:{cmdab:mod:el(}{it:string}{cmd:)}}model name; default {bf:gpt-4o}{p_end}
{synopt:{cmdab:prov:ider(}{it:string}{cmd:)}}provider; default {bf:auto}{p_end}
{synoptline}


{marker description}{...}
{title:Description}

{pstd}
{cmd:catllm summarize_academic} generates summaries for academic text using
the {bf:catademic} Python package. See {helpb catllm_summarize:catllm summarize}
for full summarization option details.


{marker examples}{...}
{title:Examples}

{pstd}Summarize paper abstracts:{p_end}
{phang2}{cmd:. catllm summarize_academic abstract, apikey($OPENAI_API_KEY) generate(summary) focus("methodology")}{p_end}

{pstd}With journal context:{p_end}
{phang2}{cmd:. catllm summarize_academic full_text, apikey($OPENAI_API_KEY) generate(brief) journalfield("public health") maxlength(100)}{p_end}
