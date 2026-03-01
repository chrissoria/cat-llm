{smcl}
{* *! version 1.0.0}{...}
{viewerjumpto "Syntax" "catllm_summarize##syntax"}{...}
{viewerjumpto "Description" "catllm_summarize##description"}{...}
{viewerjumpto "Options" "catllm_summarize##options"}{...}
{viewerjumpto "Stored results" "catllm_summarize##results"}{...}
{viewerjumpto "Examples" "catllm_summarize##examples"}{...}

{title:Title}

{phang}
{bf:catllm summarize} {hline 2} Summarize text using LLMs


{marker syntax}{...}
{title:Syntax}

{p 8 16 2}
{cmd:catllm summarize} {varname} [{cmd:if}] [{cmd:in}]{cmd:,}
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

{syntab:Model}
{synopt:{cmdab:mod:el(}{it:string}{cmd:)}}model name; default {bf:gpt-4o}{p_end}
{synopt:{cmdab:prov:ider(}{it:string}{cmd:)}}provider; default {bf:auto}{p_end}
{synopt:{cmdab:cre:ativity(}{it:real}{cmd:)}}temperature; omit for model default{p_end}

{syntab:Summarization}
{synopt:{cmdab:desc:ription(}{it:string}{cmd:)}}description of the content{p_end}
{synopt:{cmdab:ins:tructions(}{it:string}{cmd:)}}custom instructions (e.g., "use bullet points"){p_end}
{synopt:{cmdab:maxl:ength(}{it:integer}{cmd:)}}maximum summary length in words{p_end}
{synopt:{cmdab:fo:cus(}{it:string}{cmd:)}}what to emphasize (e.g., "main arguments"){p_end}

{syntab:Prompting strategies}
{synopt:{cmdab:chain:ofthought}}enable chain-of-thought reasoning (default on){p_end}
{synopt:{cmdab:step:back}}enable step-back prompting{p_end}
{synopt:{cmdab:con:text}}add expert context prompt{p_end}

{syntab:Ensemble}
{synopt:{cmdab:models(}{it:string}{cmd:)}}model specs separated by semicolons{p_end}
{synoptline}


{marker description}{...}
{title:Description}

{pstd}
{cmd:catllm summarize} generates a summary for each text observation using
a large language model. Summaries are stored as a new {bf:strL} variable,
allowing for long text output.


{marker results}{...}
{title:Stored results}

{pstd}{cmd:catllm summarize} stores the following in {cmd:r()}:

{synoptset 20 tabbed}{...}
{p2col 5 20 24 2: Scalars}{p_end}
{synopt:{cmd:r(N)}}number of observations processed{p_end}
{synopt:{cmd:r(N_summarized)}}number successfully summarized{p_end}

{p2col 5 20 24 2: Macros}{p_end}
{synopt:{cmd:r(variable)}}name of the generated variable{p_end}
{synopt:{cmd:r(model)}}model used{p_end}


{marker examples}{...}
{title:Examples}

{pstd}Basic summarization:{p_end}
{phang2}{cmd:. catllm summarize long_text, apikey($OPENAI_API_KEY) generate(summary)}{p_end}

{pstd}With instructions and length limit:{p_end}
{phang2}{cmd:. catllm summarize narrative, apikey($OPENAI_API_KEY) generate(brief) instructions("Write in bullet points") maxlength(50)}{p_end}

{pstd}Using Anthropic with focus:{p_end}
{phang2}{cmd:. catllm summarize interview, apikey($ANTHROPIC_API_KEY) model("claude-sonnet-4-20250514") provider("anthropic") generate(key_points) focus("policy recommendations")}{p_end}
