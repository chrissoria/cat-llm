{smcl}
{* *! version 1.0.0}{...}
{viewerjumpto "Syntax" "catllm_classify_academic##syntax"}{...}
{viewerjumpto "Description" "catllm_classify_academic##description"}{...}
{viewerjumpto "Options" "catllm_classify_academic##options"}{...}
{viewerjumpto "Examples" "catllm_classify_academic##examples"}{...}

{title:Title}

{phang}
{bf:catllm classify_academic} {hline 2} Classify academic text into categories using catademic


{marker syntax}{...}
{title:Syntax}

{p 8 16 2}
{cmd:catllm classify_academic} {varname} [{cmd:if}] [{cmd:in}]{cmd:,}
{cmdab:cat:egories(}{it:string}{cmd:)} {cmdab:api:key(}{it:string}{cmd:)}
[{it:options}]

{synoptset 28 tabbed}{...}
{synopthdr}
{synoptline}
{syntab:Required}
{synopt:{cmdab:cat:egories(}{it:string}{cmd:)}}category names in double quotes{p_end}
{synopt:{cmdab:api:key(}{it:string}{cmd:)}}API key for the LLM provider{p_end}

{syntab:Academic source}
{synopt:{cmdab:journal:name(}{it:string}{cmd:)}}journal name for paper retrieval{p_end}
{synopt:{cmdab:journal:issn(}{it:string}{cmd:)}}journal ISSN{p_end}
{synopt:{cmdab:journal:field(}{it:string}{cmd:)}}journal field/discipline{p_end}
{synopt:{cmdab:topic:name(}{it:string}{cmd:)}}topic for paper search{p_end}
{synopt:{cmdab:paper:limit(}{it:integer}{cmd:)}}max papers to fetch; default {bf:50}{p_end}
{synopt:{cmdab:polite:email(}{it:string}{cmd:)}}email for polite API access (Crossref){p_end}
{synopt:{cmdab:date:from(}{it:string}{cmd:)}}start date filter (YYYY-MM-DD){p_end}
{synopt:{cmdab:date:to(}{it:string}{cmd:)}}end date filter (YYYY-MM-DD){p_end}

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
{cmd:catllm classify_academic} classifies academic text (e.g., paper abstracts)
into predefined categories using the {bf:catademic} Python package. It can also
fetch papers from journals or topics via Crossref and OpenAlex.

{pstd}
This command has the same base options as {helpb catllm_classify:catllm classify}
plus academic-specific options for data sourcing.


{marker examples}{...}
{title:Examples}

{pstd}Classify abstracts already in Stata:{p_end}
{phang2}{cmd:. catllm classify_academic abstract, categories("Empirical" "Theoretical" "Review") apikey($OPENAI_API_KEY)}{p_end}

{pstd}With journal context:{p_end}
{phang2}{cmd:. catllm classify_academic abstract, categories("Quantitative" "Qualitative" "Mixed") apikey($OPENAI_API_KEY) journalname("American Sociological Review") politeemail("user@university.edu")}{p_end}
