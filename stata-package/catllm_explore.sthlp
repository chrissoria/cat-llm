{smcl}
{* *! version 1.0.0}{...}
{viewerjumpto "Syntax" "catllm_explore##syntax"}{...}
{viewerjumpto "Description" "catllm_explore##description"}{...}
{viewerjumpto "Options" "catllm_explore##options"}{...}
{viewerjumpto "Stored results" "catllm_explore##results"}{...}
{viewerjumpto "Examples" "catllm_explore##examples"}{...}

{title:Title}

{phang}
{bf:catllm explore} {hline 2} Raw category extraction for saturation analysis


{marker syntax}{...}
{title:Syntax}

{p 8 16 2}
{cmd:catllm explore} {varname} [{cmd:if}] [{cmd:in}]{cmd:,}
{cmdab:api:key(}{it:string}{cmd:)}
[{it:options}]

{synoptset 28 tabbed}{...}
{synopthdr}
{synoptline}
{syntab:Required}
{synopt:{cmdab:api:key(}{it:string}{cmd:)}}API key for the LLM provider{p_end}

{syntab:Model}
{synopt:{cmdab:mod:el(}{it:string}{cmd:)}}model name; default {bf:gpt-4o}{p_end}
{synopt:{cmdab:prov:ider(}{it:string}{cmd:)}}provider; default {bf:auto}{p_end}
{synopt:{cmdab:cre:ativity(}{it:real}{cmd:)}}temperature; omit for model default{p_end}

{syntab:Extraction}
{synopt:{cmdab:maxc:ategories(}{it:integer}{cmd:)}}max categories per merge; default {bf:12}{p_end}
{synopt:{cmdab:catsp:erchunk(}{it:integer}{cmd:)}}categories per chunk; default {bf:10}{p_end}
{synopt:{cmdab:div:isions(}{it:integer}{cmd:)}}number of data chunks; default {bf:12}{p_end}
{synopt:{cmdab:it:erations(}{it:integer}{cmd:)}}number of passes; default {bf:8}{p_end}
{synopt:{cmdab:spec:ificity(}{it:string}{cmd:)}}broad or specific; default {bf:broad}{p_end}

{syntab:Context}
{synopt:{cmdab:desc:ription(}{it:string}{cmd:)}}description of the data{p_end}
{synopt:{cmdab:research:question(}{it:string}{cmd:)}}research question context{p_end}
{synopt:{cmdab:fo:cus(}{it:string}{cmd:)}}focus the extraction{p_end}

{syntab:Output}
{synopt:{cmdab:sav:ing(}{it:filename}{cmd:)}}save raw categories to a .dta file{p_end}
{synopt:{cmdab:random:seed(}{it:integer}{cmd:)}}random seed for reproducibility{p_end}
{synoptline}


{marker description}{...}
{title:Description}

{pstd}
{cmd:catllm explore} performs raw category extraction without deduplication,
preserving every category extracted across all chunks and iterations. This
produces approximately {it:iterations} x {it:divisions} x {it:categories_per_chunk}
raw category mentions, which can be used for saturation analysis -- determining
whether additional iterations yield diminishing returns.

{pstd}
Unlike {helpb catllm_extract:catllm extract}, duplicates are intentionally
retained. Use {opt saving()} to export the raw results to a separate dataset
for further analysis.


{marker results}{...}
{title:Stored results}

{pstd}{cmd:catllm explore} stores the following in {cmd:r()}:

{synoptset 20 tabbed}{...}
{p2col 5 20 24 2: Scalars}{p_end}
{synopt:{cmd:r(n_raw)}}total raw categories extracted{p_end}
{synopt:{cmd:r(n_unique)}}number of unique categories{p_end}

{p2col 5 20 24 2: Macros}{p_end}
{synopt:{cmd:r(top_categories)}}top 20 most frequent categories{p_end}
{synopt:{cmd:r(cat1)}, {cmd:r(cat2)}, ...}unique categories (up to 100){p_end}


{marker examples}{...}
{title:Examples}

{pstd}Explore with raw output:{p_end}
{phang2}{cmd:. catllm explore response, apikey($OPENAI_API_KEY)}{p_end}

{pstd}Save raw categories for saturation analysis:{p_end}
{phang2}{cmd:. catllm explore response, apikey($OPENAI_API_KEY) iterations(12) saving(raw_cats)}{p_end}
{phang2}{cmd:. use raw_cats, clear}{p_end}
{phang2}{cmd:. tab raw_category}{p_end}
