{smcl}
{* *! version 1.0.0}{...}
{viewerjumpto "Syntax" "catllm_extract##syntax"}{...}
{viewerjumpto "Description" "catllm_extract##description"}{...}
{viewerjumpto "Options" "catllm_extract##options"}{...}
{viewerjumpto "Stored results" "catllm_extract##results"}{...}
{viewerjumpto "Examples" "catllm_extract##examples"}{...}

{title:Title}

{phang}
{bf:catllm extract} {hline 2} Discover categories from unstructured text


{marker syntax}{...}
{title:Syntax}

{p 8 16 2}
{cmd:catllm extract} {varname} [{cmd:if}] [{cmd:in}]{cmd:,}
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
{synopt:{cmdab:maxc:ategories(}{it:integer}{cmd:)}}max categories to return; default {bf:12}{p_end}
{synopt:{cmdab:catsp:erchunk(}{it:integer}{cmd:)}}categories per chunk; default {bf:10}{p_end}
{synopt:{cmdab:div:isions(}{it:integer}{cmd:)}}number of data chunks; default {bf:12}{p_end}
{synopt:{cmdab:it:erations(}{it:integer}{cmd:)}}number of extraction passes; default {bf:8}{p_end}
{synopt:{cmdab:spec:ificity(}{it:string}{cmd:)}}broad or specific; default {bf:broad}{p_end}

{syntab:Context}
{synopt:{cmdab:desc:ription(}{it:string}{cmd:)}}description of the data{p_end}
{synopt:{cmdab:research:question(}{it:string}{cmd:)}}research question context{p_end}
{synopt:{cmdab:fo:cus(}{it:string}{cmd:)}}focus the extraction (e.g., "reasons for moving"){p_end}

{syntab:Reproducibility}
{synopt:{cmdab:random:seed(}{it:integer}{cmd:)}}random seed for chunk sampling{p_end}
{synoptline}


{marker description}{...}
{title:Description}

{pstd}
{cmd:catllm extract} discovers categories from unstructured text data by
repeatedly sampling chunks of responses, extracting candidate categories via
an LLM, and merging them into a final deduplicated list. This is useful when
you do not have predefined categories and want the LLM to identify themes.

{pstd}
The algorithm splits the data into {opt divisions()} chunks, runs
{opt iterations()} passes with reshuffling, and merges the results down
to {opt maxcategories()}.


{marker options}{...}
{title:Options}

{phang}
{opt specificity(string)} controls category granularity. {bf:broad} (default)
produces general themes; {bf:specific} produces finer-grained categories.

{phang}
{opt focus(string)} narrows extraction to a specific aspect.
Example: {cmd:focus("reasons for leaving")}.


{marker results}{...}
{title:Stored results}

{pstd}{cmd:catllm extract} stores the following in {cmd:r()}:

{synoptset 20 tabbed}{...}
{p2col 5 20 24 2: Scalars}{p_end}
{synopt:{cmd:r(n_categories)}}number of categories discovered{p_end}

{p2col 5 20 24 2: Macros}{p_end}
{synopt:{cmd:r(categories)}}all categories as a quoted list{p_end}
{synopt:{cmd:r(cat1)}, {cmd:r(cat2)}, ...}individual category names{p_end}

{p2col 5 20 24 2: Matrices}{p_end}
{synopt:{cmd:r(counts)}}category frequency matrix{p_end}


{marker examples}{...}
{title:Examples}

{pstd}Discover categories from survey responses:{p_end}
{phang2}{cmd:. catllm extract open_ended, apikey($OPENAI_API_KEY)}{p_end}
{phang2}{cmd:. return list}{p_end}

{pstd}With research context:{p_end}
{phang2}{cmd:. catllm extract why_moved, apikey($OPENAI_API_KEY) description("Reasons people gave for relocating") maxcategories(8) specificity("specific")}{p_end}

{pstd}Use discovered categories for classification:{p_end}
{phang2}{cmd:. catllm extract response, apikey($OPENAI_API_KEY)}{p_end}
{phang2}{cmd:. local cats = r(categories)}{p_end}
{phang2}{cmd:. catllm classify response, categories(`cats') apikey($OPENAI_API_KEY)}{p_end}
