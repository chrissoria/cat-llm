{smcl}
{* *! version 1.0.0}{...}
{viewerjumpto "Syntax" "catllm_cerad##syntax"}{...}
{viewerjumpto "Description" "catllm_cerad##description"}{...}
{viewerjumpto "Options" "catllm_cerad##options"}{...}
{viewerjumpto "Stored results" "catllm_cerad##results"}{...}
{viewerjumpto "Examples" "catllm_cerad##examples"}{...}

{title:Title}

{phang}
{bf:catllm cerad} {hline 2} Score CERAD drawn shapes using cat_cog


{marker syntax}{...}
{title:Syntax}

{p 8 16 2}
{cmd:catllm cerad} {varname} [{cmd:if}] [{cmd:in}]{cmd:,}
{cmd:shape(}{it:string}{cmd:)} {cmdab:api:key(}{it:string}{cmd:)}
[{it:options}]

{synoptset 28 tabbed}{...}
{synopthdr}
{synoptline}
{syntab:Required}
{synopt:{cmd:shape(}{it:string}{cmd:)}}shape to score: {bf:circle}, {bf:diamond}, {bf:rectangles}, or {bf:cube}{p_end}
{synopt:{cmdab:api:key(}{it:string}{cmd:)}}API key for the LLM provider{p_end}

{syntab:Output}
{synopt:{cmdab:gen:erate(}{it:newvar}{cmd:)}}name for the score variable; default {bf:_catllm_cerad}{p_end}
{synopt:{cmd:replace}}overwrite {it:generate} if it already exists{p_end}

{syntab:Model}
{synopt:{cmdab:mod:el(}{it:string}{cmd:)}}model name; default {bf:gpt-4o}{p_end}
{synopt:{cmdab:prov:ider(}{it:string}{cmd:)}}provider; default {bf:auto}{p_end}
{synopt:{cmdab:cre:ativity(}{it:real}{cmd:)}}temperature; omit for model default{p_end}
{synoptline}


{marker description}{...}
{title:Description}

{pstd}
{cmd:catllm cerad} scores CERAD (Consortium to Establish a Registry for
Alzheimer's Disease) constructional praxis drawings using the {bf:cat_cog}
Python package. The variable should contain file paths to images of drawn
shapes. Each image is scored by a vision-capable LLM.

{pstd}
The {opt shape()} option specifies which shape the images depict. Valid
values are: {bf:circle}, {bf:diamond}, {bf:rectangles}, and {bf:cube}.


{marker results}{...}
{title:Stored results}

{pstd}{cmd:catllm cerad} stores the following in {cmd:r()}:

{synoptset 20 tabbed}{...}
{p2col 5 20 24 2: Scalars}{p_end}
{synopt:{cmd:r(N)}}number of observations processed{p_end}
{synopt:{cmd:r(N_scored)}}number successfully scored{p_end}

{p2col 5 20 24 2: Macros}{p_end}
{synopt:{cmd:r(variable)}}name of the generated variable{p_end}
{synopt:{cmd:r(model)}}model used{p_end}
{synopt:{cmd:r(shape)}}shape scored{p_end}


{marker examples}{...}
{title:Examples}

{pstd}Score circle drawings:{p_end}
{phang2}{cmd:. catllm cerad image_path, shape("circle") apikey($OPENAI_API_KEY)}{p_end}

{pstd}Score cube drawings with custom variable:{p_end}
{phang2}{cmd:. catllm cerad drawing_file, shape("cube") apikey($OPENAI_API_KEY) generate(cube_score) model("gpt-4o")}{p_end}

{pstd}Score diamond drawings using Anthropic:{p_end}
{phang2}{cmd:. catllm cerad img_path, shape("diamond") apikey($ANTHROPIC_API_KEY) model("claude-sonnet-4-20250514") provider("anthropic")}{p_end}
