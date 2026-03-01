{smcl}
{* *! version 1.0.0}{...}
{viewerjumpto "Syntax" "catllm_classify##syntax"}{...}
{viewerjumpto "Description" "catllm_classify##description"}{...}
{viewerjumpto "Options" "catllm_classify##options"}{...}
{viewerjumpto "Stored results" "catllm_classify##results"}{...}
{viewerjumpto "Examples" "catllm_classify##examples"}{...}

{title:Title}

{phang}
{bf:catllm classify} {hline 2} Classify text into categories using LLMs


{marker syntax}{...}
{title:Syntax}

{p 8 16 2}
{cmd:catllm classify} {varname} [{cmd:if}] [{cmd:in}]{cmd:,}
{cmdab:cat:egories(}{it:string}{cmd:)} {cmdab:api:key(}{it:string}{cmd:)}
[{it:options}]

{synoptset 28 tabbed}{...}
{synopthdr}
{synoptline}
{syntab:Required}
{synopt:{cmdab:cat:egories(}{it:string}{cmd:)}}category names in double quotes{p_end}
{synopt:{cmdab:api:key(}{it:string}{cmd:)}}API key for the LLM provider{p_end}

{syntab:Output}
{synopt:{cmdab:gen:erate(}{it:newvar}{cmd:)}}name for the classification variable; default {bf:_catllm_class}{p_end}
{synopt:{cmd:replace}}overwrite {it:generate} if it already exists{p_end}

{syntab:Model}
{synopt:{cmdab:mod:el(}{it:string}{cmd:)}}model name; default {bf:gpt-4o}{p_end}
{synopt:{cmdab:prov:ider(}{it:string}{cmd:)}}provider: openai, anthropic, google, etc.; default {bf:auto}{p_end}
{synopt:{cmdab:cre:ativity(}{it:real}{cmd:)}}temperature (0-2); omit for model default{p_end}

{syntab:Prompting strategies}
{synopt:{cmdab:chain:ofthought}}enable chain-of-thought reasoning{p_end}
{synopt:{cmdab:think:ing(}{it:integer}{cmd:)}}extended thinking token budget{p_end}
{synopt:{cmdab:step:back}}enable step-back prompting{p_end}
{synopt:{cmdab:con:text}}add expert context prompt{p_end}
{synopt:{cmdab:desc:ription(}{it:string}{cmd:)}}task description for the LLM{p_end}
{synopt:{cmdab:survey:question(}{it:string}{cmd:)}}survey question context{p_end}

{syntab:Ensemble (multi-model)}
{synopt:{cmdab:models(}{it:string}{cmd:)}}model specs separated by semicolons{p_end}
{synopt:{cmdab:cons:ensus(}{it:string}{cmd:)}}threshold: majority, two-thirds, unanimous, or 0-1{p_end}

{syntab:Advanced}
{synopt:{cmdab:maxw:orkers(}{it:integer}{cmd:)}}parallel workers (0 = auto){p_end}
{synopt:{cmdab:maxr:etries(}{it:integer}{cmd:)}}max API retries; default {bf:5}{p_end}
{synopt:{cmdab:retr:ydelay(}{it:real}{cmd:)}}delay between retries in seconds; default {bf:1.0}{p_end}
{synopt:{cmdab:rowd:elay(}{it:real}{cmd:)}}delay between rows in seconds; default {bf:0.0}{p_end}
{synopt:{cmdab:fail:strategy(}{it:string}{cmd:)}}partial or strict; default {bf:partial}{p_end}
{synopt:{cmd:nojsonschema}}disable structured JSON output{p_end}
{synoptline}


{marker description}{...}
{title:Description}

{pstd}
{cmd:catllm classify} assigns each text observation in {varname} to one of
the specified categories using a large language model. The result is stored
as a new string variable.

{pstd}
In single-model mode, one LLM classifies all observations. In ensemble mode
(when {opt models()} is specified), multiple LLMs classify independently
and a consensus vote determines the final category.


{marker options}{...}
{title:Options}

{dlgtab:Required}

{phang}
{opt categories(string)} specifies the category names. Enclose multi-word
categories in double quotes:
{cmd:categories("Strongly Agree" "Agree" "Disagree" "Strongly Disagree")}.

{phang}
{opt apikey(string)} specifies the API key. Use a global macro:
{cmd:apikey($OPENAI_API_KEY)}.

{dlgtab:Ensemble}

{phang}
{opt models(string)} specifies multiple models for ensemble classification.
Format: {cmd:"model provider key; model2 provider2 key2"}.
Example: {cmd:models("gpt-4o openai $OPENAI_API_KEY; claude-sonnet-4-20250514 anthropic $ANTHROPIC_API_KEY")}.

{phang}
{opt consensus(string)} sets the consensus threshold for ensemble mode.
Options: {bf:majority} (default), {bf:two-thirds}, {bf:unanimous}, or a
number between 0 and 1.


{marker results}{...}
{title:Stored results}

{pstd}{cmd:catllm classify} stores the following in {cmd:r()}:

{synoptset 20 tabbed}{...}
{p2col 5 20 24 2: Scalars}{p_end}
{synopt:{cmd:r(N)}}number of observations processed{p_end}
{synopt:{cmd:r(N_classified)}}number successfully classified{p_end}

{p2col 5 20 24 2: Macros}{p_end}
{synopt:{cmd:r(variable)}}name of the generated variable{p_end}
{synopt:{cmd:r(model)}}model used{p_end}
{synopt:{cmd:r(provider)}}provider used{p_end}


{marker examples}{...}
{title:Examples}

{pstd}Basic classification:{p_end}
{phang2}{cmd:. catllm classify open_ended, categories("Positive" "Negative" "Neutral") apikey($OPENAI_API_KEY)}{p_end}

{pstd}With custom variable name and chain-of-thought:{p_end}
{phang2}{cmd:. catllm classify response, categories("Health" "Education" "Economy" "Other") apikey($OPENAI_API_KEY) generate(topic) chainofthought}{p_end}

{pstd}Using Anthropic Claude:{p_end}
{phang2}{cmd:. catllm classify feedback, categories("Bug" "Feature" "Question") apikey($ANTHROPIC_API_KEY) model("claude-sonnet-4-20250514") provider("anthropic")}{p_end}

{pstd}Ensemble with majority vote:{p_end}
{phang2}{cmd:. catllm classify response, categories("Agree" "Disagree") apikey($OPENAI_API_KEY) models("gpt-4o openai $OPENAI_API_KEY; claude-sonnet-4-20250514 anthropic $ANTHROPIC_API_KEY") consensus("majority")}{p_end}

{pstd}Classify only certain observations:{p_end}
{phang2}{cmd:. catllm classify response if !missing(response), categories("Yes" "No") apikey($OPENAI_API_KEY)}{p_end}
