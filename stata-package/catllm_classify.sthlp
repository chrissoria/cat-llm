{smcl}
{* *! version 1.1.0}{...}
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
{synopt:{cmdab:gen:erate(}{it:prefix}{cmd:)}}prefix for one byte indicator variable per category; default {bf:cat}{p_end}
{synopt:{cmd:replace}}overwrite existing {it:prefix}{cmd:_*} variables if they exist{p_end}

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
{synopt:{cmdab:twostep:classify(}{it:string}{cmd:)}}force two-step (text-then-format) classification: {it:true}|{it:false}; default auto-on for Ollama{p_end}

{syntab:Prompt tuning (automatic prompt optimization)}
{synopt:{cmdab:prompt:tune(}{it:integer}{cmd:)}}sample size for APO (0 = off, default; >0 enables prompt_tune){p_end}
{synopt:{cmdab:tunei:terations(}{it:integer}{cmd:)}}max instruction attempts per category; default {bf:3}{p_end}
{synopt:{cmdab:tuneui(}{it:string}{cmd:)}}review interface: {it:browser} (default) or {it:terminal}{p_end}
{synopt:{cmdab:tuneo:ptimize(}{it:string}{cmd:)}}metric to maximize: {it:balanced} (default), {it:precision}, {it:sensitivity}{p_end}

{syntab:Backend selection}
{synopt:{cmdab:dom:ain(}{it:string}{cmd:)}}use a domain backend: pol, vader, ademic, survey, cog, web{p_end}
{synopt:{cmdab:pyo:ptions(}{it:string}{cmd:)}}passthrough kwargs: {cmd:"key=val, key=val"}{p_end}
{synoptline}


{marker description}{...}
{title:Description}

{pstd}
{cmd:catllm classify} classifies each text observation in {varname} against
the specified categories using a large language model. For each category,
one byte indicator variable is created in the dataset: {it:prefix}{cmd:_}{it:short_label},
holding 0/1 per row. {opt generate()} sets the prefix; the suffix is
derived from each category's short label (the part before any colon).

{pstd}
Example: {cmd:categories("Positive: ..." "Negative: ..." "Neutral: ...")
generate(sent)} creates {cmd:sent_Positive}, {cmd:sent_Negative},
{cmd:sent_Neutral} as 0/1 byte variables. This matches the wide-DataFrame
shape Python and R users see from {cmd:cat-stack}'s {cmd:classify()}.

{pstd}
In single-model mode, one LLM classifies all observations. In ensemble mode
(when {opt models()} is specified), multiple LLMs classify independently
and a consensus vote determines the final 0/1 per category.


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
Options:

{phang2}{bf:majority} (default) — STRICT majority. More than half of the
models must vote positive. Ties (50/50 splits on even-model ensembles like
2-2 of 4) resolve to {bf:0}. Matches sklearn's VotingClassifier default and
standard ensemble literature. For 2-model ensembles, {bf:majority}
effectively requires both models to agree on positive (there's no "more
than half" of 2 without being all); use 3+ models for a non-degenerate
majority vote, or pass a numeric threshold via {opt pyoptions()}
({cmd:pyoptions("consensus_threshold=0.5")}) to keep the older
"tie favors positive" semantics.{p_end}

{phang2}{bf:two-thirds} — ~67% agreement, uses {bf:>=} comparison.{p_end}

{phang2}{bf:unanimous} — 100% agreement.{p_end}

{phang2}numeric between 0 and 1 — evaluated with {bf:>=} semantics
(the user picked a number; the literal interpretation applies). Pass via
{opt pyoptions()}.{p_end}

{phang}
For even-model ensembles with {bf:majority}, pair with
{cmd:pyoptions("embedding_tiebreaker=True")} to resolve true 50/50 ties
via embedding-centroid similarity instead of the default tie -> 0. That
adds a {cmd:category_N_resolved_by} audit column to the returned data
(values {bf:vote} or {bf:centroid}). Requires {cmd:cat-stack[embeddings]}.

{dlgtab:Prompt tuning}

{phang}
{opt prompttune(integer)} enables Automatic Prompt Optimization (APO). The
value is the sample size used for tuning (typical: 5-20). With {bf:0} (the
default), APO is disabled and classification runs normally. With {bf:N>0},
{cmd:catllm} classifies a random sample of {bf:N} rows, opens a review
interface for you to correct mistakes, then iteratively rewrites the
system prompt to fix the errors you flagged, re-classifies the sample to
verify improvement, and finally applies the optimized prompt to the full
dataset. Categories themselves are never modified — only the prompt.

{phang}
{opt tuneiterations(integer)} sets the maximum instruction-rewrite attempts
per category. Default {bf:3}.

{phang}
{opt tuneui(string)} chooses the corrections review interface:
{bf:browser} (default) opens a local web page with checkboxes;
{bf:terminal} uses text-based input. Browser is recommended unless you
are running over SSH without port forwarding.

{phang}
{opt tuneoptimize(string)} selects which metric APO maximizes across
iterations: {bf:balanced} (default; average of accuracy, sensitivity,
precision), {bf:precision} (minimize false positives), or {bf:sensitivity}
(minimize false negatives).

{phang}
{opt twostepclassify(string)} forces the two-step (text-first, then
JSON-format) classification path. Accepts {bf:true} or {bf:false}. Useful
for lower-tier API models (gpt-4o-mini, claude-haiku, gemini-flash) that
struggle to emit strict per-category JSON in one shot. With the default
(unset), two-step is auto-enabled for Ollama only.

{dlgtab:Backend selection}

{phang}
{opt domain(string)} routes the call through a domain-specific Python
sub-package with prompts tuned for that text type. Valid values:
{bf:pol} (cat-pol), {bf:vader} (cat-vader), {bf:ademic} (cat-ademic),
{bf:survey} (cat-survey), {bf:cog} (cat-cog), {bf:web} (cat-web). With no
{opt domain()}, the base {bf:cat-stack} engine is used. Install a domain
package with {cmd:catllm setup, domain({it:name})}.

{phang}
{opt pyoptions(string)} forwards arbitrary keyword arguments to the
underlying Python function. Format: comma-separated {cmd:key=value} pairs.
Values are parsed as Python literals (numbers, booleans, strings, lists).
Use this to access any cat-stack parameter not otherwise wrapped as a Stata
option.

{pmore}
Common v1.6.0 features reachable via {opt pyoptions()}:

{phang2}{cmd:pyoptions("embedding_tiebreaker=True, min_centroid_size=3")}
— use embedding centroids to break true 50/50 ties on even-model
ensembles. Companion to {cmd:consensus("majority")}. Adds a
{cmd:category_N_resolved_by} audit column to the returned data. Requires
{cmd:cat-stack[embeddings]}.{p_end}

{phang2}{cmd:pyoptions("json_formatter=True")} — enable the local
JSON-repair model that fixes malformed LLM output before marking rows as
failed. Auto-downloads ~1.5 GB of dependencies on first use. Pass
{bf:False} to disable entirely; omit (or pass nothing) for the new
auto-consent behavior that prompts on the first malformed row.{p_end}

{phang2}{cmd:pyoptions("batch_mode=True, batch_timeout=86400")} — use
async batch APIs (~50% cheaper, slower turnaround). Supported providers:
OpenAI, Anthropic, Google, Mistral, xAI. HuggingFace / Perplexity / Ollama
fall back to synchronous calls. Incompatible with PDF / image input and
with {cmd:embedding_tiebreaker}.{p_end}

{phang2}{cmd:pyoptions("max_retries=3, row_delay=0.2")} — classic
rate-limit / retry tuning.{p_end}


{marker results}{...}
{title:Stored results}

{pstd}{cmd:catllm classify} stores the following in {cmd:r()}:

{synoptset 20 tabbed}{...}
{p2col 5 20 24 2: Scalars}{p_end}
{synopt:{cmd:r(N)}}number of observations processed{p_end}
{synopt:{cmd:r(N_classified)}}number successfully classified{p_end}

{p2col 5 20 24 2: Macros}{p_end}
{synopt:{cmd:r(prefix)}}prefix passed to {cmd:generate()}{p_end}
{synopt:{cmd:r(variables)}}space-separated list of indicator variables created{p_end}
{synopt:{cmd:r(model)}}model used{p_end}
{synopt:{cmd:r(provider)}}provider used{p_end}


{marker examples}{...}
{title:Examples}

{pstd}Basic classification (creates cat_Positive cat_Negative cat_Neutral):{p_end}
{phang2}{cmd:. catllm classify open_ended, categories("Positive" "Negative" "Neutral") apikey($OPENAI_API_KEY)}{p_end}

{pstd}With custom prefix and chain-of-thought (creates topic_Health, topic_Education, ...):{p_end}
{phang2}{cmd:. catllm classify response, categories("Health" "Education" "Economy" "Other") apikey($OPENAI_API_KEY) generate(topic) chainofthought}{p_end}

{pstd}Inspect the result:{p_end}
{phang2}{cmd:. list response topic_*, separator(0)}{p_end}
{phang2}{cmd:. tab1 topic_*}{p_end}

{pstd}Using Anthropic Claude:{p_end}
{phang2}{cmd:. catllm classify feedback, categories("Bug" "Feature" "Question") apikey($ANTHROPIC_API_KEY) model("claude-sonnet-4-20250514") provider("anthropic")}{p_end}

{pstd}Ensemble with majority vote:{p_end}
{phang2}{cmd:. catllm classify response, categories("Agree" "Disagree") apikey($OPENAI_API_KEY) models("gpt-4o openai $OPENAI_API_KEY; claude-sonnet-4-20250514 anthropic $ANTHROPIC_API_KEY") consensus("majority")}{p_end}

{pstd}Classify only certain observations:{p_end}
{phang2}{cmd:. catllm classify response if !missing(response), categories("Yes" "No") apikey($OPENAI_API_KEY)}{p_end}

{pstd}Use the survey-domain backend:{p_end}
{phang2}{cmd:. catllm classify response, categories("Positive" "Negative" "Neutral") apikey($OPENAI_API_KEY) domain(survey)}{p_end}

{pstd}Forward an arbitrary kwarg to the Python call:{p_end}
{phang2}{cmd:. catllm classify response, categories("A" "B") apikey($OPENAI_API_KEY) pyoptions("max_retries=3, row_delay=0.2")}{p_end}

{pstd}Automatic prompt optimization with a 5-row sample:{p_end}
{phang2}{cmd:. catllm classify response, categories("A" "B" "C") apikey($OPENAI_API_KEY) prompttune(5) tuneui(browser)}{p_end}

{pstd}Force two-step path for a weaker API model:{p_end}
{phang2}{cmd:. catllm classify response, categories("A" "B") apikey($OPENAI_API_KEY) model(gpt-4o-mini) twostepclassify(true)}{p_end}
