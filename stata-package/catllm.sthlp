{smcl}
{* *! version 2.0.0}{...}
{viewerjumpto "Syntax" "catllm##syntax"}{...}
{viewerjumpto "Description" "catllm##description"}{...}
{viewerjumpto "Subcommands" "catllm##subcommands"}{...}
{viewerjumpto "Domain option" "catllm##domain"}{...}
{viewerjumpto "Passthrough options" "catllm##pyoptions"}{...}
{viewerjumpto "Setup" "catllm##setup"}{...}
{viewerjumpto "Examples" "catllm##examples"}{...}
{viewerjumpto "Author" "catllm##author"}{...}

{title:Title}

{phang}
{bf:catllm} {hline 2} LLM-powered classification and extraction for text data


{marker syntax}{...}
{title:Syntax}

{p 8 16 2}
{cmd:catllm} {it:subcommand} [{it:varname}] [{cmd:if}] [{cmd:in}]{cmd:,} {it:options}


{marker description}{...}
{title:Description}

{pstd}
{cmd:catllm} provides a Stata interface to the {bf:cat-stack} ecosystem of
Python packages for automating the categorization and analysis of text data
using large language models (LLMs). It supports text classification, category
discovery, saturation analysis, and text summarization.

{pstd}
Domain-specific prompt templates -- for political opinion, sentiment,
academic, survey, cognitive, and web text -- are available via the
{cmd:domain()} option on each verb command.

{pstd}
Supported LLM providers: OpenAI, Anthropic, Google, HuggingFace, xAI, Mistral,
Perplexity, and Ollama (local).

{pstd}
Requires Stata 16+ with Python integration and the {bf:cat-stack} Python package.
Run {cmd:catllm setup} to install the Python backend.


{marker subcommands}{...}
{title:Subcommands}

{synoptset 30}{...}
{synopt:{helpb catllm_classify:classify}}classify text into predefined categories{p_end}
{synopt:{helpb catllm_extract:extract}}discover categories from unstructured text{p_end}
{synopt:{helpb catllm_explore:explore}}raw category extraction for saturation analysis{p_end}
{synopt:{helpb catllm_summarize:summarize}}summarize text using LLMs{p_end}
{synopt:{helpb catllm_cerad:cerad}}score CERAD drawn shapes from images{p_end}
{synopt:{helpb catllm_setup:setup}}install or check the Python backend{p_end}


{marker domain}{...}
{title:Domain option}

{pstd}
The four verb commands ({cmd:classify}, {cmd:extract}, {cmd:explore},
{cmd:summarize}) accept a {cmd:domain()} option that selects a domain-specific
Python backend with prompts tuned for that text type:

{synoptset 20}{...}
{synopt:{cmd:domain(pol)}}political opinion text (uses {bf:cat-pol}){p_end}
{synopt:{cmd:domain(vader)}}sentiment / social media text (uses {bf:cat-vader}){p_end}
{synopt:{cmd:domain(ademic)}}academic text (uses {bf:cat-ademic}){p_end}
{synopt:{cmd:domain(survey)}}open-ended survey responses (uses {bf:cat-survey}){p_end}
{synopt:{cmd:domain(cog)}}cognitive text (uses {bf:cat-cog}){p_end}
{synopt:{cmd:domain(web)}}web-scraped text (uses {bf:cat-web}){p_end}

{pstd}
With no {cmd:domain()} option, the base {bf:cat-stack} engine is used. Install
domain packages with {cmd:catllm setup, domain({it:name})} or
{cmd:catllm setup, domain(all)}.


{marker pyoptions}{...}
{title:Passthrough options}

{pstd}
The four verb commands also accept a {cmd:pyoptions()} escape hatch that
forwards arbitrary keyword arguments to the underlying Python function.
Format: comma-separated {cmd:key=value} pairs. Values are parsed as Python
literals (numbers, booleans, strings, lists).

{pstd}This lets you use any cat-stack parameter that isn't first-classed as a
Stata option, without waiting for a Stata release.

{phang2}{cmd:. catllm classify response, apikey($OPENAI_API_KEY) categories("A" "B") pyoptions("max_retries=3, retry_delay=0.5, verbose=True")}{p_end}


{marker setup}{...}
{title:Setup}

{pstd}
Install the base Python backend:

{phang2}{cmd:. catllm setup}{p_end}

{pstd}
With PDF support:

{phang2}{cmd:. catllm setup, pdf}{p_end}

{pstd}
Install a single domain package (or {cmd:domain(all)} for all six):

{phang2}{cmd:. catllm setup, domain(pol)}{p_end}
{phang2}{cmd:. catllm setup, domain(all)}{p_end}

{pstd}
Check the installation and probe each domain package:

{phang2}{cmd:. catllm setup, check}{p_end}

{pstd}
You also need an API key from a supported provider:

{phang2}{cmd:. global OPENAI_API_KEY "sk-..."}{p_end}


{marker examples}{...}
{title:Examples}

{pstd}Classify open-ended responses with the base engine:{p_end}
{phang2}{cmd:. catllm classify response, categories("Positive" "Negative" "Neutral") apikey($OPENAI_API_KEY)}{p_end}

{pstd}Same command, with the survey-domain prompts:{p_end}
{phang2}{cmd:. catllm classify response, categories("Positive" "Negative" "Neutral") apikey($OPENAI_API_KEY) domain(survey)}{p_end}

{pstd}Discover categories automatically:{p_end}
{phang2}{cmd:. catllm extract response, apikey($OPENAI_API_KEY) maxcategories(10)}{p_end}

{pstd}Summarize text responses:{p_end}
{phang2}{cmd:. catllm summarize response, apikey($OPENAI_API_KEY) generate(summary)}{p_end}

{pstd}Pass an arbitrary Python kwarg through:{p_end}
{phang2}{cmd:. catllm classify response, categories("A" "B") apikey($OPENAI_API_KEY) pyoptions("max_retries=3, row_delay=0.2")}{p_end}

{pstd}Multi-model ensemble classification:{p_end}
{phang2}{cmd:. catllm classify response, categories("Agree" "Disagree" "Unsure") apikey($OPENAI_API_KEY) models("gpt-4o openai $OPENAI_API_KEY; claude-sonnet-4-20250514 anthropic $ANTHROPIC_API_KEY")}{p_end}


{marker author}{...}
{title:Author}

{pstd}
Christopher Soria{break}
University of California, Berkeley{break}
chrissoria@berkeley.edu{break}
{browse "https://github.com/chrissoria/cat-llm"}
{p_end}
