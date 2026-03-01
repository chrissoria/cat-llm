{smcl}
{* *! version 1.0.0}{...}
{viewerjumpto "Syntax" "catllm##syntax"}{...}
{viewerjumpto "Description" "catllm##description"}{...}
{viewerjumpto "Subcommands" "catllm##subcommands"}{...}
{viewerjumpto "Setup" "catllm##setup"}{...}
{viewerjumpto "Examples" "catllm##examples"}{...}
{viewerjumpto "Author" "catllm##author"}{...}

{title:Title}

{phang}
{bf:catllm} {hline 2} LLM-powered classification and extraction for survey data


{marker syntax}{...}
{title:Syntax}

{p 8 16 2}
{cmd:catllm} {it:subcommand} [{it:varname}] [{cmd:if}] [{cmd:in}]{cmd:,} {it:options}


{marker description}{...}
{title:Description}

{pstd}
{cmd:catllm} provides a Stata interface to the {bf:catllm} Python package for
automating the categorization and analysis of open-ended survey responses using
large language models (LLMs). It supports text classification, category
discovery, saturation analysis, and text summarization.

{pstd}
Supported LLM providers: OpenAI, Anthropic, Google, HuggingFace, xAI, Mistral,
Perplexity, and Ollama (local).

{pstd}
Requires Stata 16+ with Python integration and the {bf:catllm} Python package.
Run {cmd:catllm setup} to install the Python backend.


{marker subcommands}{...}
{title:Subcommands}

{synoptset 20}{...}
{synopt:{helpb catllm_classify:classify}}classify text into predefined categories{p_end}
{synopt:{helpb catllm_extract:extract}}discover categories from unstructured text{p_end}
{synopt:{helpb catllm_explore:explore}}raw category extraction for saturation analysis{p_end}
{synopt:{helpb catllm_summarize:summarize}}summarize text using LLMs{p_end}
{synopt:{helpb catllm_setup:setup}}install or check the Python backend{p_end}


{marker setup}{...}
{title:Setup}

{pstd}
Before using {cmd:catllm}, you must install the Python backend:

{phang2}{cmd:. catllm setup}{p_end}

{pstd}
For PDF support:

{phang2}{cmd:. catllm setup, pdf}{p_end}

{pstd}
To check your installation:

{phang2}{cmd:. catllm setup, check}{p_end}

{pstd}
You also need an API key from a supported provider. Store it in a global macro:

{phang2}{cmd:. global OPENAI_API_KEY "sk-..."}{p_end}


{marker examples}{...}
{title:Examples}

{pstd}Classify open-ended survey responses:{p_end}
{phang2}{cmd:. catllm classify response, categories("Positive" "Negative" "Neutral") apikey($OPENAI_API_KEY)}{p_end}

{pstd}Discover categories automatically:{p_end}
{phang2}{cmd:. catllm extract response, apikey($OPENAI_API_KEY) maxcategories(10)}{p_end}

{pstd}Summarize text responses:{p_end}
{phang2}{cmd:. catllm summarize response, apikey($OPENAI_API_KEY) generate(summary)}{p_end}

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
