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
{cmd:catllm} provides a Stata interface to the {bf:cat-stack} ecosystem of
Python packages for automating the categorization and analysis of text data
using large language models (LLMs). It supports text classification, category
discovery, saturation analysis, and text summarization across general and
domain-specific contexts (survey, social media, academic, cognitive).

{pstd}
Supported LLM providers: OpenAI, Anthropic, Google, HuggingFace, xAI, Mistral,
Perplexity, and Ollama (local).

{pstd}
Requires Stata 16+ with Python integration and the {bf:cat-stack} Python package.
Run {cmd:catllm setup} to install the Python backend.


{marker subcommands}{...}
{title:Subcommands}

{synoptset 30}{...}
{syntab:Base commands (cat_stack)}
{synopt:{helpb catllm_classify:classify}}classify text into predefined categories{p_end}
{synopt:{helpb catllm_extract:extract}}discover categories from unstructured text{p_end}
{synopt:{helpb catllm_explore:explore}}raw category extraction for saturation analysis{p_end}
{synopt:{helpb catllm_summarize:summarize}}summarize text using LLMs{p_end}
{synopt:{helpb catllm_setup:setup}}install or check the Python backend{p_end}

{syntab:Survey domain (cat_survey)}
{synopt:{helpb catllm_classify_survey:classify_survey}}classify survey responses{p_end}
{synopt:{helpb catllm_extract_survey:extract_survey}}discover categories from survey text{p_end}
{synopt:{helpb catllm_explore_survey:explore_survey}}raw extraction from survey text{p_end}

{syntab:Social media domain (catvader)}
{synopt:{helpb catllm_classify_social:classify_social}}classify social media posts{p_end}
{synopt:{helpb catllm_extract_social:extract_social}}discover categories from social media{p_end}
{synopt:{helpb catllm_explore_social:explore_social}}raw extraction from social media{p_end}

{syntab:Academic domain (catademic)}
{synopt:{helpb catllm_classify_academic:classify_academic}}classify academic text{p_end}
{synopt:{helpb catllm_extract_academic:extract_academic}}discover categories from academic text{p_end}
{synopt:{helpb catllm_explore_academic:explore_academic}}raw extraction from academic text{p_end}
{synopt:{helpb catllm_summarize_academic:summarize_academic}}summarize academic text{p_end}

{syntab:Cognitive domain (cat_cog)}
{synopt:{helpb catllm_cerad:cerad}}score CERAD drawn shapes from images{p_end}


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
