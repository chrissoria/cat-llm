{smcl}
{* *! version 1.0.0}{...}
{viewerjumpto "Syntax" "catllm_classify_survey##syntax"}{...}
{viewerjumpto "Description" "catllm_classify_survey##description"}{...}
{viewerjumpto "Options" "catllm_classify_survey##options"}{...}
{viewerjumpto "Examples" "catllm_classify_survey##examples"}{...}

{title:Title}

{phang}
{bf:catllm classify_survey} {hline 2} Classify survey text into categories using cat_survey


{marker syntax}{...}
{title:Syntax}

{p 8 16 2}
{cmd:catllm classify_survey} {varname} [{cmd:if}] [{cmd:in}]{cmd:,}
{cmdab:cat:egories(}{it:string}{cmd:)} {cmdab:api:key(}{it:string}{cmd:)}
[{it:options}]

{synoptset 28 tabbed}{...}
{synopthdr}
{synoptline}
{syntab:Required}
{synopt:{cmdab:cat:egories(}{it:string}{cmd:)}}category names in double quotes{p_end}
{synopt:{cmdab:api:key(}{it:string}{cmd:)}}API key for the LLM provider{p_end}

{syntab:Survey-specific}
{synopt:{cmdab:survey:question(}{it:string}{cmd:)}}the survey question that produced the responses{p_end}

{syntab:Output}
{synopt:{cmdab:gen:erate(}{it:newvar}{cmd:)}}name for the classification variable; default {bf:_catllm_class}{p_end}
{synopt:{cmd:replace}}overwrite {it:generate} if it already exists{p_end}

{syntab:Model}
{synopt:{cmdab:mod:el(}{it:string}{cmd:)}}model name; default {bf:gpt-4o}{p_end}
{synopt:{cmdab:prov:ider(}{it:string}{cmd:)}}provider; default {bf:auto}{p_end}
{synopt:{cmdab:cre:ativity(}{it:real}{cmd:)}}temperature (0-2); omit for model default{p_end}

{syntab:Prompting strategies}
{synopt:{cmdab:chain:ofthought}}enable chain-of-thought reasoning{p_end}
{synopt:{cmdab:think:ing(}{it:integer}{cmd:)}}extended thinking token budget{p_end}
{synopt:{cmdab:step:back}}enable step-back prompting{p_end}
{synopt:{cmdab:con:text}}add expert context prompt{p_end}
{synopt:{cmdab:desc:ription(}{it:string}{cmd:)}}task description for the LLM{p_end}
{synoptline}


{marker description}{...}
{title:Description}

{pstd}
{cmd:catllm classify_survey} classifies survey response text into predefined
categories using the {bf:cat_survey} Python package. It is optimized for
survey data and accepts a {opt surveyquestion()} option to provide the
original question as context to the LLM.

{pstd}
This command has the same options as {helpb catllm_classify:catllm classify}
plus the survey-specific {opt surveyquestion()} option.


{marker examples}{...}
{title:Examples}

{pstd}Classify survey responses with question context:{p_end}
{phang2}{cmd:. catllm classify_survey response, categories("Positive" "Negative" "Neutral") apikey($OPENAI_API_KEY) surveyquestion("How do you feel about your neighborhood?")}{p_end}

{pstd}With chain-of-thought:{p_end}
{phang2}{cmd:. catllm classify_survey open_ended, categories("Health" "Education" "Economy") apikey($OPENAI_API_KEY) surveyquestion("What is the most important issue facing your community?") chainofthought}{p_end}
