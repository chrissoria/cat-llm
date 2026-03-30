*! catllm -- LLM-powered classification and extraction for text data
*! Version 1.1.0
*! Author: Christopher Soria (chrissoria@berkeley.edu)

program define catllm
    version 16

    gettoken subcmd 0 : 0, parse(" ,")

    if "`subcmd'" == "" {
        di as error "subcommand required"
        di as error "type {bf:help catllm} for details"
        exit 198
    }

    local subcmd = lower("`subcmd'")

    * ----- base commands -----
    if "`subcmd'" == "classify" {
        catllm_classify `0'
    }
    else if "`subcmd'" == "extract" {
        catllm_extract `0'
    }
    else if "`subcmd'" == "explore" {
        catllm_explore `0'
    }
    else if "`subcmd'" == "summarize" {
        catllm_summarize `0'
    }
    else if "`subcmd'" == "setup" {
        catllm_setup `0'
    }
    * ----- survey domain -----
    else if "`subcmd'" == "classify_survey" {
        catllm_classify_survey `0'
    }
    else if "`subcmd'" == "extract_survey" {
        catllm_extract_survey `0'
    }
    else if "`subcmd'" == "explore_survey" {
        catllm_explore_survey `0'
    }
    * ----- social media domain -----
    else if "`subcmd'" == "classify_social" {
        catllm_classify_social `0'
    }
    else if "`subcmd'" == "extract_social" {
        catllm_extract_social `0'
    }
    else if "`subcmd'" == "explore_social" {
        catllm_explore_social `0'
    }
    * ----- academic domain -----
    else if "`subcmd'" == "classify_academic" {
        catllm_classify_academic `0'
    }
    else if "`subcmd'" == "extract_academic" {
        catllm_extract_academic `0'
    }
    else if "`subcmd'" == "explore_academic" {
        catllm_explore_academic `0'
    }
    else if "`subcmd'" == "summarize_academic" {
        catllm_summarize_academic `0'
    }
    * ----- cognitive domain -----
    else if "`subcmd'" == "cerad" {
        catllm_cerad `0'
    }
    else {
        di as error "unknown subcommand: `subcmd'"
        di as error "type {bf:help catllm} for valid subcommands"
        exit 198
    }
end
