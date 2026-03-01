*! catllm -- LLM-powered classification and extraction for survey data
*! Version 1.0.0
*! Author: Christopher Soria (chrissoria@berkeley.edu)

program define catllm
    version 16

    gettoken subcmd 0 : 0, parse(" ,")

    if "`subcmd'" == "" {
        di as error "subcommand required: classify, extract, explore, summarize, or setup"
        di as error "type {bf:help catllm} for details"
        exit 198
    }

    local subcmd = lower("`subcmd'")

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
    else {
        di as error "unknown subcommand: `subcmd'"
        di as error "valid subcommands: classify, extract, explore, summarize, setup"
        exit 198
    }
end
