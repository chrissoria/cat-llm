*! version 2.0.0  17may2026
*! catllm -- LLM-powered classification and extraction for text data
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
    else if "`subcmd'" == "cerad" {
        catllm_cerad `0'
    }
    else {
        di as error "unknown subcommand: `subcmd'"
        di as error "type {bf:help catllm} for valid subcommands"
        exit 198
    }
end
