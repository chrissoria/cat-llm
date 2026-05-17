********************************************************************************
* 01-classifying-text-with-ollama.do
*
* Classify text against a LOCAL Ollama model. Zero API cost, data never
* leaves your machine. Useful when:
*   - You don't want to pay for cloud API calls
*   - Your data can't leave your machine (IRB, HIPAA, internal policy)
*   - You need fully reproducible runs (cloud models change underneath you)
*
* Requires: Ollama installed (https://ollama.com/download) and at least
* one model pulled:
*     ollama pull qwen2.5:14b   // recommended (~9 GB disk, ~10 GB RAM)
*     ollama pull qwen2.5:7b    // fallback if RAM is limited (~4.7 GB)
*
* Cost:     $0 (runs entirely on your local machine)
* Runtime:  ~30-60 seconds for 10 rows on Apple Silicon with qwen2.5:14b
********************************************************************************

clear all
set more off

adopath + "`c(pwd)'/.."   // only needed if running from a git checkout

* --- 1. Confirm the Python backend is installed -------------------------------
catllm setup, check

* --- 2. Make sure Ollama is running -------------------------------------------
* You do NOT need `ollama serve` running manually -- cat-stack's Ollama path
* will check connectivity and surface a clear error if it can't reach Ollama.
* Pull the model once, in a terminal:
*     ollama pull qwen2.5:14b   // recommended
*     ollama pull qwen2.5:7b    // if RAM is limited

* --- 3. Example data ----------------------------------------------------------
input str200 response
"The new wellness program is great, I've been using it daily."
"It's confusing and the app crashes constantly."
"I haven't tried it yet but it sounds promising."
"The interface is intuitive and I like the dashboard."
"Customer support never responded to my emails."
"Pretty average -- nothing special either way."
"Best benefit my employer has ever offered."
"I tried it once and forgot about it."
"The login keeps failing on my phone."
"Really useful for tracking my steps and sleep."
end

* --- 4. Classify with the local Ollama model ---------------------------------
* No real API key is needed for Ollama, but Stata's syntax parser requires
* a non-empty string for the apikey() option -- pass any placeholder.
* `provider("ollama")` tells cat-stack which backend to use.
*
* Verbose, definition-style labels classify several percentage points more
* accurately than one-word labels on small local models.
* Use qwen2.5:14b for best accuracy; substitute qwen2.5:7b if RAM is limited.
catllm classify response,                                                              ///
    categories(                                                                        ///
        "Positive: The respondent expresses satisfaction, approval, or favorable sentiment." ///
        "Negative: The respondent expresses dissatisfaction, frustration, or criticism."    ///
        "Neutral: The respondent is factual, ambivalent, or does not express clear sentiment." ///
        "Other: The response does not fit any of the above categories.")               ///
    apikey("_")                                                                        ///
    model("qwen2.5:14b")                                                               ///
    provider("ollama")                                                                 ///
    generate(sentiment)

list response sentiment, separator(0) abbreviate(20)
tab sentiment

* --- 5. Why local? -----------------------------------------------------------
* - Zero API cost: only your machine's electricity.
* - Data never leaves your network -- key for sensitive survey data.
* - Reproducible: model weights are pinned by `ollama pull <model>:<tag>`.
*
* The trade-off: local 7B-14B models are usually a few percentage points
* behind frontier cloud models. For high-stakes coding, validate against a
* hand-labeled subsample.

* --- 6. Suggested local models -----------------------------------------------
* qwen2.5:14b   -- Recommended: best accuracy/size tradeoff (~10 GB RAM)
* qwen2.5:7b    -- Faster, lower RAM, noticeably weaker (~5 GB RAM)
* llama3.2      -- Good generalist, smaller footprint (~2 GB)
* mistral:7b    -- Solid English, fast on Apple Silicon
*
* See https://ollama.com/library for the full catalog.
