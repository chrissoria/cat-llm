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
* one model pulled. Example: `ollama pull qwen2.5:7b` in a terminal.
*
* Cost:     $0 (runs entirely on your local machine)
* Runtime:  ~30 seconds for 10 rows on Apple Silicon with qwen2.5:7b
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
*     ollama pull qwen2.5:7b

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
catllm classify response,                           ///
    categories("Positive" "Negative" "Neutral")     ///
    apikey("_")                                     ///
    model("qwen2.5:7b")                             ///
    provider("ollama")                              ///
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
* qwen2.5:7b    -- Good baseline, fast on Apple Silicon
* qwen2.5:14b   -- Higher accuracy, ~10 GB RAM
* llama3.1:8b   -- Solid generalist
* gpt-oss:20b   -- Strong for English, slower
*
* See https://ollama.com/library for the full catalog.
