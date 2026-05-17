********************************************************************************
* 07-summarizing-text-and-pdf.do
*
* Produce structured summaries of text (and optionally PDF) inputs with
* `catllm summarize`. Useful for digesting long policy documents,
* academic abstracts, or a corpus of social media posts into a
* comparable summary column you can analyze.
*
* Requires: OPENAI_API_KEY
* PDF section additionally requires the PDF extras:
*     catllm setup, pdf
* and a directory of PDFs.
*
* Cost:    ~$0.05 for the text summary, more for PDFs depending on length
* Runtime: ~30 seconds for the text section
********************************************************************************

clear all
set more off

adopath + "`c(pwd)'/.."

global OPENAI_API_KEY    : env OPENAI_API_KEY
global ANTHROPIC_API_KEY : env ANTHROPIC_API_KEY

if "$OPENAI_API_KEY" == "" {
    di as error "OPENAI_API_KEY not set."
    exit 198
}

* --- 1. Summarize text ------------------------------------------------------
input strL doc
"The city council voted 6-3 to approve the new affordable housing ordinance, which requires developers of buildings with 10+ units to set aside 15% as affordable to households earning less than 80% of area median income. The rule applies to all new construction permits issued after January 1. Critics argue the threshold will discourage smaller developers; supporters point to comparable ordinances in three nearby cities that produced over 1,200 affordable units in their first three years."
"After three months of public comment, the planning commission has adopted new zoning rules for the downtown corridor, easing parking minimums for buildings within 1,000 feet of a transit stop. Developers can now build with as little as one parking space per four residential units, down from the previous one-per-unit minimum. The commission's chair said the change is intended to unlock new infill development on parcels where parking requirements had previously made projects financially infeasible."
end

di _n "{hline 60}"
di "1. Paragraph summary with custom instructions"
di "{hline 60}"

catllm summarize doc,                                                  ///
    apikey($OPENAI_API_KEY)                                            ///
    model("gpt-4o-mini")                                               ///
    generate(summary_short)                                            ///
    instructions("Focus on what the policy does and who it affects")

list doc summary_short, separator(0) abbreviate(50)

* --- 2. Try a different output format via pyoptions --------------------------
* cat-stack's summarize accepts a `format` kwarg with values:
*   "paragraph"       -- 2-4 sentence prose (default)
*   "bullets"         -- 3-6 short bullet points
*   "one-liner"       -- single-sentence headline
*   "structured"      -- JSON-like fields (e.g. topic, audience, key_finding)
*   "few-paragraphs"  -- 3-5 paragraph narrative
*   "single-page"     -- one-page structured summary
*   "detailed-report" -- long-form multi-section report
di _n "{hline 60}"
di "2. Bullet-point summary via pyoptions(\"format='bullets'\")"
di "{hline 60}"

catllm summarize doc,                                                  ///
    apikey($OPENAI_API_KEY)                                            ///
    model("gpt-4o-mini")                                               ///
    generate(summary_bullets)                                          ///
    instructions("Summarize each policy as 3 bullets covering what, who, why")  ///
    pyoptions("format='bullets'")

list doc summary_bullets, separator(0) abbreviate(50)

* --- 3. Multi-model ensemble for summaries -----------------------------------
* Same models() pattern as classify. For summaries the output is per-model
* columns rather than a voted consensus -- summarization isn't a discrete
* label, so you typically compare or merge per-model outputs by hand.
if "$ANTHROPIC_API_KEY" != "" {
    di _n "{hline 60}"
    di "3. Ensemble summary (OpenAI + Anthropic)"
    di "{hline 60}"

    local ens_models "gpt-4o-mini openai $OPENAI_API_KEY; claude-haiku-4-5-20251001 anthropic $ANTHROPIC_API_KEY"
    catllm summarize doc,                                                            ///
        apikey($OPENAI_API_KEY)                                                      ///
        generate(summary_ensemble)                                                   ///
        models("`ens_models'")

    list doc summary_ensemble, separator(0) abbreviate(50)
}
else {
    di _n "(Skipping ensemble section -- ANTHROPIC_API_KEY not set)"
}

* --- 4. Summarize PDFs (optional, requires PDF extras) -----------------------
* One-time install:
*     catllm setup, pdf
*
* For PDF input, pass a directory of PDFs as the input variable's contents,
* or build a variable that lists file paths. `pyoptions("input_type='pdf'")`
* tells cat-stack to treat each row as a PDF path.
*
* clear
* input strL pdf_path
* "/Users/me/policy_pdfs/doc1.pdf"
* "/Users/me/policy_pdfs/doc2.pdf"
* end
*
* catllm summarize pdf_path,                                            ///
*     apikey($OPENAI_API_KEY)                                           ///
*     generate(pdf_summary)                                             ///
*     instructions("Summarize each PDF as 5 bullets: what, who, why, when, cost")  ///
*     pyoptions("input_type='pdf', format='bullets', pdf_dpi=150")

di _n "Done."
