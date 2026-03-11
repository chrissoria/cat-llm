Classify text data in a CSV or Excel file using cat-llm.

Instructions:

## Step 1 — Find the file
Parse $ARGUMENTS for an optional INPUT_FILE and flags (--categories, --model, --col, --output).

If no INPUT_FILE was provided:
- Run `find . \( -name "*.csv" -o -name "*.xlsx" -o -name "*.xls" \) -not -path "./.git/*" | head -20`
- Use AskUserQuestion to ask: "Which file would you like to classify?"
  List each discovered file as an option (relative path). Add "Other" as the last option.
- If "Other": ask "Describe the file you're looking for:" and search by name/directory, confirm the match before proceeding.
- If the file is .xlsx/.xls with multiple sheets, run `python3 -c "import pandas as pd; print(pd.ExcelFile('PATH').sheet_names)"` and ask which sheet to use if more than one exists.

## Step 2 — Column and model location (same screen)
Read the file with pandas. Show column names, row count, and 3 sample rows.

Use AskUserQuestion with two questions on the same screen:

Question 1 — header: "Text column", question: "Which column contains the text to classify?"
  Auto-detect the most likely text column (longest average string length) and mark it "(Recommended)".
  List every column as an option. Add "Other" as the last option.
  (Skip if --col was passed.)

Question 2 — header: "Model location", question: "Where would you like to run the model?"
  Options:
  - "Cloud API (Recommended)" — core cat-llm pipeline, empirically validated, requires an API key
  - "Local model (Ollama)" — core cat-llm pipeline, empirically validated, runs on your machine
  - "Claude Code (no API key)" — native Claude Code mode, no setup needed, best for quick/casual use

If "Claude Code (no API key)" is selected: tell the user "Switching to /catllm:classify-cc — this mode uses Claude Code itself as the classifier, no API key needed." Then stop — the user should run `/catllm:classify-cc` with the same file argument.

## Step 3 — Model and API key (same screen)
Use AskUserQuestion with two questions on the same screen:

Question 1 — header: "Model", question: "Which cloud model?" (skip if local was selected)
  Options:
  - "gpt-5" (Recommended) — OpenAI
  - "claude-sonnet-4-6" — Anthropic
  - "gemini-2.0-flash" — Google
  - "Other"
  Store as `user_model`. Always set `model_source="auto"` — the library resolves the provider from the model name automatically. Only set `model_source="ollama"` for local models.

Question 2 — header: "API key", question: "Enter your API key:"
  Options:
  - "I have it in a .env file — use it" (Recommended)
  - "I'll type it now"
  - "Where do I get one?" — show: OpenAI → platform.openai.com/api-keys | Anthropic → console.anthropic.com/settings/keys | Google → aistudio.google.com/apikey

  IMPORTANT: If "I have it in a .env file", do NOT pass api_key=None. Instead run:
  ```python
  from dotenv import load_dotenv; import os; load_dotenv(override=True)
  print(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("GOOGLE_API_KEY") or "")
  ```
  and store the result as `user_api_key`. If empty, warn the user the key was not found.

  If local model: set `user_api_key = None`, `model_source = "ollama"`.
  If "Other" cloud model: ask for model name as free text first.

  For Ollama: run `ollama list`, list discovered models as options (up to 6) plus "Other".

## Step 4 — Data context (optional, helps accuracy)
Ask as free text (optional, press Enter to skip):
"Briefly describe your data (e.g. 'Open-ended survey responses about housing decisions'):"
Store as `description`.

## Step 5 — Categories
If --categories was passed, use it directly and skip to Step 6.

Use AskUserQuestion to ask: "How would you like to define categories?"
Options:
- "Auto-discover from the data" (Recommended) — run extract() first to find themes
- "I'll type them now"

If "I'll type them now": ask "Enter categories, comma-separated:" (free text). Parse into a list. Skip to Step 6.

If "Auto-discover":
  a. Ask as plain free text (required — keep asking until non-empty):
     "What survey question do respondents answer? (e.g. 'Why did you move to this neighborhood?')"
     Store as `survey_question`.

  b. Use AskUserQuestion with two questions on the same screen:
     Question 1 — header: "Specificity", question: "How specific should the categories be?"
       Options:
       - "Broad" (Recommended) — high-level themes
       - "Specific" — fine-grained distinctions

     Question 2 — header: "Categories", question: "How many categories do you want?"
       Options:
       - "5" — concise, high-level groupings
       - "8" — balanced (Recommended)
       - "12" — more granular
       - "Custom" — I'll type a number

  c. Ask two optional free-text follow-ups one at a time (press Enter to skip):
     - "Any specific research question these categories should serve?" → store as `research_question`
     - "Any specific aspect to focus on? (e.g. 'financial concerns')" → store as `focus`

  d. Run cat.extract() — use EXACTLY these parameter names (wrong names will cause TypeError):

```python
import pandas as pd, os
from dotenv import load_dotenv
load_dotenv(override=True)
import catllm as cat

df = pd.read_csv("INPUT_FILE")  # pd.read_excel() for .xlsx/.xls
input_data = df["TEXT_COL"]

result = cat.extract(
    input_data,
    api_key="USER_API_KEY",        # REQUIRED — must be explicit string, NOT None for cloud providers
    survey_question="...",         # REQUIRED — DO NOT use 'description' or any other name
    research_question="...",       # from step 5c, or None if skipped
    focus="...",                   # from step 5c, or None if skipped
    specificity="broad",           # "broad" or "specific" — DO NOT use 'specificity_level'
    max_categories=8,              # DO NOT use 'num_categories' or 'n_categories'
    iterations=8,                  # default — do not change
    divisions=12,                  # default — do not change
    creativity=None,               # default — do not change
    user_model="gpt-5",            # from step 3
    model_source="auto",           # "ollama" if local; otherwise always "auto"
)

top_categories = result["top_categories"]
print(top_categories)
```

  e. Display the discovered categories. Use AskUserQuestion to ask:
     "Would you like to use these categories or edit them?"
     Options:
     - "Use these categories" (Recommended)
     - "Edit categories"

     If "Edit categories": present each category as a numbered question, batched 4 at a time.
     Round 1 — categories 1–4 simultaneously:
       Q1 — header: "Category 1", question: "What should category 1 be?"
            Options: - "<current name>" (keep as-is)  - "Other"
       (same pattern for Q2, Q3, Q4)
     Round 2+ — remaining categories in the same pattern.
     If "Other" is selected for any category, immediately ask: "Enter the new name for category N:" (free text).
     Collect all responses into the final `categories` list.

## Step 6 — Pre-flight checks
Use AskUserQuestion with up to two questions on the same screen:

Question A — "Catch-all" (only show if no category contains the word "other", case-insensitive):
  "None of your categories is a catch-all. Adding 'Other' helps avoid forcing ambiguous responses into ill-fitting categories."
  Options:
  - "Add 'Other' as a category" (Recommended)
  - "Keep my categories as-is"

Question B — "Chunk size" (only show if len(categories) > 8):
  "You have {N} categories. Splitting them into smaller chunks per LLM call can improve accuracy for large category lists."
  Options:
  - "Auto (3–5 per call)" (Recommended) — set `categories_per_call` to `min(5, max(3, len(categories) // 3))`
  - "All at once" — set `categories_per_call = None`

Question C — "Accuracy boost" (always show):
  "Verbose categories (with descriptions and examples) significantly improve classification accuracy. Would you like Claude to add descriptions automatically?"
  Options:
  - "Yes, maximize accuracy" (Recommended)
  - "No, keep as-is"

  If "Yes, maximize accuracy": you (Claude) immediately write a verbose version of each category —
  a 1–2 sentence description plus 1–2 short examples — without calling any API.
  Format: "<Category name> — <description> (e.g. '<example 1>', '<example 2>')"
  Show all verbose categories to the user and ask for confirmation. If confirmed, replace `categories`
  with the verbose list before the classify call.

## Step 7 — Run classify
Use EXACTLY these parameter names (wrong names will cause TypeError):

```python
import pandas as pd, os
from dotenv import load_dotenv
load_dotenv(override=True)
import catllm as cat

df = pd.read_csv("INPUT_FILE")  # pd.read_excel() for .xlsx/.xls
input_data = df["TEXT_COL"]

result = cat.classify(
    input_data,
    categories,               # final list after pre-flight checks (verbose if user accepted)
    api_key="USER_API_KEY",   # from step 3 — explicit string, never None for cloud
    survey_question="...",    # from step 5a, or "" if categories were typed manually
    description="...",        # from step 4, or "" if skipped
    user_model="gpt-5",       # from step 3 — DO NOT use 'model' or 'model_name'
    model_source="auto",      # "ollama" if local; otherwise always "auto"
    add_other=False,          # already handled interactively in step 6
    check_verbosity=False,    # already handled interactively in step 6
    categories_per_call=None, # set to int (e.g. 3) if len(categories) > 10 for better accuracy
)

print(result.to_string())
```

## Step 8 — Output
Display the result table and category distribution (value counts).
If --output was passed, save using: `result.to_csv("OUTPUT_PATH", index=False)` and confirm.
If an error occurs, show the full traceback and suggest a specific fix.

allowed-tools: Bash(python3*), Bash(find*), Bash(ollama*), Read
