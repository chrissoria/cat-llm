Extract and discover categories from text data in a CSV or Excel file using cat-llm.

Instructions:

## Step 1 — Find the file
Parse $ARGUMENTS for an optional INPUT_FILE and flags (--col, --model).

If no INPUT_FILE was provided:
- Run `find . \( -name "*.csv" -o -name "*.xlsx" -o -name "*.xls" \) -not -path "./.git/*" | head -20`
- Use AskUserQuestion to ask: "Which file would you like to extract categories from?"
  List each discovered file as an option (relative path). Add "Other" as the last option.
- If "Other": ask "Describe the file you're looking for:" and search by name/directory, confirm before proceeding.
- If the file is .xlsx/.xls with multiple sheets, run `python3 -c "import pandas as pd; print(pd.ExcelFile('PATH').sheet_names)"` and ask which sheet to use if more than one exists.

## Step 2 — Column and model location (same screen)
Read the file with pandas (use `pd.read_csv()` for .csv, `pd.read_excel()` for .xlsx/.xls).
Show column names, row count, and 3 sample rows.

Use AskUserQuestion with two questions on the same screen:

Question 1 — header: "Text column", question: "Which column contains the text to classify?"
  Auto-detect the most likely text column (longest average string length) and mark it "(Recommended)".
  List every column as an option. Add "Other" as the last option.
  (Skip if --col was passed.)

Question 2 — header: "Model location", question: "Where would you like to run the model?"
  Options:
  - "Cloud model" (Recommended) — hosted API, requires an API key
  - "Local model (Ollama)" — runs on your machine, no API key needed

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

  IMPORTANT: extract() requires an explicit api_key string — passing None will raise a ValueError for
  cloud providers. If "I have it in a .env file", run:
  ```python
  from dotenv import load_dotenv; import os; load_dotenv(override=True)
  print(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("GOOGLE_API_KEY") or "")
  ```
  Store the result as `user_api_key`. If empty, warn the user the key was not found in their .env file.

  If local model: set `user_api_key = None`, `model_source = "ollama"`.
  If "Other" cloud model: ask for model name as free text first.

  For Ollama: run `ollama list`, list discovered models as options (up to 6) plus "Other".

## Step 4 — Survey question (required)
Ask as plain free text — keep asking until non-empty:
"What survey question do respondents answer? (e.g. 'Why did you move to this neighborhood?')"
Store as `survey_question`.

## Step 5 — Extraction parameters and optional context (same screen)
Use AskUserQuestion with two questions on the same screen:

Question 1 — header: "Specificity", question: "How specific should the categories be?"
  Options:
  - "Broad" (Recommended) — high-level themes across responses
  - "Specific" — fine-grained, detailed distinctions

Question 2 — header: "Categories", question: "How many categories do you want?"
  Options:
  - "5" — concise, high-level groupings
  - "8" — balanced (Recommended)
  - "12" — more granular
  - "Custom" — I'll type a number

Then ask two optional free-text follow-ups one at a time (press Enter to skip):
- "Any specific research question these categories should serve?" → store as `research_question`
- "Any specific aspect to focus on? (e.g. 'financial concerns')" → store as `focus`

## Step 6 — Run extract()
Use EXACTLY these parameter names (wrong names will cause TypeError):

```python
import pandas as pd, os
from dotenv import load_dotenv
load_dotenv(override=True)
import catllm as cat

df = pd.read_csv("INPUT_FILE")  # pd.read_excel() for .xlsx/.xls
input_data = df["TEXT_COL"]

result = cat.extract(
    input_data,
    api_key="USER_API_KEY",        # REQUIRED explicit string — NOT None for cloud providers
    survey_question="...",         # from step 4 — DO NOT use 'description' or any other name
    research_question="...",       # from step 5, or None if skipped
    focus="...",                   # from step 5, or None if skipped
    specificity="broad",           # "broad" or "specific" — DO NOT use 'specificity_level'
    max_categories=8,              # from step 5 — DO NOT use 'num_categories' or 'n_categories'
    iterations=8,                  # default — do not change
    divisions=12,                  # default — do not change
    creativity=None,               # default — do not change
    user_model="gpt-5",            # from step 3
    model_source="auto",           # "ollama" if local; otherwise always "auto"
)

counts_df = result["counts_df"]
top_categories = result["top_categories"]
print(counts_df.to_string())
print(f"\nTop categories: {top_categories}")
```

## Step 7 — Review and edit categories
Display the discovered categories ranked by count.

Use AskUserQuestion to ask:
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

## Step 8 — Output
Generate and print the suggested `/catllm:classify` follow-up command, pre-filled with:
- the same input file, text column, model, and api key already selected
- `--categories` set to the final confirmed category list

If an error occurs, show the full traceback and suggest a specific fix.

allowed-tools: Bash(python3*), Bash(find*), Bash(ollama*), Read
