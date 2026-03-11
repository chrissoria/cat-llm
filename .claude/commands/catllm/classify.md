Classify text data in a CSV or Excel file using cat-llm.

Instructions:

## Step 0 — Parse natural language input
Read $ARGUMENTS as a natural-language request. Extract whatever the user provided — any combination of:
- **file** — a file path or name (e.g. "xy.csv", "the survey file")
- **column** — a column name (e.g. "on the response column", "col: feedback")
- **categories** — category names (e.g. "positive, negative, neutral", "for sentiment")
- **description** — data context (e.g. "customer feedback about our product")
- **output** — output file path (e.g. "save to results.csv")
- **model location** — cloud, local, or Claude Code (e.g. "using ollama", "no API key")

Store whatever you found. For anything not mentioned, ask in later steps.
If the user provided enough to skip steps, do so — be conversational, not rigid.

Examples of valid inputs:
- `survey.csv` — just a file, ask everything else
- `classify xy.csv on the response column for positive, negative, neutral sentiment` — file, column, and categories provided
- `feedback.csv --output results.csv` — file and output path
- (empty) — no arguments, ask for file

Also accept explicit flags: --categories, --model, --col, --output (same as before).

## Step 1 — Find the file
Skip if a file was identified in Step 0.

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
  (Skip if column was identified in Step 0 or --col was passed.)

Question 2 — header: "Model location", question: "Where would you like to run the model?"
  Options:
  - "Cloud API (Recommended)" — core cat-llm pipeline, empirically validated, requires an API key
  - "Local model (Ollama)" — core cat-llm pipeline, empirically validated, runs on your machine
  - "Claude Code (no API key)" — native Claude Code mode, no setup needed, best for quick/casual use
  (Skip if model location was identified in Step 0.)

Store the choice as `model_location` ("cloud", "local", or "claude_code").

---

# PATH A: Cloud or Local (core cat-llm pipeline)
Follow Steps 3A–8A when `model_location` is "cloud" or "local".

## Step 3A — Model and API key (same screen)
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

## Step 4A — Data context (optional, helps accuracy)
Skip if description was provided in Step 0.

Ask as free text (optional, press Enter to skip):
"Briefly describe your data (e.g. 'Open-ended survey responses about housing decisions'):"
Store as `description`.

## Step 5A — Categories
If categories were provided in Step 0 or --categories was passed, use them directly and skip to Step 6A.

Use AskUserQuestion to ask: "How would you like to define categories?"
Options:
- "Auto-discover from the data" (Recommended) — run extract() first to find themes
- "I'll type them now"

If "I'll type them now": ask "Enter categories, comma-separated:" (free text). Parse into a list. Skip to Step 6A.

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
    research_question="...",       # from step 5Ac, or None if skipped
    focus="...",                   # from step 5Ac, or None if skipped
    specificity="broad",           # "broad" or "specific" — DO NOT use 'specificity_level'
    max_categories=8,              # DO NOT use 'num_categories' or 'n_categories'
    iterations=8,                  # default — do not change
    divisions=12,                  # default — do not change
    creativity=None,               # default — do not change
    user_model="gpt-5",            # from step 3A
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

## Step 6A — Pre-flight checks
Use AskUserQuestion with up to three questions on the same screen:

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

## Step 7A — Run classify
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
    api_key="USER_API_KEY",   # from step 3A — explicit string, never None for cloud
    survey_question="...",    # from step 5Aa, or "" if categories were typed manually
    description="...",        # from step 4A, or "" if skipped
    user_model="gpt-5",       # from step 3A — DO NOT use 'model' or 'model_name'
    model_source="auto",      # "ollama" if local; otherwise always "auto"
    add_other=False,          # already handled interactively in step 6A
    check_verbosity=False,    # already handled interactively in step 6A
    categories_per_call=None, # set to int (e.g. 3) if len(categories) > 10 for better accuracy
)

print(result.to_string())
```

## Step 8A — Output
Display the result table and category distribution (value counts).
If --output was passed or output path was provided in Step 0, save using: `result.to_csv("OUTPUT_PATH", index=False)` and confirm.
If an error occurs, show the full traceback and suggest a specific fix.

---

# PATH B: Claude Code (native classification)
Follow Steps 3B–8B when `model_location` is "claude_code".
You ARE the classifier — do not call any external API or use catllm Python functions.

## Step 3B — Scale check
Run Python to count non-null rows in the selected text column.

- If rows <= 200: proceed silently.
- If 200 < rows <= 500: warn the user:
  "This file has {N} rows. Claude Code classification works best under 200 rows. For larger datasets, the Cloud API or Ollama paths are faster and more reliable."
  Use AskUserQuestion:
  - "Continue anyway" — proceed
  - "Switch to Cloud API" — go back to Step 3A with `model_location = "cloud"`
- If rows > 500: strongly recommend switching:
  "This file has {N} rows, which exceeds the practical limit for Claude Code classification. I strongly recommend using Cloud API or Ollama."
  Use AskUserQuestion:
  - "Continue anyway (not recommended)"
  - "Switch to Cloud API" (Recommended) — go back to Step 3A with `model_location = "cloud"`

## Step 4B — Data context
Skip if description was provided in Step 0.

Ask as free text (optional, press Enter to skip):
"Briefly describe your data (e.g. 'Open-ended survey responses about housing decisions'):"
Store as `description`.

## Step 5B — Categories
If categories were provided in Step 0 or --categories was passed, show them and confirm:
  "I'll classify using these categories: {list}. Sound good?"
  Use AskUserQuestion:
  - "Yes, proceed" (Recommended)
  - "Edit categories"
  Handle edits same as below.

Otherwise, use AskUserQuestion: "How would you like to define categories?"
Options:
- "Auto-discover from the data" (Recommended) — Claude Code reads samples and suggests themes
- "I'll type them now"

If "I'll type them now": ask "Enter categories, comma-separated:" (free text). Parse into a list. Skip to Step 6B.

If "Auto-discover":
  a. Read the first 30 non-null rows from the text column using Python:
  ```python
  import pandas as pd
  df = pd.read_csv("INPUT_FILE")  # pd.read_excel() for .xlsx/.xls
  samples = df["TEXT_COL"].dropna().head(30).tolist()
  for i, s in enumerate(samples, 1):
      print(f"{i}. {s}")
  ```

  b. You (Claude Code) analyze the printed responses and identify 5-10 recurring themes.
     Consider the data description from Step 4B if provided.
     Present the discovered categories to the user as a numbered list.

  c. Use AskUserQuestion to ask:
     "Would you like to use these categories or edit them?"
     Options:
     - "Use these categories" (Recommended)
     - "Edit categories"

     If "Edit categories": present each category as a numbered question, batched 4 at a time.
     Round 1 — categories 1-4 simultaneously:
       Q1 — header: "Category 1", question: "What should category 1 be?"
            Options: - "<current name>" (keep as-is)  - "Other"
       (same pattern for Q2, Q3, Q4)
     Round 2+ — remaining categories in the same pattern.
     If "Other" is selected for any category, immediately ask: "Enter the new name for category N:" (free text).
     Collect all responses into the final `categories` list.

## Step 6B — Pre-flight checks
Use AskUserQuestion with up to two questions on the same screen:

Question A — "Catch-all" (only show if no category contains the word "other", case-insensitive):
  "None of your categories is a catch-all. Adding 'Other' helps avoid forcing ambiguous responses into ill-fitting categories."
  Options:
  - "Add 'Other' as a category" (Recommended)
  - "Keep my categories as-is"

Question B — "Accuracy boost" (always show):
  "Verbose categories (with descriptions and examples) significantly improve classification accuracy. Would you like me to add descriptions automatically?"
  Options:
  - "Yes, maximize accuracy" (Recommended)
  - "No, keep as-is"

  If "Yes, maximize accuracy": you (Claude Code) immediately write a verbose version of each category —
  a 1-2 sentence description plus 1-2 short examples — without calling any API.
  Format: "<Category name> — <description> (e.g. '<example 1>', '<example 2>')"
  Show all verbose categories to the user and ask for confirmation. If confirmed, replace `categories`
  with the verbose list before classification.

## Step 7B — Classify (Claude Code as the model)
Process the data in batches of 20 rows.

### 7B-a — Load all texts
```python
import pandas as pd, json, os, tempfile

df = pd.read_csv("INPUT_FILE")  # pd.read_excel() for .xlsx/.xls
texts = df["TEXT_COL"].fillna("").tolist()
n = len(texts)

# Create temp dir for results
tmp_dir = tempfile.mkdtemp(prefix="catllm_cc_")
results_file = os.path.join(tmp_dir, "results.json")

# Save empty results list
with open(results_file, "w") as f:
    json.dump([], f)

# Print batch info
print(f"Total rows: {n}")
print(f"Batches: {(n + 19) // 20}")
print(f"Categories: {len(categories)}")
print(f"Results file: {results_file}")
```

### 7B-b — For each batch of 20 rows
For batch_start in range(0, n, 20):
  batch_end = min(batch_start + 20, n)

  1. Print the batch texts using Python:
  ```python
  import pandas as pd
  df = pd.read_csv("INPUT_FILE")  # pd.read_excel() for .xlsx/.xls
  texts = df["TEXT_COL"].fillna("").tolist()
  for i in range(BATCH_START, BATCH_END):
      print(f"Row {i}: {texts[i]}")
  ```

  2. You (Claude Code) read each row and classify it against ALL categories.

  **Classification instructions:**
  For each response, decide which categories apply (1) or don't (0).
  Be precise — only mark 1 if the response clearly fits the category.
  When uncertain, mark 0. If no categories fit, mark all 0.

  3. Write the batch results using Python:
  ```python
  import json

  # Categories in order (for reference):
  # CATEGORY_LIST_HERE

  batch_results = [
      # Row BATCH_START: "text preview..."
      {"category_name_1": 1, "category_name_2": 0, ...},
      # Row BATCH_START+1: "text preview..."
      {"category_name_1": 0, "category_name_2": 1, ...},
      # ... one dict per row in batch
  ]

  results_file = "RESULTS_FILE_PATH"
  with open(results_file, "r") as f:
      all_results = json.load(f)
  all_results.extend(batch_results)
  with open(results_file, "w") as f:
      json.dump(all_results, f)
  print(f"Batch done: rows {BATCH_START}-{BATCH_END-1} ({len(all_results)}/{TOTAL} total)")
  ```

  4. Repeat for all batches. Show progress after each batch.

### 7B-c — Assemble output DataFrame
After all batches are complete:
```python
import pandas as pd, json

df = pd.read_csv("INPUT_FILE")  # pd.read_excel() for .xlsx/.xls

results_file = "RESULTS_FILE_PATH"
with open(results_file, "r") as f:
    all_results = json.load(f)

# Build result DataFrame
result_df = pd.DataFrame(all_results)

# Ensure integer types
for col in result_df.columns:
    result_df[col] = pd.to_numeric(result_df[col], errors="coerce").fillna(0).astype("Int64")

# Add survey_input and processing_status columns
result_df.insert(0, "survey_input", df["TEXT_COL"].fillna("").tolist()[:len(result_df)])
result_df.insert(1, "processing_status", "success")

print(result_df.to_string())
print(f"\nShape: {result_df.shape}")
```

Store the final DataFrame as `result_df`.

## Step 8B — Output
Display the result table and category distribution:
```python
print("\nCategory distribution:")
for col in result_df.columns[2:]:  # skip survey_input and processing_status
    print(f"\n{col}:")
    print(result_df[col].value_counts().to_string())
```

If --output was passed or output path was provided in Step 0, save using: `result_df.to_csv("OUTPUT_PATH", index=False)` and confirm.
If no output path, ask user if they want to save the results and where.

## Important notes for Path B (Claude Code)
- You ARE the classifier. Do not call any external API or use catllm Python functions.
- Classify carefully and consistently. Apply the same standard across all rows.
- Use the data description (Step 4B) and verbose category descriptions (Step 6B) to guide decisions.
- If a response is ambiguous, prefer marking 0 (not a match) over 1.
- Process every row — do not skip or summarize.
- Be conversational. If the user's initial prompt gave you most of what you need, skip redundant questions.

allowed-tools: Bash(python3*), Bash(find*), Bash(ollama*), Read
