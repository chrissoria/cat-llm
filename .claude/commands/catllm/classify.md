Classify text data in a CSV or Excel file using cat-llm.

Instructions:

## Step 1 — Find the file

First, parse $ARGUMENTS for any file path or name the user provided. If found, use it directly.

Otherwise:
- Run `find . \( -name "*.csv" -o -name "*.xlsx" -o -name "*.xls" \) -not -path "./.git/*" | head -20`
- Use AskUserQuestion to ask: "Which file would you like to classify?"
  List each discovered file as an option (relative path).
- If the file is .xlsx/.xls with multiple sheets, run `python3 -c "import pandas as pd; print(pd.ExcelFile('PATH').sheet_names)"` and ask which sheet to use if more than one exists.

## Step 2 — Show data and ask what they want to do

Read the file with pandas. Show column names, row count, and 3 sample rows.

Then probe the environment for API keys:
```python
from dotenv import load_dotenv; import os; load_dotenv(override=True)
keys = {}
for name in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "MISTRAL_API_KEY", "XAI_API_KEY", "HF_TOKEN"]:
    val = os.getenv(name)
    if val:
        keys[name] = val[:8] + "..."
print(keys if keys else "No API keys found")
```

Now present a single open-ended prompt. Include what you know:

"Here's your file: **{filename}** — {N} rows, {M} columns.
Columns: {list}

{If API keys found: "I found API keys in your environment: {key names}. I can use these for cloud classification."}
{If no keys found: "No API keys detected. I can classify directly (up to 200 rows) or you can provide a key."}

**What would you like to do with this data?** Tell me as much as you can — which column to classify, what categories, any context about the data. I'll figure out the rest."

Wait for the user's free-text response.

## Step 3 — Parse the response and fill gaps

From $ARGUMENTS (Step 1) and the user's Step 2 response, extract everything you can:
- **column** — which column to classify
- **categories** — explicit list, or "auto-discover", or a description of what they want (e.g. "sentiment")
- **description** — data context
- **model_location** — "cloud", "local", or "claude_code" (infer from context: "no API key" → claude_code, "ollama" → local, otherwise cloud if keys detected)
- **model** — specific model name
- **output** — output file path
- **survey_question** — what question respondents answered

**Auto-detect what you can:**
- **column**: If not specified, pick the column with the longest average string length.
- **model_location**: If API keys were found and user didn't specify, default to "cloud". If no keys and ≤200 rows, default to "claude_code". If no keys and >200 rows, ask.
- **model**: If cloud and not specified, default to the best model for the detected API key (OpenAI → "gpt-5", Anthropic → "claude-sonnet-4-6", Google → "gemini-2.0-flash").
- **categories**: If user said something like "sentiment" or "topics", interpret as categories. "sentiment" → ["Positive", "Negative", "Neutral", "Mixed"]. For vague requests like "classify it" or "categorize these", treat as auto-discover.

**Only ask follow-ups for truly missing required info.** Batch remaining questions into a single AskUserQuestion call (max 4 questions). Common follow-ups:

- If column is ambiguous (multiple text columns, none obvious): ask which column
- If categories are missing and not "auto-discover": ask for categories or offer auto-discover
- If no API keys and >200 rows and user didn't specify model location: ask cloud vs local (claude_code is capped at 200)

If the user's response gave you everything, skip straight to the appropriate path.

Also accept explicit flags anywhere in the input: --categories, --model, --col, --output.

---

# PATH A: Cloud or Local (core cat-llm pipeline)
Follow when `model_location` is "cloud" or "local".

## Step 4A — Resolve model and API key (if not already known)

**Cloud:** Set `model_source="auto"`.
  - If API key was auto-detected, store as `user_api_key`. Tell the user which key you're using (e.g. "Using your OpenAI key").
  - If no key detected, use AskUserQuestion:
    Options:
    - "I'll type it now"
    - "Where do I get one?" — show: OpenAI → platform.openai.com/api-keys | Anthropic → console.anthropic.com/settings/keys | Google → aistudio.google.com/apikey

  IMPORTANT: When using auto-detected keys, load them properly:
  ```python
  from dotenv import load_dotenv; import os; load_dotenv(override=True)
  user_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
  ```

**Local (Ollama):** Set `model_source = "ollama"`, `user_api_key = None`.
  Run `ollama list`, list discovered models as options (up to 6) plus "Other".
  Store selection as `user_model`.

## Step 5A — Categories (if not already known)

If categories were already extracted from the user's input, skip to Step 6A.

If the user said "auto-discover" or gave a vague request:
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
    user_model="USER_MODEL",       # from step 4A
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

If the user typed categories directly (comma-separated or a clear list), parse them and skip to Step 6A.

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
    api_key="USER_API_KEY",   # from step 4A — explicit string, never None for cloud
    survey_question="...",    # from step 5Aa, or "" if categories were typed manually
    description="...",        # from step 2/3, or "" if not provided
    user_model="USER_MODEL",  # from step 4A — DO NOT use 'model' or 'model_name'
    model_source="auto",      # "ollama" if local; otherwise always "auto"
    add_other=False,          # already handled interactively in step 6A
    check_verbosity=False,    # already handled interactively in step 6A
    categories_per_call=None, # set to int (e.g. 3) if len(categories) > 10 for better accuracy
)

print(result.to_string())
```

## Step 8A — Output
Display the result table and category distribution (value counts).
If output path was provided, save using: `result.to_csv("OUTPUT_PATH", index=False)` and confirm.
If an error occurs, show the full traceback and suggest a specific fix.

---

# PATH B: Claude Code (native classification)
Follow when `model_location` is "claude_code".
You ARE the classifier — do not call any external API or use catllm Python functions.

## Step 3B — Row cap check
Run Python to count non-null rows in the selected text column.

- If rows <= 200: proceed. Show token warning:
  "Note: Claude Code mode uses your token allowance. Each row is classified directly by Claude Code — no API key needed, but tokens are consumed."
- If rows > 200: **hard cap — do not proceed.** Tell the user:
  "This file has {N} rows, which exceeds the 200-row limit for Claude Code classification. Please use Cloud API or Ollama for larger datasets."
  Use AskUserQuestion:
  - "Switch to Cloud API" (Recommended) — go back to Step 4A with `model_location = "cloud"`
  - "Switch to Ollama" — go back to Step 4A with `model_location = "local"`

## Step 4B — Categories (if not already known)

If categories were already extracted from the user's input, show them and confirm:
  "I'll classify using these categories: {list}. Sound good?"
  Use AskUserQuestion:
  - "Yes, proceed" (Recommended)
  - "Edit categories"
  Handle edits same as below.

Otherwise, use AskUserQuestion: "How would you like to define categories?"
Options:
- "Auto-discover from the data" (Recommended) — Claude Code reads samples and suggests themes
- "I'll type them now"

If "I'll type them now": ask "Enter categories, comma-separated:" (free text). Parse into a list. Skip to Step 5B.

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
     Consider the data description if provided.
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

## Step 5B — Pre-flight checks
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

## Step 6B — Classify (Claude Code as the model)
Process the data in batches of 20 rows.

### 6B-a — Load all texts
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

### 6B-b — For each batch of 20 rows
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

### 6B-c — Assemble output DataFrame
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

## Step 7B — Output
Display the result table and category distribution:
```python
print("\nCategory distribution:")
for col in result_df.columns[2:]:  # skip survey_input and processing_status
    print(f"\n{col}:")
    print(result_df[col].value_counts().to_string())
```

If output path was provided, save using: `result_df.to_csv("OUTPUT_PATH", index=False)` and confirm.
If no output path, ask user if they want to save the results and where.

## Important notes for Path B (Claude Code)
- You ARE the classifier. Do not call any external API or use catllm Python functions.
- Classify carefully and consistently. Apply the same standard across all rows.
- Use the data description and verbose category descriptions to guide decisions.
- If a response is ambiguous, prefer marking 0 (not a match) over 1.
- Process every row — do not skip or summarize.

allowed-tools: Bash(python3*), Bash(find*), Bash(ollama*), Read
