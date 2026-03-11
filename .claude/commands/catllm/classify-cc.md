Classify text data in a CSV or Excel file using Claude Code as the classifier (no API key needed).

Instructions:

## Step 0 — Parse natural language input
Read $ARGUMENTS as a natural-language request. Extract whatever the user provided — any combination of:
- **file** — a file path or name (e.g. "xy.csv", "the survey file")
- **column** — a column name (e.g. "on the response column", "col: feedback")
- **categories** — category names (e.g. "positive, negative, neutral", "for sentiment")
- **description** — data context (e.g. "customer feedback about our product")
- **output** — output file path (e.g. "save to results.csv")

Store whatever you found. For anything not mentioned, you'll ask in later steps.
If the user gave categories (even implicitly like "sentiment"), note them — you can skip straight to confirmation in Step 5.
If the user gave enough to start (file + column + categories), skip directly to the first missing step.

Examples of valid inputs:
- `survey.csv` — just a file, ask everything else
- `classify xy.csv on the response column for positive, negative, neutral sentiment` — file, column, and categories all provided
- `feedback.csv --output results.csv` — file and output, ask column and categories
- (empty) — no arguments, ask for file

## Step 1 — Find the file
Skip if a file was identified in Step 0.

- Run `find . \( -name "*.csv" -o -name "*.xlsx" -o -name "*.xls" \) -not -path "./.git/*" | head -20`
- Use AskUserQuestion to ask: "Which file would you like to classify?"
  List each discovered file as an option (relative path). Add "Other" as the last option.
- If "Other": ask "Describe the file you're looking for:" and search by name/directory, confirm the match before proceeding.
- If the file is .xlsx/.xls with multiple sheets, run `python3 -c "import pandas as pd; print(pd.ExcelFile('PATH').sheet_names)"` and ask which sheet to use if more than one exists.

## Step 2 — Column selection
Read the file with pandas. Show column names, row count, and 3 sample rows.

Skip the question if a column was identified in Step 0 (just confirm: "Using column '{col}'" and proceed).

Otherwise, use AskUserQuestion:
  header: "Text column", question: "Which column contains the text to classify?"
  Auto-detect the most likely text column (longest average string length) and mark it "(Recommended)".
  List every column as an option.

## Step 3 — Scale check
Run Python to count non-null rows in the selected text column.

- If rows <= 200: proceed silently.
- If 200 < rows <= 500: warn the user:
  "This file has {N} rows. Claude Code classification works best under 200 rows. For larger datasets, `/catllm:classify` with an API key is faster and more reliable."
  Use AskUserQuestion:
  - "Continue anyway" — proceed
  - "Switch to /catllm:classify" — tell user to run `/catllm:classify` and stop
- If rows > 500: strongly recommend switching:
  "This file has {N} rows, which exceeds the practical limit for Claude Code classification. I strongly recommend `/catllm:classify` with an API key."
  Use AskUserQuestion:
  - "Continue anyway (not recommended)"
  - "Switch to /catllm:classify" (Recommended) — tell user to run `/catllm:classify` and stop

## Step 4 — Data context
Skip if the user already provided a description in Step 0.

Ask as free text (optional, press Enter to skip):
"Briefly describe your data (e.g. 'Open-ended survey responses about housing decisions'):"
Store as `description`.

## Step 5 — Categories
Skip directly to confirmation (Step 5e) if categories were provided in Step 0.

Otherwise, use AskUserQuestion: "How would you like to define categories?"
Options:
- "Auto-discover from the data" (Recommended) — Claude Code reads samples and suggests themes
- "I'll type them now"

If "I'll type them now": ask "Enter categories, comma-separated:" (free text). Parse into a list. Skip to Step 6.

If "Auto-discover":
  a. Read the first 30 non-null rows from the text column using Python:
  ```python
  import pandas as pd
  df = pd.read_csv("INPUT_FILE")
  samples = df["TEXT_COL"].dropna().head(30).tolist()
  for i, s in enumerate(samples, 1):
      print(f"{i}. {s}")
  ```

  b. You (Claude Code) analyze the printed responses and identify 5-10 recurring themes.
     Consider the data description from Step 4 if provided.
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

  e. If categories came from Step 0 (user typed them in the prompt), show them and confirm:
     "I'll classify using these categories: {list}. Sound good?"
     Use AskUserQuestion:
     - "Yes, proceed" (Recommended)
     - "Edit categories"
     Handle edits same as above.

## Step 6 — Pre-flight checks
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

## Step 7 — Classify (core logic)
Process the data in batches of 20 rows.

### 7a — Load all texts
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

### 7b — For each batch of 20 rows
For batch_start in range(0, n, 20):
  batch_end = min(batch_start + 20, n)

  1. Print the batch texts using Python:
  ```python
  import pandas as pd
  df = pd.read_csv("INPUT_FILE")
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

### 7c — Assemble output DataFrame
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

## Step 8 — Output
Display the result table and category distribution.
Show value counts for each category column:
```python
print("\nCategory distribution:")
for col in result_df.columns[2:]:  # skip survey_input and processing_status
    print(f"\n{col}:")
    print(result_df[col].value_counts().to_string())
```

If an output path was provided (Step 0 or --output flag), save using: `result_df.to_csv("OUTPUT_PATH", index=False)` and confirm.
If no output path, ask user if they want to save the results and where.

## Important notes
- You ARE the classifier. Do not call any external API or use catllm Python functions.
- Classify carefully and consistently. Apply the same standard across all rows.
- Use the data description (Step 4) and verbose category descriptions (Step 6) to guide decisions.
- If a response is ambiguous, prefer marking 0 (not a match) over 1.
- Process every row — do not skip or summarize.
- Be conversational. If the user's initial prompt gave you most of what you need, skip the redundant questions and move fast.

allowed-tools: Bash(python3*), Bash(find*), Read
