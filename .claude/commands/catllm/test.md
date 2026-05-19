Run a quick end-to-end test of cat-llm classify on sample survey data.

Usage: /catllm:test [--model MODEL] [--provider PROVIDER]

Arguments:
- --model: Model to test (default: gpt-5)
- --provider: Provider name (default: openai); used only for display

Instructions:
1. Parse --model and --provider from $ARGUMENTS. Default model is "gpt-5".
2. Run the following Python snippet:

```python
import pandas as pd
import catllm as cat

test_file = "examples/test_data/survey_responses.csv"
categories = ["Positive", "Negative", "Neutral"]
model = "gpt-5"  # replace with --model value

print(f"=== cat-llm Quick Test ===")
print(f"File: {test_file}")
print(f"Model: {model}")
print(f"Categories: {categories}\n")

df = pd.read_csv(test_file)
print(f"Loaded {len(df)} rows. Columns: {list(df.columns)}\n")

# Auto-detect text column
text_col = max(df.select_dtypes(include="object").columns,
               key=lambda c: df[c].dropna().str.len().mean())
print(f"Text column: {text_col}\n")

input_data = df[text_col]

result = cat.classify(input_data, categories, user_model=model)

print("--- Results ---")
print(result.to_string())
print("\n--- Category Distribution ---")
# Find the classification column (last non-text column)
cat_col = [c for c in result.columns if c != "text"][-1]
print(result[cat_col].value_counts().to_string())
print("\nPASS: classify() completed successfully.")
```

3. Report PASS if classify() completes without errors and returns a DataFrame.
4. Report FAIL with the full traceback if any exception occurs.
5. Show the result table and category distribution regardless of pass/fail.

allowed-tools: Bash(python3*), Read
