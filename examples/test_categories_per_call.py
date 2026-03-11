"""
Test categories_per_call parameter on a19i survey data.

Runs 10 rows with:
  1) categories_per_call=None (baseline — all 6 categories in one call)
  2) categories_per_call=2 (3 chunks of 2 categories each)

Compares column structure and prints side-by-side results.
"""

import os
import sys
import pandas as pd
from dotenv import load_dotenv

load_dotenv('/Users/chrissoria/Documents/Research/Categorization_AI_experiments/.env', override=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import catllm as cat

# Load a19i gold standard
data_path = '/Users/chrissoria/Documents/Research/empirical_investigation_llm/data/processed/unique_main_a19i.csv'
df = pd.read_csv(data_path)

# Use first 10 rows
responses = df['Response'].head(10).tolist()
categories = [
    "Partner or spouse",
    "Relationship end",
    "Own job/school/career",
    "Partner's job/school/career",
    "Financial",
    "Housing, concretely",
]

api_key = os.getenv("HUGGINGFACE_API_KEY")
if not api_key:
    print("ERROR: HUGGINGFACE_API_KEY not found")
    sys.exit(1)

common_args = dict(
    input_data=responses,
    categories=categories,
    api_key=api_key,
    user_model="Qwen/Qwen2.5-72B-Instruct",
    model_source="huggingface",
    description="Survey question: Why did you move?",
    add_other=False,
    check_verbosity=False,
    creativity=0.0,
)

# --- Run 1: Baseline (no chunking) ---
print("=" * 60)
print("RUN 1: categories_per_call=None (baseline)")
print("=" * 60)
result_baseline = cat.classify(**common_args)
print(f"\nBaseline shape: {result_baseline.shape}")
print(f"Baseline columns: {list(result_baseline.columns)}")

# --- Run 2: Chunked (2 categories per call) ---
print("\n" + "=" * 60)
print("RUN 2: categories_per_call=2 (3 chunks)")
print("=" * 60)
result_chunked = cat.classify(**common_args, categories_per_call=2)
print(f"\nChunked shape: {result_chunked.shape}")
print(f"Chunked columns: {list(result_chunked.columns)}")

# --- Compare ---
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)

# Chunked should have the same columns + extra "Other" category
baseline_cols = list(result_baseline.columns)
chunked_cols = list(result_chunked.columns)
extra_cols = [c for c in chunked_cols if c not in baseline_cols]
print(f"Baseline columns: {len(baseline_cols)}")
print(f"Chunked columns:  {len(chunked_cols)}")
if extra_cols:
    print(f"Extra columns from unified Other: {extra_cols}")

# Compare the shared category columns (1-6)
shared_cat_cols = [c for c in baseline_cols if c.startswith("category_")]
matches = 0
total = 0
for col in shared_cat_cols:
    for i in range(len(result_baseline)):
        total += 1
        if result_baseline[col].iloc[i] == result_chunked[col].iloc[i]:
            matches += 1

agreement_pct = 100 * matches / total if total > 0 else 0
print(f"Cell-level agreement (real categories): {matches}/{total} ({agreement_pct:.1f}%)")

# Show unified Other column values
other_cols = [c for c in chunked_cols if "other" in c.lower() or c in extra_cols]
if other_cols:
    print(f"\nUnified Other column: {other_cols[0] if other_cols else 'N/A'}")
    for i in range(len(result_chunked)):
        resp = str(responses[i])[:60]
        real_sum = sum(result_chunked[col].iloc[i] for col in shared_cat_cols
                       if col in result_chunked.columns)
        other_val = result_chunked[other_cols[0]].iloc[i] if other_cols else "N/A"
        print(f"  Row {i}: real_sum={real_sum}  Other={other_val}  | {resp}...")

# Show side-by-side for first few rows
print("\nSide-by-side (first 5 rows):")
all_cat_cols = [c for c in chunked_cols if c.startswith("category_")]
for i in range(min(5, len(result_baseline))):
    resp = str(responses[i])[:50]
    print(f"\n  Row {i}: {resp}...")
    for col in all_cat_cols:
        c = result_chunked[col].iloc[i]
        if col in baseline_cols:
            b = result_baseline[col].iloc[i]
            marker = " " if b == c else " <-- DIFF"
            print(f"    {col}: baseline={b}  chunked={c}{marker}")
        else:
            print(f"    {col}: chunked={c}  (unified Other)")

print("\nDone!")
