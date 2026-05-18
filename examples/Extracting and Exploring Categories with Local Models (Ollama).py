"""
Extracting and Exploring Categories with Local Models (Ollama)
==============================================================

This script demonstrates cat.extract() and cat.explore() running entirely
on your local machine via Ollama — no API keys, no cloud costs, full data
privacy.

Data: 100 open-ended responses to the UCNets survey question
      "Why did you move to your current residence?" (variable a19i)

What this script covers:
  1. Checking Ollama status and available models
  2. extract() — normalized, deduplicated category list with "such as" labels
  3. explore() — raw category frequency list for saturation analysis

Requirements:
  - pip install catllm
  - Ollama installed: https://ollama.com/download
  - Ollama server running: ollama serve
  - At least one model pulled, e.g.: ollama pull llama3.2
"""

import random
import pandas as pd
import catllm as cat

# =============================================================================
# 1. Check Ollama
# =============================================================================

if cat.check_ollama_running():
    print("Ollama is running.")
else:
    print("Ollama is not running. Start it with: ollama serve")
    raise SystemExit(1)

models = cat.list_ollama_models()
print(f"\nInstalled models ({len(models)}):")
for m in models:
    print(f"  - {m}")

# Set the model you want to use. Change to any model you have installed.
MODEL = "llama3.2"

if not cat.check_ollama_model(MODEL):
    print(f"\n'{MODEL}' is not installed. Downloading now...")
    cat.pull_ollama_model(MODEL, auto_confirm=True)

# =============================================================================
# 2. Load data — 100 random a19i responses
# =============================================================================

df = pd.read_csv(
    "/Users/chrissoria/Documents/Research/UCNets_Classification/a19i/categorized.csv"
)
responses = (
    df["a19i"]
    .dropna()
    .sample(n=100, random_state=42)
    .tolist()
)

SURVEY_QUESTION = "Why did you move to your current residence?"

print(f"\nLoaded {len(responses)} responses. Sample:")
for r in responses[:5]:
    print(f"  - {r}")

# =============================================================================
# 3. extract() — normalized category list
# =============================================================================
# extract() runs multiple iterations of chunked extraction, then semantically
# merges the results into a clean, deduplicated list with frequency counts.
# Use this when you want a ready-to-use category list — typically as a first
# step before classify().

print("\n" + "=" * 60)
print("EXTRACT")
print("=" * 60)

extract_results = cat.extract(
    input_data=responses,
    api_key="ollama",
    survey_question=SURVEY_QUESTION,
    user_model=MODEL,
    model_source="ollama",
    max_categories=10,
    iterations=3,
)

print("\nTop categories (extract):")
for cat_label in extract_results["top_categories"]:
    print(f"  {cat_label}")

print("\nCategory frequency counts:")
print(extract_results["counts_df"][["Category", "counts"]].head(15).to_string(index=False))

# =============================================================================
# 4. explore() — raw category list for saturation analysis
# =============================================================================
# explore() returns every category string from every chunk across every
# iteration — duplicates intact. Use this to check whether your category
# space is saturating (i.e. new iterations stop producing new categories).

print("\n" + "=" * 60)
print("EXPLORE")
print("=" * 60)

raw_categories = cat.explore(
    input_data=responses,
    api_key="ollama",
    description=SURVEY_QUESTION,
    user_model=MODEL,
    model_source="ollama",
    iterations=3,
)

print(f"\nTotal raw extractions: {len(raw_categories)}")
print(f"Unique categories:     {len(set(raw_categories))}")

# Simple saturation check: how many new unique categories appear each iteration?
divisions = 12
per_iter = len(raw_categories) // 3
seen = set()
print("\nSaturation by iteration:")
for i in range(3):
    chunk = raw_categories[i * per_iter : (i + 1) * per_iter]
    new = set(chunk) - seen
    seen.update(chunk)
    print(f"  Iteration {i+1}: {len(new):3d} new unique categories")

print("\nSample raw categories (first 20):")
for c in raw_categories[:20]:
    print(f"  - {c}")
