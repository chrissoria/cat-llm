"""
Test the embedding tiebreaker feature on real a19i survey data.

Uses 4 cheap models so that a 2-2 split creates a true tie with majority vote
(positive_rate = 0.5 = threshold). The tiebreaker then uses embedding centroids
to resolve these ties.

We compare the tiebreaker's decisions against the gold-standard hand-coded labels
to see whether the centroid-based resolution is making good decisions.
"""

import os
import pandas as pd
from dotenv import load_dotenv

# Load API keys
load_dotenv('/Users/chrissoria/Documents/Research/Categorization_AI_experiments/.env', override=True)

import catllm as cat

hf_key = os.getenv("HUGGINGFACE_API_KEY")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

# Load a19i data with gold-standard labels
data = pd.read_csv(
    "/Users/chrissoria/Documents/Research/empirical_investigation_llm/data/processed/unique_main_a19i.csv"
)

# Use first 50 rows for more tie opportunities
responses = data["Response"].head(50)
# Use iloc to grab gold label columns (cols 1-6) to avoid smart-quote issues
gold_labels = data.iloc[:50, 1:7].copy()
gold_col_names_actual = list(gold_labels.columns)

print(f"Testing on {len(responses)} responses from a19i\n")

# Concise categories from the empirical study (matched to gold label columns)
categories = [
    "To start living with or stay with partner/spouse.",
    "Relationship change (divorce, breakup, etc).",
    "Person's own job, school, or career change (including transfer/retirement).",
    "Partner's job, school, or career change (including transfer/retirement).",
    "Financial reasons (rent too expensive, pay raise, etc).",
    "Features of the home (bigger/smaller yard, space, etc).",
]

# 4 cheap models — 2-2 split = true tie with majority vote
models = [
    ("meta-llama/Llama-3.3-70B-Instruct", "huggingface", hf_key),
    ("Qwen/Qwen2.5-72B-Instruct", "huggingface", hf_key),
    ("claude-3-haiku-20240307", "anthropic", anthropic_key),
    ("gpt-4o-mini", "openai", openai_key),
]

print("=" * 60)
print("Running 4-model ensemble WITH embedding tiebreaker")
print("=" * 60)
result = cat.classify(
    input_data=responses,
    categories=categories,
    models=models,
    consensus_threshold="majority",
    add_other=False,
    check_verbosity=False,
    embedding_tiebreaker=True,
    min_centroid_size=2,  # lower threshold since only 50 rows
    filename="test_tiebreaker_4models.csv",
    save_directory="examples/output",
)
print(f"\nResult shape: {result.shape}")

# Analyze resolved_by columns
print("\n" + "=" * 60)
print("TIEBREAKER ANALYSIS")
print("=" * 60)

resolved_cols = [c for c in result.columns if "_resolved_by" in c]
gold_col_names = gold_col_names_actual

total_centroid_resolved = 0
correct_centroid = 0
incorrect_centroid = 0
total_vote_resolved = 0
correct_vote = 0
incorrect_vote = 0

for i, (res_col, gold_col) in enumerate(zip(resolved_cols, gold_col_names)):
    cat_num = i + 1
    consensus_col = f"category_{cat_num}_consensus"

    centroid_mask = result[res_col] == "centroid"
    vote_mask = result[res_col] == "vote"

    n_centroid = centroid_mask.sum()
    n_vote = vote_mask.sum()

    print(f"\n  Category {cat_num} ({gold_col}):")
    print(f"    Resolved by vote: {n_vote}, by centroid: {n_centroid}")

    if n_centroid > 0:
        # Compare centroid decisions against gold standard
        centroid_decisions = result.loc[centroid_mask, consensus_col].astype(int)
        centroid_gold = gold_labels.loc[centroid_mask.values, gold_col].astype(int)

        matches = (centroid_decisions.values == centroid_gold.values).sum()
        misses = n_centroid - matches
        correct_centroid += matches
        incorrect_centroid += misses
        total_centroid_resolved += n_centroid

        print(f"    Centroid accuracy: {matches}/{n_centroid} correct ({100*matches/n_centroid:.0f}%)")

        # Show the actual tied texts and decisions
        for idx in result.index[centroid_mask]:
            text = result.loc[idx, "survey_input"][:80]
            decision = result.loc[idx, consensus_col]
            truth = gold_labels.iloc[idx][gold_col]
            status = "OK" if int(decision) == int(truth) else "WRONG"
            print(f"      [{status}] predicted={decision} gold={truth} | {text}...")

    if n_vote > 0:
        vote_decisions = result.loc[vote_mask, consensus_col].astype(int)
        vote_gold = gold_labels.loc[vote_mask.values, gold_col].astype(int)
        matches = (vote_decisions.values == vote_gold.values).sum()
        correct_vote += matches
        incorrect_vote += n_vote - matches
        total_vote_resolved += n_vote

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
if total_centroid_resolved > 0:
    print(f"  Centroid-resolved: {correct_centroid}/{total_centroid_resolved} correct "
          f"({100*correct_centroid/total_centroid_resolved:.1f}%)")
else:
    print("  No ties resolved by centroid")

if total_vote_resolved > 0:
    print(f"  Vote-resolved:     {correct_vote}/{total_vote_resolved} correct "
          f"({100*correct_vote/total_vote_resolved:.1f}%)")

print("\nDone!")
