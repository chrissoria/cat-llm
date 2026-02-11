#!/usr/bin/env python
"""
Tests 8-12: step_back_prompt, context_prompt, few-shot examples, safety mode,
consensus threshold variations. Uses only cheap models (OpenAI, Anthropic, Google).
"""

import sys, os, time, traceback
import pandas as pd

src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
sys.path.insert(0, os.path.abspath(src_path))
for mod in [k for k in sys.modules if k.startswith('catllm')]:
    del sys.modules[mod]

from dotenv import load_dotenv
for p in [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'),
    os.path.expanduser('~/.env'),
    '/Users/chrissoria/Documents/Research/Categorization_AI_experiments/.env',
]:
    if os.path.isfile(p):
        load_dotenv(p, override=True)

from catllm import classify

api_keys = {
    "openai": os.getenv("OPENAI_API_KEY"),
    "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    "google": os.getenv("GOOGLE_API_KEY"),
}
print("API Key Status:")
for prov, key in api_keys.items():
    print(f"  {prov:15s} {'Found' if key else 'MISSING'}")
print()

CHEAP = [
    {"name": "OpenAI",    "provider": "openai",    "model": "gpt-4o-mini",               "key": "openai"},
    {"name": "Anthropic", "provider": "anthropic",  "model": "claude-haiku-4-5-20251001", "key": "anthropic"},
    {"name": "Google",    "provider": "google",     "model": "gemini-2.0-flash",          "key": "google"},
]

# Load test data
script_dir = os.path.dirname(os.path.abspath(__file__))
survey_df = pd.read_csv(os.path.join(script_dir, 'test_data', 'survey_responses.csv'))
sentiment_df = survey_df[survey_df['survey_question'] == "How do you feel about this product?"]
moving_df = survey_df[survey_df['survey_question'] == "Why did you move to your current residence?"]
SENT_TEXTS = sentiment_df['response_text'].tolist()
MOVE_TEXTS = moving_df['response_text'].tolist()
CATEGORIES_SENT = ["Positive sentiment", "Negative sentiment", "Neutral sentiment"]
CATEGORIES_MOVE = ["Employment", "Family", "Financial", "Housing",
                   "Education", "Safety", "Lifestyle", "Life transition"]

output_dir = os.path.join(os.path.dirname(script_dir), 'test_output')
os.makedirs(output_dir, exist_ok=True)

all_results = []

def run_test(test_name, cfg, kwargs):
    key = api_keys.get(cfg["key"])
    result = {"test": test_name, "provider": cfg["name"], "model": cfg["model"],
              "status": None, "rows": None, "time_s": None, "error": None}
    if not key:
        result["status"] = "SKIPPED"; result["error"] = "No API key"
        return result
    print(f"  {cfg['name']:15s} ({cfg['model']}) ... ", end="", flush=True)
    t0 = time.time()
    try:
        df = classify(**kwargs, user_model=cfg["model"], model_source=cfg["provider"], api_key=key)
        elapsed = round(time.time() - t0, 1)
        result.update(status="PASS", rows=len(df), time_s=elapsed)
        print(f"PASS  ({len(df)} rows, {elapsed}s)")
    except Exception as e:
        elapsed = round(time.time() - t0, 1)
        result.update(status="FAIL", time_s=elapsed, error=str(e))
        print(f"FAIL  ({elapsed}s)")
        traceback.print_exc(); print()
    return result

# ===================================================================
# TEST 8: Step-back prompting
# ===================================================================
print("=" * 70)
print("TEST 8: Step-back prompting (step_back_prompt=True)")
print("=" * 70)
for cfg in CHEAP:
    r = run_test("stepback", cfg, dict(
        input_data=MOVE_TEXTS[:5], categories=CATEGORIES_MOVE,
        description="Survey responses about reasons for moving",
        survey_question="Why did you move to your current residence?",
        chain_of_thought=True, step_back_prompt=True, creativity=0,
        filename=f"stepback_{cfg['key']}.csv", save_directory=output_dir,
    ))
    all_results.append(r)
print()

# ===================================================================
# TEST 9: Context prompting
# ===================================================================
print("=" * 70)
print("TEST 9: Context prompting (context_prompt=True)")
print("=" * 70)
for cfg in CHEAP:
    r = run_test("context", cfg, dict(
        input_data=SENT_TEXTS[:3], categories=CATEGORIES_SENT,
        description="Customer feedback responses",
        chain_of_thought=True, context_prompt=True, creativity=0,
        filename=f"context_{cfg['key']}.csv", save_directory=output_dir,
    ))
    all_results.append(r)
print()

# ===================================================================
# TEST 10: Few-shot examples
# ===================================================================
print("=" * 70)
print("TEST 10: Few-shot examples (example1, example2, example3)")
print("=" * 70)
for cfg in CHEAP:
    r = run_test("fewshot", cfg, dict(
        input_data=SENT_TEXTS[:3], categories=CATEGORIES_SENT,
        description="Customer feedback responses",
        chain_of_thought=True,
        example1="'Great quality, very satisfied' -> Positive sentiment",
        example2="'Awful, broke after one day' -> Negative sentiment",
        example3="'It works fine, nothing special' -> Neutral sentiment",
        creativity=0,
        filename=f"fewshot_{cfg['key']}.csv", save_directory=output_dir,
    ))
    all_results.append(r)
print()

# ===================================================================
# TEST 11: Safety mode (incremental saves)
# ===================================================================
print("=" * 70)
print("TEST 11: Safety mode (safety=True)")
print("=" * 70)
for cfg in CHEAP:
    r = run_test("safety", cfg, dict(
        input_data=SENT_TEXTS[:3], categories=CATEGORIES_SENT,
        description="Customer feedback responses",
        chain_of_thought=True, safety=True, creativity=0,
        filename=f"safety_{cfg['key']}.csv", save_directory=output_dir,
    ))
    all_results.append(r)
print()

# ===================================================================
# TEST 12: Consensus threshold variations (3-model ensemble)
# ===================================================================
print("=" * 70)
print("TEST 12: Consensus threshold variations (3-model ensemble)")
print("=" * 70)

ensemble_models = []
ensemble_names = []
for cfg in CHEAP:
    key = api_keys.get(cfg["key"])
    if key:
        ensemble_models.append((cfg["model"], cfg["provider"], key))
        ensemble_names.append(cfg["name"])

THRESHOLDS = [
    ("two-thirds", "two-thirds"),
    ("unanimous", "unanimous"),
    ("custom_075", 0.75),
]

if len(ensemble_models) >= 2:
    for label, threshold in THRESHOLDS:
        print(f"  threshold={threshold} ... ", end="", flush=True)
        t0 = time.time()
        try:
            df = classify(
                input_data=MOVE_TEXTS[:5],
                categories=CATEGORIES_MOVE,
                description="Survey responses about reasons for moving",
                models=ensemble_models,
                consensus_threshold=threshold,
                chain_of_thought=True, creativity=0,
                filename=f"threshold_{label}.csv", save_directory=output_dir,
            )
            elapsed = round(time.time() - t0, 1)
            print(f"PASS  ({len(df)} rows, {elapsed}s)")
            all_results.append({"test": "threshold", "provider": f"threshold={label}",
                                "model": "ensemble", "status": "PASS", "rows": len(df),
                                "time_s": elapsed, "error": None})
        except Exception as e:
            elapsed = round(time.time() - t0, 1)
            print(f"FAIL  ({elapsed}s)")
            traceback.print_exc()
            all_results.append({"test": "threshold", "provider": f"threshold={label}",
                                "model": "ensemble", "status": "FAIL", "rows": None,
                                "time_s": elapsed, "error": str(e)})
else:
    print("  SKIPPED - need at least 2 API keys")
    for label, _ in THRESHOLDS:
        all_results.append({"test": "threshold", "provider": f"threshold={label}",
                            "model": "ensemble", "status": "SKIPPED", "rows": None,
                            "time_s": None, "error": "Missing API keys"})
print()

# ===================================================================
# SUMMARY
# ===================================================================
print("=" * 70)
print("SUMMARY")
print("=" * 70)

test_names = {
    "stepback": "Step-Back Prompting",
    "context": "Context Prompting",
    "fewshot": "Few-Shot Examples",
    "safety": "Safety Mode",
    "threshold": "Consensus Thresholds",
}

for test_key, test_label in test_names.items():
    subset = [r for r in all_results if r["test"] == test_key]
    passed = sum(1 for r in subset if r["status"] == "PASS")
    failed = sum(1 for r in subset if r["status"] == "FAIL")
    skipped = sum(1 for r in subset if r["status"] == "SKIPPED")
    print(f"\n  {test_label}:")
    print(f"    Passed: {passed}  |  Failed: {failed}  |  Skipped: {skipped}")
    for r in subset:
        icon = {"PASS": "+", "FAIL": "X", "SKIPPED": "-"}.get(r["status"], "?")
        detail = f"{r['rows']} rows, {r['time_s']}s" if r["status"] == "PASS" else (r["error"] or "")
        print(f"    [{icon}] {r['provider']:20s} {detail}")

total_passed = sum(1 for r in all_results if r["status"] == "PASS")
total_failed = sum(1 for r in all_results if r["status"] == "FAIL")
total_skipped = sum(1 for r in all_results if r["status"] == "SKIPPED")

print(f"\n{'=' * 70}")
print(f"TOTAL:  {total_passed} passed  |  {total_failed} failed  |  {total_skipped} skipped  |  {len(all_results)} total")
print(f"{'=' * 70}")

sys.exit(1 if total_failed > 0 else 0)
