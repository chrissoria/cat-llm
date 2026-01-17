#!/usr/bin/env python
"""
Test script for catllm.extract() function across all supported models.

This script tests the extract function with the 'focus' parameter
across all LLM providers supported by CatLLM.

Run with: python examples/test_explore_corpus.py --model gemini-2.5-flash --provider google
"""

import sys
import os

# Add the src directory to path so we import from local code, not installed package
src_path = '/Users/chrissoria/Documents/Research/cat-llm/src'
sys.path.insert(0, src_path)

# Remove any cached catllm modules to ensure we load from local
modules_to_remove = [key for key in sys.modules.keys() if key.startswith('catllm')]
for mod in modules_to_remove:
    del sys.modules[mod]

import pandas as pd
import argparse
import time
import json
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from catllm import extract
from catllm.text_functions import detect_provider
import catllm

# Load API keys from .env file
load_dotenv('/Users/chrissoria/Documents/Research/Categorization_AI_experiments/.env')

# Get API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
mistral_api_key = os.getenv("MISTRAL_API_KEY")
xai_api_key = os.getenv("XAI_API_KEY")
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")

print(f"Testing local catllm version: {catllm.__version__}")
print(f"Loaded from: {catllm.__file__}")
print()

# Verify keys loaded
print("API keys loaded:")
print(f"  OpenAI: {'Yes' if openai_api_key else 'No'}")
print(f"  Anthropic: {'Yes' if anthropic_api_key else 'No'}")
print(f"  Google: {'Yes' if google_api_key else 'No'}")
print(f"  Mistral: {'Yes' if mistral_api_key else 'No'}")
print(f"  xAI: {'Yes' if xai_api_key else 'No'}")
print(f"  HuggingFace: {'Yes' if huggingface_api_key else 'No'}")
print(f"  Perplexity: {'Yes' if perplexity_api_key else 'No'}")
print()

# Output directory
output_dir = os.path.join(os.getcwd(), 'examples', 'test_output')
os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# Model Configuration
# =============================================================================

# One model per provider for testing
TEST_MODELS = [
    ("gpt-4o-mini", "openai"),
    ("claude-3-haiku-20240307", "anthropic"),
    ("gemini-2.5-flash", "google"),
    ("mistral-small-latest", "mistral"),
    ("grok-3-mini-fast", "xai"),
    ("meta-llama/Llama-3.3-70B-Instruct", "huggingface"),
]

# =============================================================================
# Test Data
# =============================================================================

TEST_SURVEY_QUESTION = "Why did you move to your current residence?"

# Load test responses from HuggingFace app example data
_example_data_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'hf_space', 'example_data.csv'
)
_example_df = pd.read_csv(_example_data_path)
TEST_RESPONSES = _example_df['Response'].tolist()

TEST_FOCUS = "decisions related to moving and relocation"

# =============================================================================
# Helper Functions
# =============================================================================

def get_api_key_for_provider(provider: str) -> str:
    """Get API key from environment for a provider."""
    key_map = {
        "openai": openai_api_key,
        "anthropic": anthropic_api_key,
        "google": google_api_key,
        "mistral": mistral_api_key,
        "xai": xai_api_key,
        "huggingface": huggingface_api_key,
        "perplexity": perplexity_api_key,
    }
    return key_map.get(provider, "")


def test_extract(
    model: str,
    provider: str,
    api_key: str,
    use_focus: bool = True,
) -> dict:
    """
    Test extract with a single model.

    Returns:
        dict with 'success', 'time', 'error', 'categories', 'num_categories'
    """
    result = {
        "model": model,
        "provider": provider,
        "use_focus": use_focus,
        "success": False,
        "time": 0,
        "error": None,
        "categories": None,
        "num_categories": 0,
    }

    if not api_key:
        result["error"] = f"No API key found for provider '{provider}'"
        return result

    try:
        start_time = time.time()

        extraction_result = extract(
            input_data=TEST_RESPONSES,
            api_key=api_key,
            input_type="text",
            description=TEST_SURVEY_QUESTION,
            user_model=model,
            model_source=provider,
            specificity="broad",
            categories_per_chunk=5,
            divisions=2,
            creativity=0.3,
            filename=None,  # Don't save to file during test
            focus=TEST_FOCUS if use_focus else None,
        )

        elapsed = time.time() - start_time
        result["time"] = elapsed

        # Check if extraction was successful
        # extract() returns a dict with 'counts_df', 'top_categories', 'raw_top_text'
        if extraction_result is not None and 'counts_df' in extraction_result:
            df = extraction_result['counts_df']
            if df is not None and len(df) > 0:
                result["success"] = True
                result["categories"] = df['Category'].tolist()
                result["num_categories"] = len(df)
            else:
                result["error"] = "Empty result DataFrame"
        else:
            result["error"] = "Invalid extraction result"

    except Exception as e:
        result["error"] = str(e)

    return result


def print_result(result: dict, index: int, total: int):
    """Pretty print a test result."""
    status_icon = "PASS" if result["success"] else "FAIL"

    print(f"\n{'='*70}")
    print(f"[{index}/{total}] {result['model']}")
    print(f"{'='*70}")
    print(f"  Provider:   {result['provider']}")
    print(f"  Focus used: {result['use_focus']}")
    print(f"  Status:     {status_icon}")

    if result["time"] > 0:
        print(f"  Time:       {result['time']:.2f}s")

    if result["error"]:
        print(f"  Error:      {result['error']}")

    if result["categories"] and result["success"]:
        print(f"  Categories extracted ({result['num_categories']}):")
        for i, cat in enumerate(result["categories"][:10], 1):  # Show first 10
            print(f"    {i}. {cat}")
        if result["num_categories"] > 10:
            print(f"    ... and {result['num_categories'] - 10} more")


def print_summary(results: list):
    """Print summary of all test results."""
    passed = sum(1 for r in results if r["success"])
    failed = len(results) - passed

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Total:  {len(results)}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"{'='*70}")

    if failed > 0:
        print(f"\nFailed models:")
        for r in results:
            if not r["success"]:
                print(f"  - {r['model']}: {r['error']}")

    print(f"\nPassed models:")
    for r in results:
        if r["success"]:
            print(f"  - {r['model']} ({r['time']:.2f}s, {r['num_categories']} categories)")


def save_results(results: list, timestamp: str):
    """Save results to CSV and JSON files."""
    # Flatten results for CSV
    rows = []
    for r in results:
        row = {
            "model": r["model"],
            "provider": r["provider"],
            "use_focus": r["use_focus"],
            "success": r["success"],
            "time_seconds": r["time"],
            "num_categories": r["num_categories"],
            "error": r["error"],
            "categories": "; ".join(r["categories"]) if r["categories"] else None,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = os.path.join(output_dir, f"extract_test_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Save detailed JSON
    json_path = os.path.join(output_dir, f"extract_test_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "catllm_version": catllm.__version__,
            "test_survey_question": TEST_SURVEY_QUESTION,
            "test_focus": TEST_FOCUS,
            "num_test_responses": len(TEST_RESPONSES),
            "results": results,
        }, f, indent=2, default=str)
    print(f"Detailed results: {json_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test catllm.extract() function across all providers")
    parser.add_argument("--model", type=str, help="Test a specific model only")
    parser.add_argument("--provider", type=str, help="Test a specific provider only")
    parser.add_argument("--no-focus", action="store_true", help="Test without the focus parameter")
    parser.add_argument("--both", action="store_true", help="Test both with and without focus")
    parser.add_argument("--list", action="store_true", help="List all available models and exit")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")

    args = parser.parse_args()

    # List models and exit
    if args.list:
        print("TEST MODELS:")
        for model, provider in TEST_MODELS:
            key = get_api_key_for_provider(provider)
            has_key = "Yes" if key else "No"
            print(f"  [{has_key}] {model} ({provider})")
        return

    # Build list of models to test
    models_to_test = []

    if args.model:
        # Test specific model
        provider = args.provider or detect_provider(args.model)
        models_to_test = [(args.model, provider)]
    elif args.provider:
        # Test all models for a specific provider
        models_to_test = [(m, p) for m, p in TEST_MODELS if p == args.provider]
        if not models_to_test:
            print(f"Error: No models found for provider '{args.provider}'")
            sys.exit(1)
    else:
        models_to_test = TEST_MODELS

    # Determine focus testing mode
    if args.both:
        focus_modes = [True, False]
    elif args.no_focus:
        focus_modes = [False]
    else:
        focus_modes = [True]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print(f"\n{'#'*70}")
    print(f"# CatLLM extract() Test Suite")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Testing {len(models_to_test)} model(s)")
    print(f"# Focus modes: {focus_modes}")
    print(f"{'#'*70}")

    print(f"\nTest configuration:")
    print(f"  Survey question: {TEST_SURVEY_QUESTION}")
    print(f"  Focus: {TEST_FOCUS}")
    print(f"  Responses: {len(TEST_RESPONSES)} sample responses")
    print(f"  Categories per chunk: 5")
    print(f"  Divisions: 2")

    # Run tests
    results = []
    total_tests = len(models_to_test) * len(focus_modes)
    test_num = 0

    for model, provider in models_to_test:
        api_key = get_api_key_for_provider(provider)

        for use_focus in focus_modes:
            test_num += 1
            focus_str = "with focus" if use_focus else "without focus"
            print(f"\n>>> Testing {model} ({provider}) {focus_str}...")

            result = test_extract(model, provider, api_key, use_focus=use_focus)
            results.append(result)

            print_result(result, test_num, total_tests)

            # Small delay between API calls to avoid rate limiting
            if test_num < total_tests:
                time.sleep(2)

    # Print summary
    print_summary(results)

    # Save results
    if not args.no_save:
        save_results(results, timestamp)

    # Exit with error code if any tests failed
    failed_count = sum(1 for r in results if not r["success"])
    sys.exit(1 if failed_count > 0 else 0)


if __name__ == "__main__":
    main()
