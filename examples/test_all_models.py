#!/usr/bin/env python
"""
Test script for all models available in the HuggingFace app.

This script tests the unified text classification function with every model
that's available in the CatLLM HuggingFace Space.

Run with: python examples/test_all_models.py
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
from catllm.text_functions import multi_class, detect_provider
import catllm

# Load API keys from .env file
os.chdir('/Users/chrissoria/Documents/Research/Categorization_AI_experiments')
_ = load_dotenv(find_dotenv())
os.chdir('/Users/chrissoria/Documents/Research/cat-llm')

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
print(f"  OpenAI: {'✅' if openai_api_key else '❌'}")
print(f"  Anthropic: {'✅' if anthropic_api_key else '❌'}")
print(f"  Google: {'✅' if google_api_key else '❌'}")
print(f"  Mistral: {'✅' if mistral_api_key else '❌'}")
print(f"  xAI: {'✅' if xai_api_key else '❌'}")
print(f"  HuggingFace: {'✅' if huggingface_api_key else '❌'}")
print(f"  Perplexity: {'✅' if perplexity_api_key else '❌'}")
print()

# Output directory
output_dir = os.path.join(os.getcwd(), 'examples', 'test_output')
os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# Model Configuration (mirrors hf_space/app.py)
# =============================================================================

FREE_MODEL_CHOICES = [
    "Qwen/Qwen3-VL-235B-A22B-Instruct:novita",
    "deepseek-ai/DeepSeek-V3.1:novita",
    "meta-llama/Llama-3.3-70B-Instruct:groq",
    "gemini-2.5-flash",
    "gpt-4o",
    "mistral-medium-2505",
    "claude-3-haiku-20240307",
    "grok-4-fast-non-reasoning",
]

PAID_MODEL_CHOICES = [
    "gpt-4.1",
    "gpt-4o",
    "gpt-4o-mini",
    "claude-sonnet-4-5-20250929",
    "claude-opus-4-20250514",
    "claude-3-5-haiku-20241022",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "mistral-large-latest",
]

# Models routed through HuggingFace
HF_ROUTED_MODELS = [
    "Qwen/Qwen3-VL-235B-A22B-Instruct:novita",
    "deepseek-ai/DeepSeek-V3.1:novita",
    "meta-llama/Llama-3.3-70B-Instruct:groq",
]

# =============================================================================
# Test Data
# =============================================================================

TEST_RESPONSES = [
    "I moved because I got a new job in another city",
    "My family needed to be closer to my elderly parents",
    "The rent was too expensive so I had to find somewhere cheaper",
]

TEST_CATEGORIES = [
    "Employment/Job-related",
    "Family reasons",
    "Financial/Cost of living",
    "Education",
    "Health reasons",
]

# =============================================================================
# Helper Functions
# =============================================================================

def get_provider_for_model(model: str) -> str:
    """Determine the provider for a given model."""
    if model in HF_ROUTED_MODELS:
        return "huggingface"

    model_lower = model.lower()
    if "gpt" in model_lower or "o1" in model_lower:
        return "openai"
    elif "claude" in model_lower:
        return "anthropic"
    elif "gemini" in model_lower:
        return "google"
    elif "mistral" in model_lower:
        return "mistral"
    elif "grok" in model_lower:
        return "xai"
    elif any(x in model_lower for x in [":novita", ":groq", "qwen", "llama", "deepseek"]):
        return "huggingface"
    else:
        return "unknown"


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


def test_single_model(model: str, provider: str, api_key: str, verbose: bool = True) -> dict:
    """
    Test a single model with the unified classification function.

    Returns:
        dict with 'success', 'time', 'error', 'results'
    """
    result = {
        "model": model,
        "provider": provider,
        "success": False,
        "time": 0,
        "error": None,
        "results": None,
    }

    if not api_key:
        result["error"] = f"No API key found for provider '{provider}'"
        return result

    try:
        start_time = time.time()

        df = multi_class(
            survey_input=TEST_RESPONSES[:1],  # Just test with 1 response for speed
            categories=TEST_CATEGORIES,
            api_key=api_key,
            model=model,
            provider=provider,
            survey_question="Why did you move to your current residence?",
            creativity=0.1,
            chain_of_thought=True,
            use_json_schema=True,
        )

        elapsed = time.time() - start_time
        result["time"] = elapsed

        # Check if classification was successful
        if df is not None and len(df) > 0:
            status = df['processing_status'].iloc[0]
            if status == 'success':
                result["success"] = True
                result["results"] = df.to_dict('records')[0]
            else:
                result["error"] = f"Processing status: {status}"
                result["results"] = df.to_dict('records')[0]
        else:
            result["error"] = "Empty result DataFrame"

    except Exception as e:
        result["error"] = str(e)

    return result


def print_result(result: dict, index: int, total: int):
    """Pretty print a test result."""
    status_icon = "✅" if result["success"] else "❌"

    print(f"\n{'='*70}")
    print(f"[{index}/{total}] {result['model']}")
    print(f"{'='*70}")
    print(f"  Provider: {result['provider']}")
    print(f"  Status:   {status_icon} {'PASS' if result['success'] else 'FAIL'}")

    if result["time"] > 0:
        print(f"  Time:     {result['time']:.2f}s")

    if result["error"]:
        print(f"  Error:    {result['error']}")

    if result["results"] and result["success"]:
        print(f"  Categories detected:")
        for i, cat in enumerate(TEST_CATEGORIES, 1):
            val = result["results"].get(f"category_{i}", "?")
            print(f"    {i}. {cat}: {val}")


def print_summary(results: list):
    """Print summary of all test results."""
    passed = sum(1 for r in results if r["success"])
    failed = len(results) - passed

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Total:  {len(results)}")
    print(f"  Passed: {passed} ✅")
    print(f"  Failed: {failed} ❌")
    print(f"{'='*70}")

    if failed > 0:
        print(f"\nFailed models:")
        for r in results:
            if not r["success"]:
                print(f"  - {r['model']}: {r['error']}")

    print(f"\nPassed models:")
    for r in results:
        if r["success"]:
            print(f"  - {r['model']} ({r['time']:.2f}s)")


def save_results(results: list, timestamp: str):
    """Save results to CSV and JSON files."""
    # Flatten results for CSV
    rows = []
    for r in results:
        row = {
            "model": r["model"],
            "provider": r["provider"],
            "success": r["success"],
            "time_seconds": r["time"],
            "error": r["error"],
        }
        # Add category results if available
        if r["results"]:
            for i, cat in enumerate(TEST_CATEGORIES, 1):
                row[f"category_{i}"] = r["results"].get(f"category_{i}", None)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = os.path.join(output_dir, f"model_test_results_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Save detailed JSON
    json_path = os.path.join(output_dir, f"model_test_results_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "catllm_version": catllm.__version__,
            "test_responses": TEST_RESPONSES,
            "test_categories": TEST_CATEGORIES,
            "results": results,
        }, f, indent=2, default=str)
    print(f"Detailed results: {json_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test all models available in CatLLM HuggingFace app")
    parser.add_argument("--free-only", action="store_true", help="Only test free models")
    parser.add_argument("--paid-only", action="store_true", help="Only test paid models")
    parser.add_argument("--model", type=str, help="Test a specific model only")
    parser.add_argument("--list", action="store_true", help="List all available models and exit")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")

    args = parser.parse_args()

    # List models and exit
    if args.list:
        print("FREE MODELS (Space pays):")
        for m in FREE_MODEL_CHOICES:
            provider = get_provider_for_model(m)
            key = get_api_key_for_provider(provider)
            has_key = "✅" if key else "❌"
            print(f"  {has_key} {m} ({provider})")

        print("\nPAID MODELS (User provides key):")
        for m in PAID_MODEL_CHOICES:
            provider = get_provider_for_model(m)
            key = get_api_key_for_provider(provider)
            has_key = "✅" if key else "❌"
            print(f"  {has_key} {m} ({provider})")

        return

    # Build list of models to test
    models_to_test = []

    if args.model:
        # Test specific model
        all_models = FREE_MODEL_CHOICES + PAID_MODEL_CHOICES
        if args.model in all_models:
            models_to_test = [args.model]
        else:
            print(f"Error: Model '{args.model}' not found in available models.")
            print("Use --list to see available models.")
            sys.exit(1)
    elif args.free_only:
        models_to_test = FREE_MODEL_CHOICES
    elif args.paid_only:
        models_to_test = PAID_MODEL_CHOICES
    else:
        # Test all models (free first, then paid)
        models_to_test = FREE_MODEL_CHOICES + PAID_MODEL_CHOICES

    # Remove duplicates while preserving order
    seen = set()
    unique_models = []
    for m in models_to_test:
        if m not in seen:
            seen.add(m)
            unique_models.append(m)
    models_to_test = unique_models

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print(f"\n{'#'*70}")
    print(f"# CatLLM Model Test Suite")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Testing {len(models_to_test)} model(s)")
    print(f"{'#'*70}")

    print(f"\nTest data:")
    print(f"  Responses: {len(TEST_RESPONSES)} sample responses")
    print(f"  Categories: {len(TEST_CATEGORIES)} categories")

    # Run tests
    results = []

    for i, model in enumerate(models_to_test, 1):
        provider = get_provider_for_model(model)
        api_key = get_api_key_for_provider(provider)

        print(f"\n>>> Testing {model} ({provider})...")

        result = test_single_model(model, provider, api_key, verbose=args.verbose)
        results.append(result)

        print_result(result, i, len(models_to_test))

        # Small delay between API calls to avoid rate limiting
        if i < len(models_to_test):
            time.sleep(1)

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
