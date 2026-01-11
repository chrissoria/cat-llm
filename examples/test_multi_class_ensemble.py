#!/usr/bin/env python
"""
Test script for multi_class_ensemble function.

This script tests the multi-model ensemble classification function which
calls multiple LLM models in parallel and combines results using majority voting.

Run with: python examples/test_multi_class_ensemble.py
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
from catllm.text_functions_ensemble import multi_class_ensemble
import catllm

# Load API keys from .env files (try both locations)
load_dotenv('/Users/chrissoria/Documents/Research/cat-llm/.env')
load_dotenv('/Users/chrissoria/Documents/Research/Categorization_AI_experiments/.env')

# Get API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
mistral_api_key = os.getenv("MISTRAL_API_KEY")
xai_api_key = os.getenv("XAI_API_KEY")

print(f"Testing local catllm version: {catllm.__version__}")
print(f"Loaded from: {catllm.__file__}")
print()

# Verify keys loaded
print("API keys loaded:")
print(f"  OpenAI: {'Y' if openai_api_key else 'N'}")
print(f"  Anthropic: {'Y' if anthropic_api_key else 'N'}")
print(f"  Google: {'Y' if google_api_key else 'N'}")
print(f"  Mistral: {'Y' if mistral_api_key else 'N'}")
print(f"  xAI: {'Y' if xai_api_key else 'N'}")
print()

# Output directory
output_dir = os.path.join(os.getcwd(), 'examples', 'test_output')
os.makedirs(output_dir, exist_ok=True)

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
# Model Configurations
# =============================================================================

def get_available_models():
    """Get list of available model configurations based on loaded API keys."""
    models = []

    if openai_api_key:
        models.append(("gpt-4o-mini", "openai", openai_api_key))

    if anthropic_api_key:
        models.append(("claude-3-5-haiku-20241022", "anthropic", anthropic_api_key))

    if google_api_key:
        models.append(("gemini-2.0-flash", "google", google_api_key))

    if mistral_api_key:
        models.append(("mistral-small-latest", "mistral", mistral_api_key))

    return models


# =============================================================================
# Test Functions
# =============================================================================

def test_ensemble(models: list, num_responses: int = 1, verbose: bool = True,
                  chain_of_verification: bool = False, safety: bool = False,
                  safety_filename: str = None) -> dict:
    """
    Test the multi_class_ensemble function.

    Returns:
        dict with test results
    """
    result = {
        "success": False,
        "time": 0,
        "error": None,
        "models_used": [m[0] for m in models],
        "num_responses": num_responses,
        "chain_of_verification": chain_of_verification,
        "safety": safety,
        "results": None,
    }

    if len(models) < 2 and not chain_of_verification and not safety:
        result["error"] = "Need at least 2 models for ensemble testing (or use --cove/--safety for single model)"
        return result

    try:
        start_time = time.time()

        results_dict = multi_class_ensemble(
            survey_input=TEST_RESPONSES[:num_responses],
            categories=TEST_CATEGORIES,
            models=models,
            survey_question="Why did you move to your current residence?",
            creativity=0.1,
            chain_of_thought=True,
            chain_of_verification=chain_of_verification,
            use_json_schema=True,
            consensus_threshold=0.5,
            fail_strategy="partial",
            safety=safety,
            filename=safety_filename if safety else None,
            save_directory=output_dir if safety else None,
        )

        elapsed = time.time() - start_time
        result["time"] = elapsed

        # Handle both single-model (DataFrame) and multi-model (dict) returns
        if isinstance(results_dict, pd.DataFrame):
            combined_df = results_dict
            consensus_df = None
            per_model_dfs = []
        else:
            combined_df = results_dict.get("combined")
            consensus_df = results_dict.get("consensus")
            per_model_dfs = [k for k in results_dict.keys() if k not in ['combined', 'consensus']]

        if combined_df is not None and len(combined_df) > 0:
            # Check if any response was successful
            statuses = combined_df['processing_status'].tolist()
            if 'success' in statuses or 'partial' in statuses:
                result["success"] = True
                result["results"] = {
                    "combined_shape": combined_df.shape,
                    "combined_columns": list(combined_df.columns),
                    "consensus_shape": consensus_df.shape if consensus_df is not None else None,
                    "per_model_dfs": per_model_dfs,
                    "sample_row": combined_df.iloc[0].to_dict(),
                }
            else:
                result["error"] = f"All processing failed: {statuses}"
        else:
            result["error"] = "Empty result DataFrame"

    except Exception as e:
        import traceback
        result["error"] = f"{str(e)}\n{traceback.format_exc()}"

    return result


def print_result(result: dict):
    """Pretty print a test result."""
    status_icon = "PASS" if result["success"] else "FAIL"

    print(f"\n{'='*70}")
    test_type = "ENSEMBLE TEST" if len(result['models_used']) > 1 else "SINGLE MODEL TEST"
    if result.get('chain_of_verification'):
        test_type += " (with CoVe)"
    print(f"{test_type} RESULT: {status_icon}")
    print(f"{'='*70}")
    print(f"  Models used: {', '.join(result['models_used'])}")
    print(f"  Responses:   {result['num_responses']}")
    print(f"  CoVe:        {'Yes' if result.get('chain_of_verification') else 'No'}")
    print(f"  Safety:      {'Yes' if result.get('safety') else 'No'}")
    print(f"  Time:        {result['time']:.2f}s")

    if result["error"]:
        print(f"  Error:       {result['error']}")

    if result["results"]:
        print(f"\n  Output DataFrames:")
        print(f"    Combined shape: {result['results']['combined_shape']}")
        print(f"    Consensus shape: {result['results']['consensus_shape']}")
        print(f"    Per-model DFs: {result['results']['per_model_dfs']}")

        print(f"\n  Combined columns:")
        for col in result['results']['combined_columns']:
            print(f"    - {col}")

        print(f"\n  Sample row (first response):")
        sample = result['results']['sample_row']
        print(f"    Input: {sample.get('survey_input', 'N/A')[:50]}...")
        print(f"    Status: {sample.get('processing_status', 'N/A')}")
        print(f"    Failed models: {sample.get('failed_models', 'N/A')}")

        # Show per-model and consensus results
        print(f"\n  Category results:")
        for i, cat in enumerate(TEST_CATEGORIES, 1):
            print(f"    {i}. {cat}:")
            # Per-model results
            for model_name in result['results']['per_model_dfs']:
                col = f"category_{i}_{model_name}"
                val = sample.get(col, "?")
                print(f"       {model_name}: {val}")
            # Consensus
            consensus_val = sample.get(f"category_{i}_consensus", "?")
            agreement_val = sample.get(f"category_{i}_agreement", "?")
            print(f"       CONSENSUS: {consensus_val} (agreement: {agreement_val})")


def save_results(result: dict, combined_df: pd.DataFrame, timestamp: str):
    """Save results to files."""
    # Save combined DataFrame
    csv_path = os.path.join(output_dir, f"ensemble_test_{timestamp}.csv")
    combined_df.to_csv(csv_path, index=False)
    print(f"\nCombined results saved to: {csv_path}")

    # Save test metadata
    json_path = os.path.join(output_dir, f"ensemble_test_{timestamp}.json")
    with open(json_path, 'w') as f:
        # Make result JSON serializable
        result_copy = result.copy()
        if result_copy.get('results'):
            result_copy['results'] = {
                k: str(v) if not isinstance(v, (str, int, float, bool, list, dict, type(None))) else v
                for k, v in result_copy['results'].items()
            }
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "catllm_version": catllm.__version__,
            "test_responses": TEST_RESPONSES,
            "test_categories": TEST_CATEGORIES,
            "result": result_copy,
        }, f, indent=2, default=str)
    print(f"Test metadata saved to: {json_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test multi_class_ensemble function")
    parser.add_argument("--responses", "-n", type=int, default=1,
                       help="Number of test responses to classify (default: 1)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")
    parser.add_argument("--models", type=int, default=None,
                       help="Number of models to use (default: all available)")
    parser.add_argument("--cove", action="store_true",
                       help="Enable Chain of Verification (single model only recommended)")
    parser.add_argument("--safety", action="store_true",
                       help="Enable safety/incremental saves during processing")

    args = parser.parse_args()

    # Get available models
    available_models = get_available_models()

    print(f"\n{'#'*70}")
    print(f"# CatLLM Ensemble Test")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")

    print(f"\nAvailable models ({len(available_models)}):")
    for model, provider, _ in available_models:
        print(f"  - {model} ({provider})")

    if len(available_models) < 2 and not args.cove and not args.safety:
        print("\nERROR: Need at least 2 models with API keys for ensemble testing.")
        print("       Or use --cove/--safety flag with single model.")
        print("Please set API keys in your .env file.")
        sys.exit(1)

    # For CoVe or safety test with single model, just use first available
    if (args.cove or args.safety) and len(available_models) >= 1:
        if args.models is None:
            args.models = 1  # Default to single model for CoVe/safety

    # Select models to use
    models_to_use = available_models
    if args.models and args.models < len(available_models):
        models_to_use = available_models[:args.models]

    print(f"\nUsing {len(models_to_use)} models for ensemble:")
    for model, provider, _ in models_to_use:
        print(f"  - {model} ({provider})")

    print(f"\nTest data:")
    print(f"  Responses: {args.responses} of {len(TEST_RESPONSES)} available")
    print(f"  Categories: {len(TEST_CATEGORIES)}")
    for i, cat in enumerate(TEST_CATEGORIES, 1):
        print(f"    {i}. {cat}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Generate safety filename if needed
    safety_filename = f"ensemble_safety_{timestamp}.csv" if args.safety else None

    # Run test
    if args.cove:
        print(f"\n>>> Running classification with Chain of Verification...")
    elif args.safety:
        print(f"\n>>> Running classification with safety/incremental saves...")
    else:
        print(f"\n>>> Running ensemble classification...")
    result = test_ensemble(models_to_use, num_responses=args.responses, verbose=args.verbose,
                          chain_of_verification=args.cove, safety=args.safety,
                          safety_filename=safety_filename)

    # Print results
    print_result(result)

    # Save results
    if not args.no_save and result["success"]:
        # Re-run to get the actual DataFrame for saving
        results_dict = multi_class_ensemble(
            survey_input=TEST_RESPONSES[:args.responses],
            categories=TEST_CATEGORIES,
            models=models_to_use,
            survey_question="Why did you move to your current residence?",
            creativity=0.1,
            chain_of_thought=True,
            chain_of_verification=args.cove,
        )
        # Handle both single-model (DataFrame) and multi-model (dict) returns
        if isinstance(results_dict, pd.DataFrame):
            combined_df = results_dict
        else:
            combined_df = results_dict["combined"]
        save_results(result, combined_df, timestamp)

    # Exit code
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
