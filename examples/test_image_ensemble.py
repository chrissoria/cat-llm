#!/usr/bin/env python
"""
Test script for image classification with multi_class_ensemble.

Run with: python examples/test_image_ensemble.py
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
from datetime import datetime
from dotenv import load_dotenv
from catllm.text_functions_ensemble import multi_class_ensemble
import catllm

# Load API keys from .env files
load_dotenv('/Users/chrissoria/Documents/Research/cat-llm/.env')
load_dotenv('/Users/chrissoria/Documents/Research/Categorization_AI_experiments/.env')

# Get API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

print(f"Testing local catllm version: {catllm.__version__}")
print(f"Loaded from: {catllm.__file__}")
print()

# Verify keys loaded
print("API keys loaded:")
print(f"  OpenAI: {'Y' if openai_api_key else 'N'}")
print(f"  Anthropic: {'Y' if anthropic_api_key else 'N'}")
print()

# Test images
TEST_IMAGES = [
    "/Users/chrissoria/Documents/Research/cat-llm/tests/9924971435_3ac76eed4f_c.jpg",
    "/Users/chrissoria/Documents/Research/cat-llm/tests/9688423031_065013996f_c.jpg",
    "/Users/chrissoria/Documents/Research/cat-llm/tests/9688419667_16c304583c_c.jpg",
]

# Categories for image classification
TEST_CATEGORIES = [
    "Contains people",
    "Outdoor scene",
    "Contains buildings/architecture",
    "Contains nature/landscape",
    "Contains vehicles",
]

# Output directory
output_dir = os.path.join(os.getcwd(), 'examples', 'test_output')
os.makedirs(output_dir, exist_ok=True)


def test_single_model_image():
    """Test image classification with a single model."""
    print("\n" + "="*70)
    print("TEST: Single Model Image Classification")
    print("="*70)

    if not openai_api_key:
        print("ERROR: No OpenAI API key available")
        return None

    print(f"Images to classify: {len(TEST_IMAGES)}")
    for img in TEST_IMAGES:
        print(f"  - {os.path.basename(img)}")

    print(f"\nCategories ({len(TEST_CATEGORIES)}):")
    for i, cat in enumerate(TEST_CATEGORIES, 1):
        print(f"  {i}. {cat}")

    print("\n>>> Running classification with gpt-4o-mini...")

    try:
        result = multi_class_ensemble(
            survey_input=TEST_IMAGES,
            categories=TEST_CATEGORIES,
            models=[("gpt-4o-mini", "openai", openai_api_key)],
            input_description="Photographs to classify",
            creativity=0.1,
            chain_of_thought=True,
            use_json_schema=True,
        )

        print("\nRESULT:")
        print(f"  Type: {type(result)}")

        if isinstance(result, pd.DataFrame):
            print(f"  Shape: {result.shape}")
            print(f"  Columns: {list(result.columns)}")
            print("\n  Sample results:")
            for idx, row in result.iterrows():
                img_path = row.get('image_path', 'N/A')
                status = row.get('processing_status', 'N/A')
                print(f"\n  Image: {os.path.basename(img_path) if img_path else 'N/A'}")
                print(f"  Status: {status}")
                for i, cat in enumerate(TEST_CATEGORIES, 1):
                    val = row.get(f'category_{i}_gpt_4o_mini', '?')
                    print(f"    {cat}: {val}")

            # Save results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_path = os.path.join(output_dir, f"image_single_model_{timestamp}.csv")
            result.to_csv(csv_path, index=False)
            print(f"\n  Results saved to: {csv_path}")

            return result
        else:
            print(f"  Unexpected return type: {type(result)}")
            return None

    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
        return None


def test_ensemble_image():
    """Test image classification with multiple models (ensemble)."""
    print("\n" + "="*70)
    print("TEST: Ensemble Image Classification")
    print("="*70)

    models = []
    if openai_api_key:
        models.append(("gpt-4o-mini", "openai", openai_api_key))
    if anthropic_api_key:
        models.append(("claude-3-5-haiku-20241022", "anthropic", anthropic_api_key))

    if len(models) < 2:
        print("WARNING: Need at least 2 models for ensemble test. Skipping...")
        return None

    print(f"Models to use: {[m[0] for m in models]}")
    print(f"Images to classify: {len(TEST_IMAGES)}")

    print("\n>>> Running ensemble classification...")

    try:
        result = multi_class_ensemble(
            survey_input=TEST_IMAGES,
            categories=TEST_CATEGORIES,
            models=models,
            input_description="Photographs to classify",
            creativity=0.1,
            chain_of_thought=True,
            use_json_schema=True,
            consensus_threshold=0.5,
        )

        print("\nRESULT:")
        print(f"  Type: {type(result)}")

        if isinstance(result, dict):
            combined_df = result.get('combined')
            consensus_df = result.get('consensus')

            print(f"  Combined shape: {combined_df.shape if combined_df is not None else 'N/A'}")
            print(f"  Consensus shape: {consensus_df.shape if consensus_df is not None else 'N/A'}")

            if combined_df is not None:
                print(f"\n  Combined columns: {list(combined_df.columns)}")

                print("\n  Consensus results:")
                for idx, row in combined_df.iterrows():
                    img_path = row.get('image_path', 'N/A')
                    status = row.get('processing_status', 'N/A')
                    print(f"\n  Image: {os.path.basename(img_path) if img_path else 'N/A'}")
                    print(f"  Status: {status}")
                    for i, cat in enumerate(TEST_CATEGORIES, 1):
                        consensus = row.get(f'category_{i}_consensus', '?')
                        agreement = row.get(f'category_{i}_agreement', '?')
                        print(f"    {cat}: {consensus} (agreement: {agreement})")

                # Save results
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                csv_path = os.path.join(output_dir, f"image_ensemble_{timestamp}.csv")
                combined_df.to_csv(csv_path, index=False)
                print(f"\n  Results saved to: {csv_path}")

            return result
        else:
            print(f"  Unexpected return type: {type(result)}")
            return None

    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
        return None


def test_directory_input():
    """Test image classification with directory input (instead of file list)."""
    print("\n" + "="*70)
    print("TEST: Directory Input for Images")
    print("="*70)

    if not openai_api_key:
        print("ERROR: No OpenAI API key available")
        return None

    # Use the tests directory containing the images
    test_dir = "/Users/chrissoria/Documents/Research/cat-llm/tests"

    print(f"Input directory: {test_dir}")
    print(f"\nCategories ({len(TEST_CATEGORIES)}):")
    for i, cat in enumerate(TEST_CATEGORIES, 1):
        print(f"  {i}. {cat}")

    print("\n>>> Running classification with directory input...")

    try:
        result = multi_class_ensemble(
            survey_input=test_dir,  # Directory path instead of file list
            categories=TEST_CATEGORIES,
            models=[("gpt-4o-mini", "openai", openai_api_key)],
            input_description="Photographs to classify",
            creativity=0.1,
            chain_of_thought=True,
            use_json_schema=True,
        )

        print("\nRESULT:")
        print(f"  Type: {type(result)}")

        if isinstance(result, pd.DataFrame):
            print(f"  Shape: {result.shape}")
            print(f"  Images found: {len(result)}")

            for idx, row in result.iterrows():
                img_path = row.get('image_path', 'N/A')
                status = row.get('processing_status', 'N/A')
                print(f"  - {os.path.basename(img_path) if img_path else 'N/A'}: {status}")

            return result
        else:
            print(f"  Unexpected return type: {type(result)}")
            return None

    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
        return None


def main():
    print("#" * 70)
    print("# CatLLM Image Classification Test")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 70)

    # Test single model
    single_result = test_single_model_image()

    # Test ensemble
    ensemble_result = test_ensemble_image()

    # Test directory input
    dir_result = test_directory_input()

    # Summary
    print("\n" + "#" * 70)
    print("# TEST SUMMARY")
    print("#" * 70)
    print(f"  Single model test: {'PASS' if single_result is not None else 'FAIL'}")
    print(f"  Ensemble test: {'PASS' if ensemble_result is not None else 'FAIL/SKIPPED'}")
    print(f"  Directory input test: {'PASS' if dir_result is not None else 'FAIL'}")


if __name__ == "__main__":
    main()
