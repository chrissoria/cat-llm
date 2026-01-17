#!/usr/bin/env python
"""
Comprehensive test script for all image functions with multiple providers.

Tests:
1. explore_image_categories - category extraction from images
2. classify_images - classify images into predefined categories
3. image_features - extract specific features/answers from images
"""

import sys
import os

# Add the src directory to path so we import from local code
src_path = '/Users/chrissoria/Documents/Research/cat-llm/src'
sys.path.insert(0, src_path)

# Clear any cached modules
modules_to_remove = [key for key in sys.modules.keys() if key.startswith('catllm')]
for mod in modules_to_remove:
    del sys.modules[mod]

from dotenv import load_dotenv

# Load API keys
load_dotenv('/Users/chrissoria/Documents/Research/Categorization_AI_experiments/.env')

# Get all API keys
api_keys = {
    "openai": os.getenv("OPENAI_API_KEY"),
    "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    "google": os.getenv("GOOGLE_API_KEY"),
    "mistral": os.getenv("MISTRAL_API_KEY"),
    "xai": os.getenv("XAI_API_KEY"),
}

print("=" * 70)
print("Comprehensive Image Functions Test")
print("=" * 70)
print()

# Print API key status
print("API Keys Status:")
for provider, key in api_keys.items():
    status = "Found" if key else "MISSING"
    print(f"  {provider}: {status}")
print()

# Create test image if it doesn't exist
test_image = "/Users/chrissoria/Documents/Research/cat-llm/tests/test_image.png"
if not os.path.exists(test_image):
    try:
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (200, 200), color='white')
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 20, 180, 180], outline='blue', width=3)
        draw.ellipse([50, 50, 150, 150], outline='red', width=2)
        draw.text((60, 90), "Test", fill='black')
        img.save(test_image)
        print(f"Created test image: {test_image}")
    except ImportError:
        print("PIL not installed, using existing test image")

print(f"Test image: {test_image}")
print()

# Import the functions
from catllm.image_functions import explore_image_categories, image_multi_class, image_features

# Vision-capable models for each provider
vision_models = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-haiku-20240307",
    "google": "gemini-2.0-flash",
    "mistral": "pixtral-12b-2409",
    "xai": "grok-2-vision-1212",
}

results = {"explore": [], "classify": [], "features": []}

# ============================================================================
# TEST 1: explore_image_categories
# ============================================================================
print("\n" + "=" * 70)
print("TEST 1: explore_image_categories")
print("=" * 70)

for provider, model in vision_models.items():
    print(f"\n{'-' * 50}")
    print(f"Provider: {provider} | Model: {model}")
    print(f"{'-' * 50}")

    if not api_keys.get(provider):
        print("  SKIPPED - No API key")
        results["explore"].append({"provider": provider, "success": None, "skipped": True})
        continue

    try:
        result = explore_image_categories(
            image_input=test_image,
            api_key=api_keys[provider],
            image_description="simple test image with geometric shapes",
            max_categories=5,
            categories_per_chunk=5,
            divisions=1,
            user_model=model,
            creativity=0.3,
            specificity="broad",
            mode="image",
            model_source=provider,
            iterations=1
        )

        if result and result.get('top_categories'):
            cats = result['top_categories'][:3]
            print(f"  Categories: {cats}")
            print(f"  PASS")
            results["explore"].append({"provider": provider, "success": True, "skipped": False})
        else:
            print(f"  FAIL - No categories extracted")
            results["explore"].append({"provider": provider, "success": False, "skipped": False})
    except Exception as e:
        print(f"  ERROR: {str(e)[:80]}")
        results["explore"].append({"provider": provider, "success": False, "skipped": False, "error": str(e)})

# ============================================================================
# TEST 2: image_multi_class (classification)
# ============================================================================
print("\n" + "=" * 70)
print("TEST 2: image_multi_class (classification)")
print("=" * 70)

test_categories = ["geometric shapes", "text content", "photograph", "chart", "logo"]

for provider, model in vision_models.items():
    print(f"\n{'-' * 50}")
    print(f"Provider: {provider} | Model: {model}")
    print(f"{'-' * 50}")

    if not api_keys.get(provider):
        print("  SKIPPED - No API key")
        results["classify"].append({"provider": provider, "success": None, "skipped": True})
        continue

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = image_multi_class(
                image_input=test_image,
                categories=test_categories,
                api_key=api_keys[provider],
                user_model=model,
                model_source=provider,
                image_description="test image with shapes and text",
                creativity=0.3,
            )

        if result is not None and len(result) > 0:
            # Get the classification result
            if hasattr(result, 'iloc'):
                classification = result.iloc[0].to_dict() if len(result) > 0 else {}
            else:
                classification = result[0] if isinstance(result, list) else result
            print(f"  Classification: {classification}")
            print(f"  PASS")
            results["classify"].append({"provider": provider, "success": True, "skipped": False})
        else:
            print(f"  FAIL - No classification result")
            results["classify"].append({"provider": provider, "success": False, "skipped": False})
    except Exception as e:
        print(f"  ERROR: {str(e)[:80]}")
        results["classify"].append({"provider": provider, "success": False, "skipped": False, "error": str(e)})

# ============================================================================
# TEST 3: image_features
# ============================================================================
print("\n" + "=" * 70)
print("TEST 3: image_features")
print("=" * 70)

test_questions = [
    "What color is the rectangle border?",
    "Is there text in the image?",
    "How many shapes are visible?",
]

for provider, model in vision_models.items():
    print(f"\n{'-' * 50}")
    print(f"Provider: {provider} | Model: {model}")
    print(f"{'-' * 50}")

    if not api_keys.get(provider):
        print("  SKIPPED - No API key")
        results["features"].append({"provider": provider, "success": None, "skipped": True})
        continue

    try:
        # image_features expects a list of file paths or a directory
        result = image_features(
            image_description="test image with shapes",
            image_input=[test_image],  # Pass as list
            features_to_extract=test_questions,
            api_key=api_keys[provider],
            user_model=model,
            model_source=provider,
            creativity=0.3,
        )

        if result is not None and len(result) > 0:
            # Get the features result
            if hasattr(result, 'iloc'):
                features = result.iloc[0].to_dict() if len(result) > 0 else {}
            else:
                features = result[0] if isinstance(result, list) else result
            print(f"  Features: {features}")
            print(f"  PASS")
            results["features"].append({"provider": provider, "success": True, "skipped": False})
        else:
            print(f"  FAIL - No features extracted")
            results["features"].append({"provider": provider, "success": False, "skipped": False})
    except Exception as e:
        print(f"  ERROR: {str(e)[:80]}")
        results["features"].append({"provider": provider, "success": False, "skipped": False, "error": str(e)})

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

total_passed = 0
total_failed = 0
total_skipped = 0

for test_name, test_results in results.items():
    passed = sum(1 for r in test_results if r.get('success') is True)
    failed = sum(1 for r in test_results if r.get('success') is False)
    skipped = sum(1 for r in test_results if r.get('skipped') is True)

    total_passed += passed
    total_failed += failed
    total_skipped += skipped

    print(f"\n{test_name}:")
    print(f"  Passed: {passed}, Failed: {failed}, Skipped: {skipped}")

    if failed > 0:
        print(f"  Failed providers:")
        for r in test_results:
            if r.get('success') is False:
                error = r.get('error', 'Unknown')[:50] if r.get('error') else 'No output'
                print(f"    - {r['provider']}: {error}")

print(f"\n{'=' * 70}")
print(f"TOTAL: Passed={total_passed}, Failed={total_failed}, Skipped={total_skipped}")
print(f"{'=' * 70}")

sys.exit(1 if total_failed > 0 else 0)
