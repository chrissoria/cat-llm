#!/usr/bin/env python
"""
Test script to verify all providers work for both image and PDF classification.

Run with: python examples/test_all_providers.py
"""

import sys
import os

# Add the src directory to path so we import from local code
src_path = '/Users/chrissoria/Documents/Research/cat-llm/src'
sys.path.insert(0, src_path)

# Remove any cached catllm modules
modules_to_remove = [key for key in sys.modules.keys() if key.startswith('catllm')]
for mod in modules_to_remove:
    del sys.modules[mod]

import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from catllm.text_functions_ensemble import multi_class_ensemble
import catllm

# Load API keys
load_dotenv('/Users/chrissoria/Documents/Research/cat-llm/.env')
load_dotenv('/Users/chrissoria/Documents/Research/Categorization_AI_experiments/.env')

# Get API keys
API_KEYS = {
    "openai": os.getenv("OPENAI_API_KEY"),
    "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    "google": os.getenv("GOOGLE_API_KEY"),
    "mistral": os.getenv("MISTRAL_API_KEY"),
    "xai": os.getenv("XAI_API_KEY"),
}

# Model configurations for each provider
PROVIDER_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-20241022",
    "google": "gemini-2.0-flash",
    "mistral": "mistral-small-latest",
    "xai": "grok-2-vision-1212",
}

# Test files
TEST_IMAGES = [
    "/Users/chrissoria/Documents/Research/cat-llm/tests/9688423031_065013996f_c.jpg",
    "/Users/chrissoria/Documents/Research/cat-llm/tests/9688419667_16c304583c_c.jpg",
]

TEST_PDF = "/Users/chrissoria/Documents/Research/cat-llm/tests/title_page.pdf"

# Categories
IMAGE_CATEGORIES = [
    "Contains people",
    "Outdoor scene",
    "Contains text",
]

PDF_CATEGORIES = [
    "Title page",
    "Contains author name",
    "Academic document",
]

print(f"Testing local catllm version: {catllm.__version__}")
print(f"Loaded from: {catllm.__file__}")
print()

# Check which providers have API keys
print("API keys loaded:")
for provider, key in API_KEYS.items():
    status = 'Y' if key else 'N'
    print(f"  {provider}: {status}")
print()


def test_provider_image(provider: str) -> dict:
    """Test a single provider with image input."""
    result = {
        "provider": provider,
        "mode": "image",
        "success": False,
        "error": None,
        "time": 0,
    }

    api_key = API_KEYS.get(provider)
    if not api_key:
        result["error"] = "No API key"
        return result

    model = PROVIDER_MODELS.get(provider)
    if not model:
        result["error"] = "No model configured"
        return result

    print(f"\n  Testing {provider} ({model}) with IMAGES...")

    try:
        import time
        start = time.time()

        df = multi_class_ensemble(
            survey_input=TEST_IMAGES,
            categories=IMAGE_CATEGORIES,
            models=[(model, provider, api_key)],
            input_description="Photographs",
            creativity=0.1,
            chain_of_thought=True,
            use_json_schema=True,
        )

        elapsed = time.time() - start
        result["time"] = elapsed

        if isinstance(df, pd.DataFrame) and len(df) > 0:
            statuses = df['processing_status'].tolist()
            if 'success' in statuses:
                result["success"] = True
                # Show sample result
                for idx, row in df.iterrows():
                    status = row.get('processing_status', 'N/A')
                    cat1 = row.get('category_1_' + model.replace('-', '_').replace('.', '_'), '?')
                    print(f"    Row {idx}: status={status}, category_1={cat1}")
            else:
                result["error"] = f"All failed: {statuses}"
        else:
            result["error"] = "Empty result"

    except Exception as e:
        import traceback
        result["error"] = str(e)
        print(f"    ERROR: {e}")

    return result


def test_provider_pdf(provider: str) -> dict:
    """Test a single provider with PDF input."""
    result = {
        "provider": provider,
        "mode": "pdf",
        "success": False,
        "error": None,
        "time": 0,
    }

    api_key = API_KEYS.get(provider)
    if not api_key:
        result["error"] = "No API key"
        return result

    model = PROVIDER_MODELS.get(provider)
    if not model:
        result["error"] = "No model configured"
        return result

    print(f"\n  Testing {provider} ({model}) with PDF...")

    try:
        import time
        start = time.time()

        df = multi_class_ensemble(
            survey_input=TEST_PDF,
            categories=PDF_CATEGORIES,
            models=[(model, provider, api_key)],
            input_description="Document pages",
            creativity=0.1,
            chain_of_thought=True,
            use_json_schema=True,
        )

        elapsed = time.time() - start
        result["time"] = elapsed

        if isinstance(df, pd.DataFrame) and len(df) > 0:
            statuses = df['processing_status'].tolist()
            if 'success' in statuses:
                result["success"] = True
                # Show sample result
                for idx, row in df.iterrows():
                    status = row.get('processing_status', 'N/A')
                    cat1_col = [c for c in df.columns if c.startswith('category_1_') and not c.endswith('_consensus') and not c.endswith('_agreement')]
                    cat1 = row.get(cat1_col[0], '?') if cat1_col else '?'
                    print(f"    Row {idx}: status={status}, category_1={cat1}")
            else:
                result["error"] = f"All failed: {statuses}"
        else:
            result["error"] = "Empty result"

    except Exception as e:
        import traceback
        result["error"] = str(e)
        print(f"    ERROR: {e}")

    return result


def main():
    print("#" * 70)
    print("# CatLLM All-Provider Test (Images & PDFs)")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 70)

    # Providers to test (those with API keys)
    providers_to_test = [p for p, k in API_KEYS.items() if k]

    print(f"\nProviders to test: {providers_to_test}")
    print(f"Test images: {len(TEST_IMAGES)}")
    print(f"Test PDF: {TEST_PDF}")

    results = []

    # Test each provider
    for provider in providers_to_test:
        print(f"\n{'='*60}")
        print(f"PROVIDER: {provider.upper()}")
        print(f"{'='*60}")

        # Test images
        img_result = test_provider_image(provider)
        results.append(img_result)

        # Test PDF
        pdf_result = test_provider_pdf(provider)
        results.append(pdf_result)

    # Summary
    print("\n" + "#" * 70)
    print("# TEST SUMMARY")
    print("#" * 70)

    print("\n{:<12} {:<8} {:<8} {:<8} {}".format(
        "Provider", "Mode", "Status", "Time", "Error"
    ))
    print("-" * 70)

    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        time_str = f"{r['time']:.1f}s" if r["time"] > 0 else "-"
        error = r.get("error", "") or ""
        if len(error) > 30:
            error = error[:30] + "..."
        print(f"{r['provider']:<12} {r['mode']:<8} {status:<8} {time_str:<8} {error}")

    # Count results
    passed = sum(1 for r in results if r["success"])
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")

    # Exit code
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
